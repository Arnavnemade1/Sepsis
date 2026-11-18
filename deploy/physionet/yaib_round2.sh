#!/bin/bash
# Round 2 of the eICU benchmark: a registered validation-only sweep of extension candidates
# against round 1's chosen configuration, then the 25 scored runs ONLY if a candidate
# displaces it. Run on the credentialed worker after the paired follow-up has finished:
#
#   REV=<this source rev> WAIT_FOR=<paired follow-up unit, or none> systemd-run --unit aegis-round2-$REV --collect /srv/aegis/launch/yaib_round2.sh
#
# The unit must not be named yaib-* / eicu-* (see yaib_conductor.sh). This script holds no
# heavy lock; each container takes it, so every step queues behind whatever holds the box.
#
# Environment: the round-1 image (aegis-yaib-sweep:$IMAGE) runs the new source, mounted
# read-only at /src. Same Python and libraries, so the store's 17.8 GB rich matrix keeps its
# identity and is reused, not rebuilt; only the extension matrix is built.
# Aggregates and this log go to /srv/aegis; rows never leave the lake.
set -uo pipefail
REV=${REV:?source revision staged at /srv/aegis/sepsis-yaib-r2-$REV}
IMAGE=${IMAGE:-fd9dd6f}
COHORT=${COHORT:-eicu}
SRC=/srv/aegis/sepsis-yaib-r2-$REV
PLAN=$SRC/artifacts/yaib/eicu_round2_prereg.json
O=/srv/aegis/yaib-round2-$REV
WORK=/work/sweep-$COHORT-r2-$REV
LOG=$O/round2.log
mkdir -p -m 700 "$O"
step() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
[ -d "$SRC/backend/ml/yaib" ] && [ -f "$PLAN" ] || { step "FAILED: stage $REV first (git archive into $SRC)"; exit 1; }
[ -f "$SRC/backend/ml/yaib/extension.py" ] || { step "FAILED: $SRC predates the extension"; exit 1; }

# The paired follow-up releases the lock between its 25 baseline runs and its comparison;
# name its unit so this sweep cannot take the box in that gap (WAIT_FOR=none if it is done).
WAIT_FOR=${WAIT_FOR:?the paired follow-up unit to wait for, or none}
step "round 2 start (source $REV, image $IMAGE)"
if [ "$WAIT_FOR" != none ]; then
  while systemctl is-active -q "$WAIT_FOR"; do sleep 60; done
  step "$WAIT_FOR finished"
fi
free_gb=$(df -BG /srv/lake | awk 'NR==2 {gsub("G", "", $4); print $4}')
step "free disk on the lake: ${free_gb} GB"

container() {   # name, then the python module and its arguments
  local name=$1; shift
  docker run --name "$name" --rm --network none --user 0 \
    --memory 7g --memory-swap 30g --cpus 2 --pids-limit 256 --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=512m --tmpfs /heavy:rw,nosuid,nodev,size=1m \
    --mount type=bind,source=/run/lock/aegis-heavy.lock,target=/heavy/shared.lock \
    --mount type=bind,source=/usr/local/sbin/aegis-heavy,target=/opt/aegis-heavy,readonly \
    --env AEGIS_HEAVY_LOCK=/heavy/shared.lock --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/src \
    --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
    -v /srv/lake/yaib-cohorts:/cohorts:ro -v /srv/lake/yaib-work:/work:rw -v "$SRC":/src:ro -v "$O":/output:rw \
    --entrypoint sh "aegis-yaib-sweep:$IMAGE" -c 'umask 077; exec "$@" >> /output/private-run.log 2>&1' "$name" \
    /opt/aegis-heavy run "$name" -- timeout --signal=TERM --kill-after=30s "$HOURS"h python -m "$@"
}

# ---- the registered sweep: validation AUROC only, no test metric of any candidate
HOURS=40 container aegis-r2-sweep-$REV backend.ml.scripts.yaib_sweep /cohorts/sepsis/$COHORT --plan /src/artifacts/yaib/eicu_round2_prereg.json \
  --work "$WORK" --feature-store /work/feature-store --out /output --threads 2 --fit-timeout-hours 8 --max-hours 36
rc=$?
[ -s "$O/selection.json" ] || { step "FAILED: sweep exit $rc without selection.json"; exit 1; }
chosen=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' "$O/selection.json")
reference=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["plan"]["reference"]["name"])' "$O/selection.json")
step "sweep exit $rc: $(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("chosen", s["chosen"], "mean validation AUROC", round(s["chosen_mean_val_auroc"], 4), "lead", round(s["lead_over_reference"], 4), "failed", sorted(s["failed"]))' "$O/selection.json")"
if [ "$chosen" = "$reference" ]; then
  step "the round-1 configuration keeps its place; its 25-run result stands and nothing is rescored"
  exit 0
fi

# ---- the 25 scored runs of the chosen candidate, resuming its two development fits
HOURS=120 container aegis-r2-final-$REV backend.ml.scripts.yaib_final /cohorts/sepsis/$COHORT --selection /output/selection.json \
  --work "$WORK" --feature-store /work/feature-store --predictions /work/ours-$COHORT-r2-$REV \
  --out /output/final.json --threads 2
rc=$?
step "final exit $rc$([ -s "$O/final.json" ] && echo ', final.json written')"
exit $rc
