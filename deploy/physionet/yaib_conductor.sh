#!/bin/bash
# After a YAIB sweep on the worker: lend the box to the Forge optimization worker for its
# one bounded job, then run all 25 runs with the configuration the sweep chose.
#
#   SWEEP=<sweep source rev> FINAL=<final image source rev> \
#     systemd-run --unit aegis-conductor-$FINAL --collect /srv/aegis/launch/yaib_conductor.sh
#
# The unit must NOT be named yaib-* or eicu-*: the Forge worker treats those as the
# benchmark still running. This script holds no heavy lock itself. Every heavy step takes
# the lock inside its own container; the final run queues behind whoever holds it.
# Aggregates and this log go to /srv/aegis; rows never leave the lake.
set -uo pipefail
SWEEP=${SWEEP:?source revision of the sweep}
FINAL=${FINAL:?source revision of the final-run image}
COHORT=${COHORT:-eicu}
O=/srv/aegis/yaib-sweep-$SWEEP
F=/srv/aegis/yaib-final-$FINAL
LOG=$O/conductor.log
RELEASE=/srv/aegis/forge-sepsis-optimization-release.json
FORGE_STATE=/srv/aegis/forge-sepsis-runs/optimization-v3/worker-state.json
step() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
forge() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","?"))' $FORGE_STATE 2>/dev/null || echo unreadable; }

step "conductor start (sweep $SWEEP, final $FINAL)"
while systemctl is-active -q yaib-sweep-$SWEEP; do sleep 60; done
[ -s $O/selection.json ] || { step "FAILED: the sweep ended without selection.json"; exit 1; }
step "sweep finished: $(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("chosen", s["chosen"], "mean validation AUROC", round(s["chosen_mean_val_auroc"], 4), "lead", round(s["lead_over_reference"], 4), "failed", sorted(s["failed"]))' $O/selection.json)"

# ---- lend the box to Forge: it starts only with the marker, no yaib-*/eicu-* unit, no
# ---- container running and the heavy lock free; its job is capped at six hours
if [ "${LEND_TO_FORGE:-1}" = 1 ] && [ "$(forge)" = waiting ]; then
  if [ ! -e $RELEASE ]; then
    printf '{"released": true, "target": "sepsis-optimization-v3"}\n' > $RELEASE.tmp && mv $RELEASE.tmp $RELEASE
    step "released the box to forge-sepsis-optimization-v3"
  fi
  for _ in $(seq 1 30); do [ "$(forge)" != waiting ] && break; sleep 30; done
  step "forge status after release: $(forge)"
  waited=0
  while [ "$(forge)" = running ] && [ $waited -lt 25200 ]; do sleep 60; waited=$((waited + 60)); done
  step "forge status after ${waited}s: $(forge)"
else
  step "forge not waiting ($(forge)); nothing lent"
fi

# ---- all 25 runs, resuming the sweep's fits; queues behind the heavy lock
mkdir -p -m 700 $F
step "final 25 runs queued"
docker run --name yaib-final-$FINAL --rm --network none --user 0 \
  --memory 7g --memory-swap 30g --cpus 2 --pids-limit 256 --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,size=512m --tmpfs /heavy:rw,nosuid,nodev,size=1m \
  --mount type=bind,source=/run/lock/aegis-heavy.lock,target=/heavy/shared.lock \
  --mount type=bind,source=/usr/local/sbin/aegis-heavy,target=/opt/aegis-heavy,readonly \
  --env AEGIS_HEAVY_LOCK=/heavy/shared.lock --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
  -v /srv/lake/yaib-cohorts:/cohorts:ro -v /srv/lake/yaib-work:/work:rw -v $O:/sweep:ro -v $F:/output:rw \
  --entrypoint sh aegis-yaib-sweep:$FINAL -c 'umask 077; exec "$@" > /output/private-run.log 2>&1' final \
  /opt/aegis-heavy run yaib-final-$FINAL -- timeout --signal=TERM --kill-after=30s 432000s \
  python -m backend.ml.scripts.yaib_final /cohorts/sepsis/$COHORT --selection /sweep/selection.json \
    --work /work/sweep-$COHORT-$SWEEP --feature-store /work/feature-store \
    --predictions /work/ours-$COHORT-$FINAL --out /output/final.json --threads 2
rc=$?
step "final exit $rc$([ -s $F/final.json ] && echo ', final.json written')"
exit $rc
