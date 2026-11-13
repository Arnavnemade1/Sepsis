#!/bin/bash
# One YAIB baseline on the worker, run by YAIB's own code with the fixed published
# hyperparameters (backend/ml/yaib/gin/<MODEL>_eicu_published.gin), keeping per-hour test
# predictions for the paired comparison.
#
#   REV=<staged source rev> MODEL=LGBMClassifier MODE=probe \
#     systemd-run --unit aegis-baseline-<model>-<mode> --collect /srv/aegis/launch/yaib_baseline.sh
#
# MODE=probe fits only the first run (repetition 0, fold 0) and records wall time and peak
# memory: YAIB holds its generated features in RAM, and whether 25 runs fit this worker is
# decided from that measurement, not assumed. MODE=full runs all 25.
#
# The unit must not be named yaib-* / eicu-* (see yaib_conductor.sh). By default the script
# starts no container while the Forge optimization worker is waiting or running, because a
# container that merely queues for the heavy lock still keeps that worker from starting.
# Predictions and YAIB's logs (rows) stay in the lake; only aggregates and resource numbers
# go to /srv/aegis.
set -uo pipefail
REV=${REV:?staged source revision (/srv/aegis/sepsis-yaib-sweep-$REV)}
MODEL=${MODEL:?LGBMClassifier or LogisticRegression}
MODE=${MODE:-probe}
COHORT=${COHORT:-eicu}
SRC=/srv/aegis/sepsis-yaib-sweep-$REV
GIN=$SRC/backend/ml/yaib/gin/${MODEL}_${COHORT}_published.gin
NAME=baseline-$(echo "$MODEL" | tr 'A-Z' 'a-z')-$MODE
O=/srv/aegis/yaib-$NAME
LOGS=/srv/lake/yaib-work/yaib-logs-$MODE          # rows: inside the lake
FORGE_STATE=/srv/aegis/forge-sepsis-runs/optimization-v3/worker-state.json
mkdir -p -m 700 "$O" "$LOGS"
step() { echo "$(date -u +%FT%TZ) $*" >> "$O/steps.log"; }
forge() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status","?"))' $FORGE_STATE 2>/dev/null || echo unreadable; }
[ -f "$GIN" ] || { step "FAILED: no $GIN"; exit 1; }

step "start $MODEL $MODE (source $REV)"
if [ "${WAIT_FOR:-}" ]; then
  while systemctl is-active -q "$WAIT_FOR"; do sleep 60; done
  step "$WAIT_FOR finished"
fi
if [ "${WAIT_FORGE:-1}" = 1 ]; then
  sleep 120                                   # let the Forge worker see the box is free
  waited=0
  while { [ "$(forge)" = waiting ] || [ "$(forge)" = running ]; } && [ $waited -lt 30000 ]; do sleep 60; waited=$((waited + 60)); done
  step "forge status after ${waited}s: $(forge)"
fi

EXTRA=()
[ "$MODE" = probe ] && EXTRA=(-hp execute_repeated_cv.cv_repetitions_to_train=1 execute_repeated_cv.cv_folds_to_train=1)
HOURS=${HOURS:-$([ "$MODE" = probe ] && echo 8 || echo 120)}
( peak=0; swap=0
  while sleep 30; do
    m=$(docker stats --no-stream --format '{{.MemUsage}}' yaib-$NAME 2>/dev/null | awk '{print $1}') || true
    [ -z "$m" ] && continue
    mb=$(python3 -c 'import sys; v=sys.argv[1]; print(int(float(v[:-3]) * (1024 if v.endswith("GiB") else 1)))' "$m" 2>/dev/null || echo 0)
    s=$(free -m | awk '/^Swap/ {print $3}')
    [ "$mb" -gt "$peak" ] && peak=$mb; [ "$s" -gt "$swap" ] && swap=$s
    echo "{\"peak_container_mb\": $peak, \"peak_host_swap_mb\": $swap}" > "$O/resources.json"
  done ) &
SAMPLER=$!
t0=$(date +%s)
docker run --name yaib-$NAME --rm --network none --user 0 \
  --memory "${BASELINE_MEMORY:-7g}" --memory-swap "${BASELINE_MEMORY_SWAP:-30g}" \
  --cpus "${BASELINE_CPUS:-2}" --pids-limit 512 \
  --tmpfs /heavy:rw,nosuid,nodev,size=1m \
  --mount type=bind,source=/run/lock/aegis-heavy.lock,target=/heavy/shared.lock \
  --mount type=bind,source=/usr/local/sbin/aegis-heavy,target=/opt/aegis-heavy,readonly \
  --env AEGIS_HEAVY_LOCK=/heavy/shared.lock --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/src \
  --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 --env WANDB_MODE=disabled \
  -v /srv/lake/yaib-cohorts:/cohorts:ro -v "$LOGS":/logs:rw -v "$SRC":/src:ro -v "$O":/output:rw \
  -v "$GIN":/yaib/configs/prediction_models/${MODEL}_published.gin:ro \
  --entrypoint sh yaib:af9a5b3 -c 'umask 077; exec "$@" > /output/private-run.log 2>&1' baseline \
  /opt/aegis-heavy run yaib-$NAME -- timeout --signal=TERM --kill-after=30s ${HOURS}h \
  python -m backend.ml.yaib.run_baseline train -d /cohorts/sepsis/$COHORT -n $COHORT \
    -t BinaryClassification -tn Sepsis -m ${MODEL}_published --seed 1234 -l /logs "${EXTRA[@]}"
rc=$?
kill $SAMPLER 2>/dev/null
secs=$(( $(date +%s) - t0 ))
# YAIB's own aggregate (means and its "std" field over the runs it trained): an aggregate, no rows
agg=$(find "$LOGS/$COHORT/Sepsis/${MODEL}_published" -name aggregated_test_metrics.json 2>/dev/null | sort | tail -1)
[ -n "$agg" ] && cp "$agg" "$O/aggregated_test_metrics.json"
runs=$(find "$LOGS/$COHORT/Sepsis/${MODEL}_published" -name pred_indicators.csv 2>/dev/null | wc -l)
step "exit $rc after ${secs}s; runs with predictions: $runs; $(cat "$O/resources.json" 2>/dev/null)"
exit $rc
