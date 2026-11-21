#!/bin/bash
# The paired eICU comparison, after yaib_lean_check.sh PASSED for this revision: YAIB's
# LightGBM, then logistic regression, 25 runs each through the lean driver (YAIB's own
# code and published hyperparameters, without the copies only deep learning reads), each
# followed by the paired patient bootstrap against our 25 runs on the same test rows.
#
#   REV=<staged rev> systemd-run --unit aegis-lean-followup-$REV --collect \
#     --setenv REV=$REV /srv/aegis/launch/yaib_lean_followup.sh
#
# Rows (YAIB logs, predictions, compare logs) stay under /srv/lake; aggregates go to
# /srv/aegis/yaib-paired-$REV/{steps.log,paired-<model>.json}.
set -uo pipefail
REV=${REV:?staged source revision}
MODELS=${MODELS:-LGBMClassifier LogisticRegression}
OURS=/srv/lake/yaib-work/ours-eicu-fd9dd6f        # the scored 25 runs (final.json)
LOGS=/srv/lake/yaib-work/yaib-logs-full
PAIRED=/srv/aegis/yaib-paired-$REV
PRIVATE=/srv/lake/yaib-work/paired-$REV
COMPARE_SRC=${COMPARE_SRC:-/srv/aegis/sepsis-yaib-sweep-$REV}
COMPARE=${COMPARE:-backend.ml.scripts.yaib_paired_fast}
COMPRESS=${COMPRESS:-1}
mkdir -p -m 700 "$PAIRED" "$PRIVATE"
step() { echo "$(date -u +%FT%TZ) $*" >> "$PAIRED/steps.log"; }

python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["verdict"] != "PASS")' \
  /srv/aegis/yaib-lean-check-$REV/verdict.json || { step "REFUSED: the lean check did not pass for $REV"; exit 1; }
n_ours=$(ls "$OURS"/ours_r*_f*.parquet 2>/dev/null | wc -l)
[ "$n_ours" -eq 25 ] || { step "REFUSED: $n_ours of our 25 prediction files"; exit 1; }
step "lean check passed; our 25 runs present"
[ -f "$COMPARE_SRC/$(echo "${COMPARE%% *}" | tr . /).py" ] || { step "REFUSED: $COMPARE not in $COMPARE_SRC"; exit 1; }

for MODEL in $MODELS; do
  m=$(echo "$MODEL" | tr 'A-Z' 'a-z')
  dir=$LOGS/eicu/Sepsis/${MODEL}_published
  [ -e "$dir" ] && { step "REFUSED $MODEL: $dir already exists (a second run would duplicate runs)"; continue; }
  # 25 prediction files are ~3.8 GB on full eICU; running out a day in loses the day
  free=$(df -BM --output=avail / | tail -1 | tr -dc 0-9)
  [ "$free" -ge "${MIN_FREE_MB:-5000}" ] || { step "REFUSED $MODEL: ${free} MB free < ${MIN_FREE_MB:-5000} MB"; continue; }
  step "starting 25-run $MODEL (lean driver, ${free} MB free)"
  REV=$REV MODEL=$MODEL MODE=full WAIT_FORGE=0 HOURS=${HOURS:-72} \
    DRIVER=backend.ml.scripts.yaib_lean_baseline \
    BASELINE_MEMORY=7g BASELINE_MEMORY_SWAP=30g BASELINE_CPUS=2 \
    /srv/aegis/launch/yaib_baseline.sh
  rc=$?
  n=$(find "$dir" -name pred_indicators.csv 2>/dev/null | wc -l)
  step "$MODEL: exit $rc, $n of 25 runs with predictions"
  [ "$n" -eq 25 ] || continue
  # COMPARE: backend.ml.scripts.yaib_paired_fast (same statistic as `bench compare`, checked
  # by verify_yaib_paired_fast; ~8x faster, logs progress, reads .csv.gz). It needs a staged
  # source that has it; COMPARE_SRC=/srv/aegis/sepsis-yaib-sweep-fd9dd6f +
  # COMPARE="backend.ml.yaib.bench compare" reproduces the original call.
  /usr/local/sbin/aegis-heavy run yaib-paired-$m -- docker run --rm --network none --user 0 \
    --memory 7g --memory-swap 30g --cpus 2 --pids-limit 512 \
    --env PYTHONPATH=/src --env PYTHONDONTWRITEBYTECODE=1 \
    -v "$COMPARE_SRC":/src:ro -v "$OURS":/ours:ro \
    -v "$dir":/yaib:ro -v "$PAIRED":/out:rw \
    --entrypoint python aegis-yaib-sweep:fd9dd6f \
    -m $COMPARE --ours /ours --yaib /yaib \
    --out /out/paired-$m.json --n-boot 1000 > "$PRIVATE/compare-$m.log" 2>&1
  step "$MODEL paired: exit $?; $(cat "$PAIRED/paired-$m.json" 2>/dev/null | tr -d '\n ')"
  # the next model's 25 prediction files need the room (~3.8 GB each on full eICU); kept, compressed
  if [ "$COMPRESS" = 1 ] && [ -s "$PAIRED/paired-$m.json" ]; then
    find "$dir" -name pred_indicators.csv -print0 | xargs -0 -r -n1 nice -n 19 gzip -6
    step "$MODEL: predictions gzipped; $(df -BM --output=avail / | tail -1 | tr -d ' ') free"
  fi
done
step done
