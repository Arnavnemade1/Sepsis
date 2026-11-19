#!/bin/bash
# Gate for backend/ml/scripts/yaib_lean_baseline.py: stock YAIB and the lean driver, on
# YAIB's own (open) eICU demo cohort, must write byte-identical pred_indicators.csv for
# every run, for each ML baseline we intend to run on full eICU.
#
#   REV=<staged source rev> systemd-run --unit aegis-lean-check --collect /srv/aegis/launch/yaib_lean_check.sh
#
# Result: /srv/aegis/yaib-lean-check-$REV/{steps.log,verdict.json}. PASS needs, per model,
# the same set of runs in both arms, at least one run, and cmp-identical predictions.
set -uo pipefail
REV=${REV:?staged source revision (/srv/aegis/sepsis-yaib-sweep-$REV)}
MODELS=${MODELS:-LGBMClassifier LogisticRegression}
REPS=${REPS:-1}
SRC=/srv/aegis/sepsis-yaib-sweep-$REV
O=/srv/aegis/yaib-lean-check-$REV
W=/srv/lake/yaib-work/lean-check-$REV
rm -rf "$W"; mkdir -p -m 700 "$O" "$W"
: > "$O/steps.log"
step() { echo "$(date -u +%FT%TZ) $*" >> "$O/steps.log"; }

arm() {  # arm <stock|lean> <model>
  local module=backend.ml.yaib.run_baseline
  [ "$1" = lean ] && module=backend.ml.scripts.yaib_lean_baseline
  mkdir -p "$W/$1"
  docker run --rm --network none --user 0 --memory 6g --cpus 2 --pids-limit 512 \
    --env PYTHONDONTWRITEBYTECODE=1 --env PYTHONPATH=/src --env WANDB_MODE=disabled \
    --env OMP_NUM_THREADS=2 --env OPENBLAS_NUM_THREADS=2 \
    -v "$SRC":/src:ro -v "$W/$1":/logs:rw \
    -v "$SRC/backend/ml/yaib/gin/${2}_eicu_published.gin":/yaib/configs/prediction_models/${2}_published.gin:ro \
    --entrypoint python yaib:af9a5b3 -m $module train -d demo_data/sepsis/eicu_demo -n eicu_demo \
      -t BinaryClassification -tn Sepsis -m ${2}_published --seed 1234 -l /logs \
      -hp execute_repeated_cv.cv_repetitions_to_train=$REPS > "$W/$1-$2.log" 2>&1
}

verdict=PASS
summary=()
for M in $MODELS; do
  arm stock $M; rs=$?
  arm lean $M; rl=$?
  # runs keyed by repetition_x/fold_y (the timestamp directory differs between arms)
  list() { (cd "$W/$1" && find . -path "*${M}_published*" -name pred_indicators.csv | sed -E 's#.*/(repetition_[0-9]+/fold_[0-9]+)/pred_indicators.csv#\1#' | sort); }
  s=$(list stock); l=$(list lean); n=$(echo -n "$s" | grep -c . )
  same=0; diff=0
  if [ "$s" = "$l" ] && [ "$n" -gt 0 ]; then
    for r in $s; do
      a=$(cd "$W/stock" && find . -path "*${M}_published*/$r/pred_indicators.csv")
      b=$(cd "$W/lean" && find . -path "*${M}_published*/$r/pred_indicators.csv")
      if cmp -s "$W/stock/$a" "$W/lean/$b"; then same=$((same + 1)); else diff=$((diff + 1)); fi
    done
  fi
  lean_used=$(grep -c "LeanPredictionPolarsDataset" "$W/lean-$M.log")
  ok=PASS
  { [ $rs -ne 0 ] || [ $rl -ne 0 ] || [ "$s" != "$l" ] || [ "$n" -eq 0 ] || [ $diff -ne 0 ] || [ "$lean_used" -eq 0 ]; } && ok=FAIL
  [ $ok = FAIL ] && verdict=FAIL
  step "$M: stock exit $rs, lean exit $rl, runs $n, identical $same, different $diff, lean class logged $lean_used: $ok"
  summary+=("\"$M\": {\"stock_exit\": $rs, \"lean_exit\": $rl, \"runs\": $n, \"identical\": $same, \"different\": $diff, \"lean_class_logged\": $lean_used, \"verdict\": \"$ok\"}")
done
( IFS=,; echo "{\"rev\": \"$REV\", \"cohort\": \"eicu_demo\", \"verdict\": \"$verdict\", \"models\": {${summary[*]}}}" ) > "$O/verdict.json"
step "verdict $verdict"
[ $verdict = PASS ]
