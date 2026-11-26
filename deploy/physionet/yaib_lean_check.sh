#!/bin/bash
# Gate for backend/ml/scripts/yaib_lean_baseline.py: stock YAIB and the lean driver, on
# YAIB's own (open) eICU demo cohort, must write the same pred_indicators.csv for every
# run, for each ML baseline we intend to run on full eICU.
#
# "The same" is everything backend.ml.yaib.bench compare reads: the label and both
# probabilities byte-identical row by row, and the identical set of (stay, hour) ids.
# The id columns' ORDER is not compared: YAIB af9a5b3 writes them from the frame before
# its sort, and that frame comes out of an unordered .unique(), so their order changes
# between two stock runs (a second stock arm measures that, and is reported). bench
# pairs rows by sorted position for exactly this reason (see bench._join).
#
#   REV=<staged source rev> systemd-run --unit aegis-lean-check --collect /srv/aegis/launch/yaib_lean_check.sh
#
# Result: /srv/aegis/yaib-lean-check-$REV/{steps.log,verdict.json}. PASS needs, per model,
# the same set of runs in both arms, at least one run, and identical values as above.
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

arm() {  # arm <stock|stock2|lean> <model>
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
  arm stock2 $M
  arm lean $M; rl=$?
  # runs keyed by repetition_x/fold_y (the timestamp directory differs between arms)
  list() { (cd "$W/$1" && find . -path "*${M}_published*" -name pred_indicators.csv | sed -E 's#.*/(repetition_[0-9]+/fold_[0-9]+)/pred_indicators.csv#\1#' | sort); }
  s=$(list stock); l=$(list lean); n=$(echo -n "$s" | grep -c . )
  same=0; diff=0; stock_raw_same=0
  pred() { echo "$W/$1/$(cd "$W/$1" && find . -path "*${M}_published*/$2/pred_indicators.csv")"; }
  if [ "$s" = "$l" ] && [ "$n" -gt 0 ]; then
    for r in $s; do
      a=$(pred stock $r); b=$(pred lean $r); c=$(pred stock2 $r)
      if cmp -s <(cut -d, -f3-5 "$a") <(cut -d, -f3-5 "$b") \
         && cmp -s <(cut -d, -f1-2 "$a" | LC_ALL=C sort) <(cut -d, -f1-2 "$b" | LC_ALL=C sort); then
        same=$((same + 1)); else diff=$((diff + 1)); fi
      cmp -s "$a" "$c" && stock_raw_same=$((stock_raw_same + 1))
    done
  fi
  lean_used=$(grep -c "LeanPredictionPolarsDataset" "$W/lean-$M.log")
  ok=PASS
  { [ $rs -ne 0 ] || [ $rl -ne 0 ] || [ "$s" != "$l" ] || [ "$n" -eq 0 ] || [ $diff -ne 0 ] || [ "$lean_used" -eq 0 ]; } && ok=FAIL
  [ $ok = FAIL ] && verdict=FAIL
  step "$M: stock exit $rs, lean exit $rl, runs $n, identical $same, different $diff, lean class logged $lean_used, two stock runs byte-identical $stock_raw_same of $n: $ok"
  summary+=("\"$M\": {\"stock_exit\": $rs, \"lean_exit\": $rl, \"runs\": $n, \"identical\": $same, \"different\": $diff, \"lean_class_logged\": $lean_used, \"stock_vs_stock_raw_identical\": $stock_raw_same, \"verdict\": \"$ok\"}")
done
( IFS=,; echo "{\"rev\": \"$REV\", \"cohort\": \"eicu_demo\", \"verdict\": \"$verdict\", \"models\": {${summary[*]}}}" ) > "$O/verdict.json"
step "verdict $verdict"
[ $verdict = PASS ]
