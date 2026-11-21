#!/bin/bash
# After the paired LGBM bootstrap wrote its aggregate, gzip YAIB LGBM's per-run predictions
# (kept, just compressed) to leave disk room for the LR run's 25 prediction files.
# Re-running `bench compare` on LGBM needs them gunzipped first.
set -u
P=/srv/aegis/yaib-paired-a76fe61
D=/srv/lake/yaib-work/yaib-logs-full/eicu/Sepsis/LGBMClassifier_published
while [ ! -s $P/paired-lgbmclassifier.json ]; do
  systemctl is-active -q aegis-lean-followup-a76fe61 || { echo "$(date -u +%FT%TZ) compress: chain ended without paired json; not compressing" >> $P/steps.log; exit 1; }
  sleep 60
done
sleep 30
before=$(df -BM --output=avail / | tail -1)
find $D -name pred_indicators.csv -print0 | xargs -0 -n1 nice -n 19 gzip -6
n=$(find $D -name 'pred_indicators.csv.gz' | wc -l)
echo "$(date -u +%FT%TZ) compress: $n LGBM pred files gzipped; free ${before} -> $(df -BM --output=avail / | tail -1)" >> $P/steps.log
