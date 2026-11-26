#!/bin/bash
# After the paired LGBM bootstrap wrote its aggregate, gzip YAIB LGBM's per-run predictions
# (kept, just compressed) to leave disk room for the LR run's 25 prediction files.
# `yaib_paired_fast` reads them as they are; `bench compare` needs them gunzipped first.
# For a chain started before yaib_lean_followup.sh compressed by itself (REV a76fe61):
#   REV=<rev> MODEL=<YAIB model> systemd-run --unit aegis-compress-<model> --setenv REV=.. --setenv MODEL=.. <this>
set -u
REV=${REV:?staged source revision of the running chain}; MODEL=${MODEL:-LGBMClassifier}
m=$(echo "$MODEL" | tr 'A-Z' 'a-z')
P=/srv/aegis/yaib-paired-$REV
D=/srv/lake/yaib-work/yaib-logs-full/eicu/Sepsis/${MODEL}_published
while [ ! -s $P/paired-$m.json ]; do
  systemctl is-active -q aegis-lean-followup-$REV || { echo "$(date -u +%FT%TZ) compress: chain ended without paired json; not compressing" >> $P/steps.log; exit 1; }
  sleep 60
done
sleep 30
before=$(df -BM --output=avail / | tail -1)
find $D -name pred_indicators.csv -print0 | xargs -0 -n1 nice -n 19 gzip -6
n=$(find $D -name 'pred_indicators.csv.gz' | wc -l)
echo "$(date -u +%FT%TZ) compress: $n $MODEL pred files gzipped; free ${before} -> $(df -BM --output=avail / | tail -1)" >> $P/steps.log
