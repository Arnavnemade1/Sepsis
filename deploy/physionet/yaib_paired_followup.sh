#!/bin/bash
# Continue the eICU paired comparison after a bounded, single-fold YAIB memory probe.
# Run on the credentialed worker. All row-level outputs stay under /srv/lake.
set -euo pipefail

PROBE_UNIT=${PROBE_UNIT:-aegis-baseline-lgbm-highmem-probe}
BASELINE=/srv/aegis/launch/yaib_baseline_highmem.sh
PROBE=/srv/aegis/yaib-baseline-lgbmclassifier-probe
FULL=/srv/aegis/yaib-baseline-lgbmclassifier-full
FULL_LOGS=/srv/lake/yaib-work/yaib-logs-full
PAIRED=/srv/aegis/yaib-paired-fd9dd6f
PRIVATE=/srv/lake/yaib-work/paired-fd9dd6f
mkdir -p -m 700 "$PAIRED" "$PRIVATE"
step() { echo "$(date -u +%FT%TZ) $*" >> "$PAIRED/steps.log"; }

step "waiting for $PROBE_UNIT"
while systemctl is-active -q "$PROBE_UNIT"; do sleep 60; done
test -f "$PROBE/steps.log"
grep -Eq 'exit 0 after [0-9]+s; runs with predictions: 1;' "$PROBE/steps.log"
test ! -e "$FULL"
test ! -e "$FULL_LOGS"
test "$(df -BG / | awk 'NR==2 {gsub("G", "", $4); print $4}')" -ge 4
step 'one-fold probe passed; starting 25-run YAIB LightGBM'

REV=fd9dd6f MODEL=LGBMClassifier MODE=full WAIT_FORGE=0 HOURS=72 \
  BASELINE_MEMORY=7g BASELINE_MEMORY_SWAP=38g BASELINE_CPUS=2 \
  "$BASELINE"
test "$(find "$FULL_LOGS" -name pred_indicators.csv | wc -l)" -eq 25
step '25 baseline predictions present; starting paired bootstrap'

/usr/local/sbin/aegis-heavy run yaib-paired-compare -- docker run --rm --network none --user 0 \
  --memory 7g --memory-swap 38g --cpus 2 --pids-limit 512 \
  --env PYTHONPATH=/src --env PYTHONDONTWRITEBYTECODE=1 \
  -v /srv/aegis/sepsis-yaib-sweep-fd9dd6f:/src:ro \
  -v /srv/lake/yaib-work/ours-eicu-fd9dd6f:/ours:ro \
  -v "$FULL_LOGS":/yaib:ro -v "$PAIRED":/out:rw \
  --entrypoint python aegis-yaib-sweep:fd9dd6f \
  -m backend.ml.yaib.bench compare --ours /ours --yaib /yaib \
  --out /out/paired.json --n-boot 1000 > "$PRIVATE/compare.log" 2>&1
test -s "$PAIRED/paired.json"
step 'paired aggregate ready'
