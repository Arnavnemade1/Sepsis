#!/bin/bash
# Box-side watchdog for the paired YAIB chain (needs no laptop login). Every 5 min: record
# status, write alerts, and take only pre-approved non-destructive disk actions.
#
#   REV=<rev> systemd-run --unit aegis-yaib-guardian --on-calendar='*:0/5' \
#     --setenv REV=<rev> /srv/aegis/launch/yaib_guardian.sh
#
# Reads /srv/aegis/guardian/{status.json,alerts.log}; alerts repeat at most hourly per kind.
set -u
REV=${REV:?staged source revision of the running chain}
P=/srv/aegis/yaib-paired-$REV; G=${GUARDIAN_DIR:-/srv/aegis/guardian}; mkdir -p $G
L=/srv/lake/yaib-work/yaib-logs-full/eicu/Sepsis
now=$(date -u +%FT%TZ)
alert() { k=$1; shift; f=$G/.last-$k; t=$(date +%s)
  [ -f $f ] && [ $((t - $(cat $f))) -lt 3600 ] && return; echo $t > $f
  echo "$now ALERT $k: $*" >> $G/alerts.log; }
free_mb() { df -BM --output=avail / | tail -1 | tr -dc 0-9; }
fr=$(free_mb)
chain=$(systemctl is-active aegis-lean-followup-$REV)
conts=$(docker ps --format '{{.Names}}' | paste -sd, -)
swap=$(free -m | awk '/Swap/{print $3}')
lg_csv=$(find $L/LGBMClassifier_published -name pred_indicators.csv 2>/dev/null | wc -l)
lg_gz=$(find $L/LGBMClassifier_published -name pred_indicators.csv.gz 2>/dev/null | wc -l)
lr=$(find $L/LogisticRegression_published -name pred_indicators.csv 2>/dev/null | wc -l)
last=$(tail -1 $P/steps.log)
# disk: harmless cleanups first, then compressing LGBM predictions (only once their paired json exists)
if [ "$fr" -lt 1500 ]; then
  journalctl --vacuum-size=20M -q; dnf clean all -q; docker builder prune -f >/dev/null 2>&1
  if [ "$(free_mb)" -lt 1500 ] && [ -s $P/paired-lgbmclassifier.json ] && [ "$lg_csv" -gt 0 ]; then
    find $L/LGBMClassifier_published -name pred_indicators.csv -print0 | xargs -0 -n1 nice -n 19 gzip -6
    echo "$now guardian: emergency-compressed LGBM preds" >> $P/steps.log
  fi
  fr2=$(free_mb); alert disk "free ${fr}MB -> ${fr2}MB after safe cleanup"; fr=$fr2
fi
[ "$fr" -lt 800 ] && alert disk_critical "only ${fr}MB free; LR prediction writes may fail"
if [ "$chain" != active ]; then
  grep -q ' done$' $P/steps.log && alert finished "chain finished: $last" || alert chain_stopped "chain unit $chain; last step: $last"
fi
grep -qE 'exit [1-9]|REFUSED' $P/steps.log && alert step_failed "$(grep -E 'exit [1-9]|REFUSED' $P/steps.log | tail -1)"
[ "$swap" -gt 27000 ] && alert swap "swap ${swap}MB used of 30g container cap"
printf '{"at":"%s","free_mb":%s,"chain":"%s","containers":"%s","swap_mb":%s,"lgbm_csv":%s,"lgbm_gz":%s,"lr_runs_done":%s,"last_step":"%s"}\n' \
  "$now" "$fr" "$chain" "$conts" "$swap" "$lg_csv" "$lg_gz" "$lr" "$(echo "$last" | tr -d '"' | cut -c1-300)" > $G/status.json
