#!/bin/bash
# Checks aegis-heavy on a Linux host with flock and systemd (the worker), on a private lock
# file so it never touches the real one: `sudo bash deploy/physionet/test_heavy.sh [heavy.sh]`.
set -uo pipefail
H="${1:-$(dirname "$0")/heavy.sh}"
export AEGIS_HEAVY_LOCK="$(mktemp -u /run/lock/aegis-heavy-test.XXXXXX)"
T="$(mktemp -d)"
trap 'rm -rf "$T" "$AEGIS_HEAVY_LOCK" "$AEGIS_HEAVY_LOCK.holder"' EXIT
fail=0
check() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }

check "free when nobody holds it" '[[ "$(bash "$H" status)" == free ]]'

bash "$H" run A -- bash -c "sleep 3; date +%s.%N > $T/a_end" & A=$!
sleep 0.5
check "status names the holder" '[[ "$(bash "$H" status)" == "busy: A pid="* ]]'
bash "$H" run B --no-wait -- touch "$T/b_ran"; rc=$?
check "--no-wait refuses with 75 while busy" '[[ $rc == 75 ]]'
check "--no-wait did not run its command" '[[ ! -e $T/b_ran ]]'
bash "$H" run C -- bash -c "date +%s.%N > $T/c_start"
check "a waiting job starts only after the holder ends" \
  '[[ -s $T/a_end && -s $T/c_start ]] && awk "BEGIN{exit !($(cat $T/c_start) >= $(cat $T/a_end))}"'
wait $A
check "free again afterwards" '[[ "$(bash "$H" status)" == free ]]'

bash "$H" run D -- false; rc=$?
check "the command's exit code comes back" '[[ $rc == 1 ]]'

setsid bash "$H" run E -- bash -c 'sleep 1; sleep 100' & sleep 0.5
check "a child keeps the lock" '[[ "$(bash "$H" status)" == busy* ]]'
pkill -9 -f "sleep 100"; pkill -9 -f "run E --"; sleep 0.5
check "killing the holder releases the lock" '[[ "$(bash "$H" status)" == free ]]'

U="heavytest-$$"
systemd-run --quiet --unit "$U" sleep 4
bash "$H" hold-while "$U.service" & W=$!
sleep 1
check "hold-while holds while the unit is active" '[[ "$(bash "$H" status)" == "busy: hold:$U.service"* ]]'
wait $W
check "hold-while lets go when the unit ends" '[[ "$(bash "$H" status)" == free ]]'

exit $fail
