#!/bin/bash
# One heavy job at a time on the worker, enforced by the kernel instead of by agreement.
#
#   aegis-heavy run <name> [--no-wait] -- <command...>   run while holding the box's one heavy-job lock
#   aegis-heavy hold-while <unit>                         hold it while a systemd unit started without it is active
#   aegis-heavy status                                    "free", or the holder's name, pid and start time
#
# The lock is flock(2) on a file in /run/lock, so it is released when its holder exits or is
# killed, and a reboot clears it. `run` waits for the lock by default, so heavy jobs queue in
# order; --no-wait exits 75 (EX_TEMPFAIL) at once when the box is busy. Children inherit the
# lock: a pipeline that starts several stages holds it across all of them. A `docker run` must
# stay in the foreground (no -d), or the container outlives the lock.
set -uo pipefail

LOCK="${AEGIS_HEAVY_LOCK:-/run/lock/aegis-heavy.lock}"
HOLDER="$LOCK.holder"

usage() { sed -n '4,6p' "$0" | sed 's/^# *//' >&2; exit 64; }

status() {
  exec 9>>"$LOCK"
  if flock -n 9; then
    echo free
  else
    echo "busy: $(cat "$HOLDER" 2>/dev/null || echo 'holder unknown')"
  fi
}

run() {
  local name="${1:-}"; shift || true
  local mode=""
  [[ "${1:-}" == "--no-wait" ]] && { mode="-n"; shift; }
  [[ -n "$name" && "${1:-}" == "--" ]] || usage
  shift
  [[ $# -gt 0 ]] || usage
  exec 9>>"$LOCK"
  if ! flock $mode 9; then
    echo "aegis-heavy: busy, not starting '$name': $(cat "$HOLDER" 2>/dev/null)" >&2
    exit 75
  fi
  printf '%s pid=%s since=%s\n' "$name" "$$" "$(date -u +%FT%TZ)" > "$HOLDER"
  trap 'rm -f "$HOLDER"' EXIT
  "$@"
}

case "${1:-}" in
  run) shift; run "$@" ;;
  hold-while)
    [[ -n "${2:-}" ]] || usage
    run "hold:$2" -- bash -c 'while systemctl is-active -q "$1"; do sleep 20; done' _ "$2" ;;
  status) status ;;
  *) usage ;;
esac
