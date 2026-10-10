#!/usr/bin/env bash
# Safety guard for a serving or benchmark session (run as root from the repo root, in its own
# terminal or with setsid; stop it with bash loop/stop_guard.sh):
#   bash loop/guard.sh
#  - kernel log: Xid / AER / soft lockup / GPU off the bus  => docker stop the containers, write HALT
#  - optional wall power: if POWER_CMD and KILL_W are set, sample every second and docker kill
#    the containers above KILL_W watts, write HALT
# run_candidate.sh, queue.sh and devlock.sh refuse to start while the guard is down or a HALT
# file exists. It never touches power limits or clocks and never reboots.
#
# Env:
#   KILL_W      wall-power trip in watts (empty = no power trip)
#   POWER_CMD   prints the instantaneous wall power, e.g. via the BMC:
#               POWER_CMD="ipmitool dcmi power reading"   (parsed: 'Instantaneous ... N Watts')
#   CONTAINERS  containers to stop/kill (default: glm53-vllm)
set -uo pipefail
KILL_W=${KILL_W:-}
POWER_CMD=${POWER_CMD:-}
CONTAINERS=${CONTAINERS:-"glm53-vllm"}
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=${LOG:-$HERE/../logs/guard}
mkdir -p "$LOG"

halt() {
  echo "$(date -Is) HALT: $*" | tee -a "$LOG/HALT"
  for c in $CONTAINERS; do docker "$1" "$c" >/dev/null 2>&1 || true; done
}

journalctl -kf -n0 -o short-iso 2>/dev/null | while read -r line; do
  if echo "$line" | grep -qE 'Xid|AER|BadTLP|RxErr|soft lockup|fallen off the bus'; then
    echo "$line" >> "$LOG/kernel-events.log"
    if echo "$line" | grep -qE 'Xid.*\b(79|31|154|48|94|95)\b|fallen off the bus|soft lockup'; then
      halt stop "kernel: $line"
    fi
  fi
done &

while true; do
  w=""
  if [[ -n $POWER_CMD ]]; then
    w=$($POWER_CMD 2>/dev/null | awk '/Instantaneous/{print $4}')
  fi
  echo "$(date +%s) ${w:-NA}" >> "$LOG/wall.txt"
  if [[ -n $KILL_W && -n $w && $w -gt $KILL_W ]]; then
    halt kill "wall power ${w} W > ${KILL_W} W"
  fi
  sleep 1
done
