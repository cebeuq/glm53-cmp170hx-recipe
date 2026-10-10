#!/usr/bin/env bash
# Run a GPU dev job (microbench, kernel test, training) on specific GPUs while the engine is down.
# usage (from the repo root): bash loop/devlock.sh GPUS [MAX_MIN] -- cmd args...
#   GPUS = comma list (e.g. 4,5). The command sees DEVLOCK_GPUS=GPUS (use it for --gpus device=...).
# Refuses if the engine container, a queue/candidate runner, or a guard HALT is live, or the guard is off.
# Takes a non-blocking flock per GPU (logs/devlock/gpuN.lock); start_queue.sh refuses while any is held.
# Kills the job after MAX_MIN minutes (default 30).
set -u
cd "$(dirname "$0")/.."
GPUS=$1; shift
MAX_MIN=30
[[ ${1:-} =~ ^[0-9]+$ ]] && { MAX_MIN=$1; shift; }
[[ ${1:-} == -- ]] && shift
[[ $# -gt 0 ]] || { echo "usage: devlock.sh GPUS [MAX_MIN] -- cmd..."; exit 2; }
docker ps --format '{{.Names}}' | grep -qx glm53-vllm && { echo "engine is running; dev jobs only while it is down"; exit 1; }
if pgrep -f '^bash loop/queue\.sh' >/dev/null || pgrep -f '^bash /.*/loop/run_candidate\.sh' >/dev/null; then
  echo "a queue/candidate runner is alive"; exit 1
fi
[ -s logs/guard/HALT ] && { echo "HALT present"; exit 1; }
pgrep -f '^bash loop/guard\.sh' >/dev/null || { echo "guard not running"; exit 1; }
mkdir -p logs/devlock
for g in ${GPUS//,/ }; do
  exec {fd}>"logs/devlock/gpu$g.lock"
  flock -n "$fd" || { echo "GPU $g is locked by another dev job"; exit 1; }
done
echo "$(date -Is) gpus=$GPUS max=${MAX_MIN}m: $*" >> logs/devlock/history.log
DEVLOCK_GPUS=$GPUS timeout --kill-after=30 "${MAX_MIN}m" "$@"
rc=$?
echo "$(date -Is) gpus=$GPUS exit=$rc" >> logs/devlock/history.log
exit $rc
