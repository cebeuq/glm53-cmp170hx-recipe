#!/usr/bin/env bash
# Start one queue cleanly: refuse if another queue/runner is alive, stop the server, start the queue.
# usage: bash loop/start_queue.sh loop/queues/NAME.txt   (from the repo root)
set -u
cd "$(dirname "$0")/.."
if pgrep -f '^bash loop/queue\.sh' >/dev/null || pgrep -f '^bash /.*/loop/run_candidate\.sh' >/dev/null; then
  echo "another queue/runner is alive; stop it first"; exit 1
fi
[ -s logs/guard/HALT ] && { echo "HALT present"; exit 1; }
pgrep -f '^bash loop/guard\.sh' >/dev/null || { echo "guard not running"; exit 1; }
for l in logs/devlock/gpu*.lock; do
  [ -e "$l" ] || continue
  exec {fd}>"$l"; flock -n "$fd" || { echo "dev job holds $l (loop/devlock.sh); wait for it"; exit 1; }; exec {fd}>&-
done
(cd engine && MODEL_DIR=${MODEL_DIR:-/dev/null} docker compose down >/dev/null 2>&1)
name=$(basename "$1" .txt)
setsid bash loop/queue.sh "$1" > "logs/$name.console" 2>&1 < /dev/null &
echo "started $1"
