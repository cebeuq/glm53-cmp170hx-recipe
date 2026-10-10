#!/usr/bin/env bash
# Run candidates sequentially on the GPU server. Queue file lines: CAND VARS_FILE [run_candidate flags...]
# Blank lines and '#' comments are skipped. Lines starting with '!' are GPU-idle shell steps. Stops on guard HALT or any Xid. Leaves the last server up.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
Q=$(realpath "$1")
while read -r line; do
  [[ -z ${line// } || $line == \#* ]] && continue
  if [[ $line == '!'* ]]; then   # GPU-idle shell step: stop the server, run the command from the repo root
    (cd "$ROOT/engine" && MODEL_DIR=${MODEL_DIR:-/dev/null} docker compose down >/dev/null 2>&1)
    echo "$(date -Is) queue: step ${line:1}"
    (cd "$ROOT" && bash -c "${line:1}") >> "$ROOT/logs/queue-steps.log" 2>&1
    echo "$(date -Is) queue: step exit=$?"
    continue
  fi
  read -r cand vars flags <<< "$line"
  [[ -s $ROOT/logs/guard/HALT ]] && { echo "$(date -Is) queue: HALT present, stopping"; exit 3; }
  echo "$(date -Is) queue: start $cand"
  # shellcheck disable=SC2086
  bash "$ROOT/loop/run_candidate.sh" "$cand" "$ROOT/$vars" $flags
  v=$(tail -1 "$ROOT/RESULTS.ndjson" | python3 -c "import json,sys;print(json.load(sys.stdin).get('verdict'))")
  echo "$(date -Is) queue: $cand -> $v"
  [[ $v == XID || $v == BRINGUP_HALT ]] && { echo "queue: stopping on $v"; exit 4; }
done < "$Q"
echo "$(date -Is) queue: done"
