#!/usr/bin/env bash
# Stop queue runners, candidate runners and bench clients (anchored patterns: never matches ssh/bash -c shells).
for pat in '^bash loop/queue\.sh' '^bash /.*/loop/run_candidate\.sh' '^python3 -I /.*/bench/(quality_gate|conc_gate|gsm8k_eval|harness|ctx_crash_ladder)\.py'; do
  p=$(pgrep -f "$pat"); [ -n "$p" ] && kill $p
done
sleep 2; cd "$(dirname "$0")/../engine" && MODEL_DIR=${MODEL_DIR:-/dev/null} docker compose down >/dev/null 2>&1; echo stopped
