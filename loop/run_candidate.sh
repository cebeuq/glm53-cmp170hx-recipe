#!/usr/bin/env bash
# One candidate config on the GPU server: (re)launch -> health -> harness -> optional gates -> ledger row.
# usage (as a user that can run docker): run_candidate.sh CAND_ID VARS_FILE [--no-quality|--capture]
#        [--ladder 65536,262144,520000] [--conc-gate] [--accept] [--gsm8k N] [--exact] [--conc N] [--min-tps X]
# VARS_FILE is a compose --env-file for engine/ (see engine/prod.vars.example).
# Results: logs/cand-CAND_ID/*.json and one JSON row appended to RESULTS.ndjson in the repo root.
# Never touches power/clocks. Stops on guard HALT. Leaves the candidate server running for inspection.
set -uo pipefail
CAND=$1; VARS=$(realpath "$2"); shift 2
QUALITY=1; CAPTURE=0; LADDER=""; CONC=0; GSM8K=0; ACCEPT=0; EXACT=0; HARNESS_ARGS=${HARNESS_ARGS:-}
while [[ $# -gt 0 ]]; do case $1 in
  --no-quality) QUALITY=0;; --capture) CAPTURE=1; QUALITY=0;; --ladder) LADDER=${2//,/ }; shift;; --conc-gate) CONC=1;; --min-tps) MIN_TPS=$2; shift;; --gsm8k) GSM8K=$2; shift;; --accept) ACCEPT=1;; --exact) EXACT=1;; --conc) HARNESS_ARGS="$HARNESS_ARGS --conc $2"; shift;; esac; shift; done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/logs/cand-$CAND; mkdir -p "$OUT"
ENG=$ROOT/engine
GOLD=${GOLD:-$ROOT/goldens/goldens-baseline.json}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-2700}
log() { echo "$(date -Is) [$CAND] $*" | tee -a "$OUT/run.log"; }
row() { python3 - "$@" <<'PY'
import json, sys, os
out, cand, vars_file, verdict = sys.argv[1:5]
r = {"cand": cand, "verdict": verdict, "vars": open(vars_file).read()}
for name in ("startup.json", "harness.json", "quality.json", "ladder.json", "conc.json", "power.json", "gsm8k.json", "accept.json"):
    p = os.path.join(out, name)
    if os.path.exists(p):
        try:
            r[name.split(".")[0]] = json.load(open(p))
        except Exception as e:
            r[name.split(".")[0]] = {"parse_error": str(e)}
with open(os.path.join(os.path.dirname(out), "..", "RESULTS.ndjson"), "a") as f:
    f.write(json.dumps(r) + "\n")
print(json.dumps({k: r.get(k) for k in ("cand", "verdict")}))
PY
}
[[ -s $ROOT/logs/guard/HALT ]] && { log "guard HALT present; refusing to run"; exit 3; }
pgrep -f loop/guard.sh >/dev/null || { log "guard not running; refusing to run"; exit 3; }

cp "$VARS" "$OUT/vars"
cd "$ENG"
docker compose --env-file "$VARS" config --quiet || { log "compose config invalid"; row "$OUT" "$CAND" "$VARS" CONFIG_INVALID; exit 2; }
xid0=$(dmesg | grep -c Xid || true)
log "down previous"; MODEL_DIR=${MODEL_DIR:-/dev/null} docker compose down >/dev/null 2>&1
t0=$(date +%s)
log "up"; docker compose --env-file "$VARS" up -d >/dev/null 2>&1
status=TIMEOUT
while (( $(date +%s) - t0 < HEALTH_TIMEOUT )); do
  if curl -sf -m 3 http://127.0.0.1:8000/health >/dev/null; then status=HEALTHY; break; fi
  if [[ -s $ROOT/logs/guard/HALT ]]; then status=HALT; break; fi
  if ! docker ps -q -f name=glm53-vllm -f status=running | grep -q .; then status=EXITED; break; fi
  if docker logs glm53-vllm 2>&1 | grep -q -E "hit an exception|EngineCore failed"; then status=CRASHED; break; fi
  sleep 10
done
t1=$(date +%s)
docker logs glm53-vllm > "$OUT/vllm-boot.log" 2>&1
kv=$(grep -oE "GPU KV cache size: [0-9,]+" "$OUT/vllm-boot.log" | tail -1 | grep -oE "[0-9,]+$" | tr -d ,)
avail=$(python3 - "$OUT/vllm-boot.log" <<'PY'
import re, sys
vals = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"\(([0-9.]+), ([0-9.]+) GiB\)\. Actual usage is ([0-9.]+) GiB for consumed memory .*?, ([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph", line)
    if m:
        vals.append(float(m.group(2)) - float(m.group(3)) - float(m.group(4)) - float(m.group(5)))
print(round(min(vals), 2) if vals else "")
PY
)
echo "{\"status\":\"$status\",\"startup_s\":$((t1 - t0)),\"kv_tokens\":${kv:-null},\"min_avail_kv_gib\":${avail:-null}}" > "$OUT/startup.json"
log "startup: $(cat "$OUT/startup.json")"
if [[ $status != HEALTHY ]]; then
  grep -E "Error|error|Traceback" "$OUT/vllm-boot.log" | tail -20 >> "$OUT/run.log"
  row "$OUT" "$CAND" "$VARS" "BRINGUP_$status"; exit 1
fi

ts0=$(date +%s)
log "harness"; timeout 1800 python3 -I "$ROOT/bench/harness.py" --tag "$CAND" $HARNESS_ARGS > "$OUT/harness.json" 2> "$OUT/harness.err"
ts1=$(date +%s)
awk -v a=$ts0 -v b=$ts1 '$1>=a && $1<=b && $2!="NA"{s+=$2;n++; if($2>m)m=$2} END{printf "{\"wall_mean_W\":%.0f,\"wall_peak_W\":%d}\n", (n?s/n:0), m}' \
  "$ROOT/logs/guard/wall.txt" > "$OUT/power.json"
log "harness: $(cat "$OUT/harness.json")"
spec=$(curl -s -m 10 http://127.0.0.1:8000/metrics | grep -E "^vllm:spec_decode_num_(draft_tokens|accepted_tokens)_total" | awk '{print $NF}' | tr '\n' ' ')
[ -n "$spec" ] && log "spec draft/accepted: $spec"

verdict=MEASURED
if (( CAPTURE )); then
  log "capturing goldens -> $GOLD"
  mkdir -p "$(dirname "$GOLD")"; chmod 700 "$(dirname "$GOLD")"
  timeout 3600 python3 -I "$ROOT/bench/quality_gate.py" capture http://127.0.0.1:8000 "$GOLD" > "$OUT/capture.out" 2>&1
  cap=$?; log "capture exit=$cap"; [[ $cap -eq 0 ]] && verdict=GOLDENS_CAPTURED || verdict=CAPTURE_FAIL
fi
if (( EXACT )); then
  log "exact-match greedy run"
  timeout 3600 python3 -I "$ROOT/pipespec/exact_match.py" run http://127.0.0.1:8000 "$ROOT/bench/accept_prompts.jsonl" "$OUT/exact.json" --max-tokens 256 --n 36 > "$OUT/exact.out" 2>&1
  log "exact: rc=$? $(tail -1 "$OUT/exact.out" | cut -c1-200)"
fi
if (( ACCEPT )); then
  log "accept eval"
  timeout 3600 python3 -I "$ROOT/bench/accept_eval.py" http://127.0.0.1:8000 "$ROOT/bench/accept_prompts.jsonl" "$OUT/accept-full.json" > "$OUT/accept.json" 2> "$OUT/accept.err"
  log "accept: $(cat "$OUT/accept.json")"
fi
if (( GSM8K > 0 )); then
  log "gsm8k n=$GSM8K"
  timeout 5400 python3 -I "$ROOT/bench/gsm8k_eval.py" http://127.0.0.1:8000 "${GSM8K_DATA:-$ROOT/bench/data/gsm8k_test_300.json}" "$OUT/gsm8k-full.json" --n "$GSM8K" > "$OUT/gsm8k.json" 2> "$OUT/gsm8k.err"
  log "gsm8k: $(cat "$OUT/gsm8k.json")"
fi
MIN_TPS=${MIN_TPS:-0}  # also settable with --min-tps
tps=$(python3 -c "import json;print(json.load(open('$OUT/harness.json')).get('decode_c1_tps',0))" 2>/dev/null || echo 0)
if python3 -c "import sys;sys.exit(0 if float('$tps') < float('$MIN_TPS') else 1)"; then
  log "decode_c1_tps $tps < MIN_TPS $MIN_TPS: skipping quality gate"; QUALITY=0; verdict=SLOWER
fi
if (( QUALITY )) && [[ -f $GOLD ]]; then
  log "quality gate"
  timeout 3600 python3 -I "$ROOT/bench/quality_gate.py" compare http://127.0.0.1:8000 "$GOLD" "$OUT/quality.json" > "$OUT/quality.out" 2>&1
  q=$?; log "quality exit=$q: $(tail -3 "$OUT/quality.out" | tr '\n' ' ')"
  [[ $q -eq 0 ]] && verdict=QUALITY_PASS || verdict=QUALITY_FAIL
fi
if (( CONC )); then
  timeout 1800 python3 -I "$ROOT/bench/conc_gate.py" http://127.0.0.1:8000 > "$OUT/conc.out" 2>&1
  c=$?; echo "{\"exit\":$c}" > "$OUT/conc.json"; log "conc_gate exit=$c"
  [[ $c -ne 0 ]] && verdict=CONC_FAIL
fi
if [[ -n $LADDER ]]; then
  log "ctx ladder: $LADDER"
  timeout 5400 python3 -I "$ROOT/bench/ctx_crash_ladder.py" http://127.0.0.1:8000 $LADDER > "$OUT/ladder.out" 2>&1
  l=$?; echo "{\"exit\":$l,\"rungs\":\"$LADDER\"}" > "$OUT/ladder.json"; log "ladder exit=$l: $(tail -4 "$OUT/ladder.out" | tr '\n' ' ')"
  [[ $l -ne 0 ]] && verdict=LADDER_FAIL
fi
xid1=$(dmesg | grep -c Xid || true)
if (( xid1 > xid0 )); then verdict=XID; log "NEW XID: $(dmesg | grep Xid | tail -3)"; fi
row "$OUT" "$CAND" "$VARS" "$verdict"
log "done verdict=$verdict"
