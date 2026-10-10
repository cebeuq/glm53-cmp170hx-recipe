#!/bin/bash
# P2P test battery (runs inside the p2ptools container; all 8 GPUs; engine must be DOWN).
# Host side (repo root, engine stopped):
#   mkdir -p p2p/test-out && docker compose -f p2p/tests/compose.yaml run --rm p2ptools [STAGES]
# STAGES (default: all): topo probe bw nccl ar  (bw/nccl/ar P2P parts need 'probe' in the same run)
# Works before install too (gives the no-P2P baseline: probe reports "not advertised").
# Every step is time-bounded; a hang is reported as a failure and the battery moves on.
set -uo pipefail
STAGES=${*:-topo probe bw nccl ar}
OUT=/test-out/$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
log() { echo "[p2p-test] $*" | tee -a "$OUT/summary.txt"; }
run() { local name=$1 t=$2; shift 2; log "RUN $name: $*"
        timeout -k 10 "$t" "$@" > "$OUT/$name.txt" 2>&1; local rc=$?
        log "END $name rc=$rc"; return $rc; }
has() { [[ " $STAGES " == *" $1 "* ]]; }
# Anything that pushes data across the UPI link with P2P runs only if the 8-GPU byte-exact probe passed
# in this same battery (a broken cross-socket path can hang a kernel and wedge a GPU).
island_ok() { grep -q '"passed":true' "$OUT/probe_check_0123.txt" 2>/dev/null; }
xsock_ok() { grep -q '"passed":true' "$OUT/probe_check_01234567.txt" 2>/dev/null; }

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
[[ ${used:-0} -lt 2048 ]] || { log "GPUs busy (max ${used} MiB used); stop the engine first"; exit 2; }
log "driver: $(head -1 /proc/driver/nvidia/version)"
grep -q ForceP2P /proc/driver/nvidia/params 2>/dev/null && log "params: $(grep -i registry /proc/driver/nvidia/params)"

if has topo; then
  run topo_m 30 nvidia-smi topo -m
  for c in r w n a p; do run "topo_p2p_$c" 30 nvidia-smi topo -p2p "$c"; done
fi

if has probe; then
  # Byte-exact content check; islands first (same PLX switch), then all 8 (crosses the UPI link).
  for set in 0,1 0,1,2,3 4,5,6,7 0,1,2,3,4,5,6,7; do
    tag=${set//,/}
    run "probe_identity_$tag" 60 env CUDA_VISIBLE_DEVICES=$set p2p_probe --identity
    run "probe_check_$tag" 300 env CUDA_VISIBLE_DEVICES=$set p2p_probe --check
  done
fi

if has bw; then
  # Enables peer access on every advertised pair: all 8 only if P2P is not advertised (baseline) or the
  # cross-socket probe passed; otherwise per switch island, and only if the island probe passed.
  ident=$OUT/probe_identity_01234567.txt
  if [[ ! -s $ident ]]; then log "SKIP bw (needs the probe stage in the same run)"
  else
    if xsock_ok || grep -q '"advertised":false' "$ident"; then run p2pBW_01234567 900 p2pBandwidthLatencyTest; fi
    if island_ok; then
      run p2pBW_0123 600 env CUDA_VISIBLE_DEVICES=0,1,2,3 p2pBandwidthLatencyTest
      run p2pBW_4567 600 env CUDA_VISIBLE_DEVICES=4,5,6,7 p2pBandwidthLatencyTest
    fi
  fi
fi

if has nccl; then
  # Graph-launched (-G) all-reduce at decode sizes, one thread per GPU. P2P off = today's path.
  for set in 0,1 0,1,2,3 0,1,2,3,4,5,6,7; do
    n=$(( $(tr -cd , <<<"$set" | wc -c) + 1 )); tag=${set//,/}
    for p2p in off on; do
      for proto in LL Simple; do
        if [[ $p2p == off ]]; then env_p2p=(NCCL_P2P_DISABLE=1); else env_p2p=(NCCL_P2P_DISABLE=0 NCCL_P2P_LEVEL=PXB); fi
        [[ $p2p == on ]] && ! island_ok && { log "SKIP nccl p2p-on (island probe not passed in this run)"; continue; }
        run "nccl_ar_${tag}_p2p${p2p}_${proto}" 300 env CUDA_VISIBLE_DEVICES=$set NCCL_PROTO=$proto NCCL_IB_DISABLE=1 \
          "${env_p2p[@]}" all_reduce_perf -t "$n" -g 1 -b 12K -e 1536K -f 2 -d bfloat16 -w 20 -n 200 -G 20 -c 1
      done
    done
  done
  # Cross-socket P2P
  xsock_ok && run nccl_ar_01234567_p2psys_LL 300 env NCCL_PROTO=LL NCCL_IB_DISABLE=1 NCCL_P2P_LEVEL=SYS \
    all_reduce_perf -t 8 -g 1 -b 12K -e 1536K -f 2 -d bfloat16 -w 20 -n 200 -G 20 -c 1
fi

if has ar; then
  # vLLM CustomAllreduce vs PyNccl (as wired in the engine), CUDA graphs, exact-sum checked.
  for p2p in 1 0; do
    for spec in "0,1:" "2,3:" "0,1,2,3:--force-pcie-custom-ar" "0,1,2,3,4,5,6,7:--force-pcie-custom-ar"; do
      set=${spec%%:*}; flag=${spec#*:}; tag=${set//,/}
      # 8 GPUs: custom AR would use cross-socket P2P; without a passed 8-GPU probe measure NCCL only.
      if [[ $tag == 01234567 ]] && ! xsock_ok; then flag="--no-custom"; fi
      island_ok || flag="--no-custom"
      run "vllm_ar_${tag}_nccldisable${p2p}" 600 env NCCL_P2P_DISABLE=$p2p NCCL_P2P_LEVEL=PXB NCCL_PROTO=LL NCCL_IB_DISABLE=1 \
        python3 /tests/ar_bench.py --gpus "$set" $flag
    done
  done
fi
log "results in $OUT"
