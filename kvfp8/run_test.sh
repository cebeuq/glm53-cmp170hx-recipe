#!/usr/bin/env bash
# int8/fp8 latent-KV unit tests + kernel microbench on one GPU, inside the serving image.
# Run from the repository root with the engine stopped, e.g.
#   GPU=0 MODEL_DIR=/path/to/model bash kvfp8/run_test.sh --model /model --bench --quick --sweep
#   GPU=0 CMP_KVFP8_BLOCK_H=32 bash kvfp8/run_test.sh --only-bench      # try a kernel tunable
# MODEL_DIR (mounted at /model) is optional: real kv_a_layernorm weights for the scale tests.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${IMAGE:-lazymio/vllm-backport@sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e}
GPU=${GPU:-0}
ENVS=()
for v in $(env | grep -o '^CMP_KVFP8_[A-Z_0-9]*' || true); do ENVS+=(-e "$v"); done
MOUNTS=(-v "$PWD/kvfp8:/kv:ro" -v "$PWD/patches/vllm:/p:ro")
[[ -n ${MODEL_DIR:-} ]] && MOUNTS+=(-v "$MODEL_DIR:/model:ro")
docker run --rm --gpus "device=$GPU" --network none --shm-size 8g --entrypoint python3 \
  "${ENVS[@]}" "${MOUNTS[@]}" "$IMAGE" -I /kv/test_kvfp8.py "$@"
