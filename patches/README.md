# Patches

The engine is the stock [`lazymio/vllm-backport`](https://hub.docker.com/r/lazymio/vllm-backport)
`latest-sm80` image pinned at
`sha256:2627d2a97e7f539b99ac98f511c589f5f520d3fc5630b195aca7f6ef865ead4e`
(vLLM `0.1.dev` `g29e66dad4`, torch 2.13 / CUDA 13.0). Nothing is rebuilt. Each file in
`vllm/` is a complete copy of one file from that image with our changes applied, and
`engine/compose*.yaml` bind-mounts it read-only over the original.

- `vllm/` - the files that are mounted (full copies, ready to use).
- `diffs/` - the same changes as unified diffs against the image originals, for review and
  for porting to a newer image. `cd <site-packages> && patch -p1 < diffs/<name>.diff`
  reproduces `vllm/<name>.py` byte for byte. New files (`kvfp8_sparse_mla.py`,
  `pipespec_core.py`) have no diff.

Every switch is an environment variable that defaults to off, so a mounted file with its
switch unset behaves like the original. Files derived from vLLM keep vLLM's Apache-2.0
SPDX header.

| File | Mounted over (`vllm/...`) | What it changes | Switch | Production |
|---|---|---|---|---|
| `marlin_utils_fp4.py` | `model_executor/layers/quantization/utils/marlin_utils_fp4.py` | NVFP4 -> Marlin repack in bounded chunks with the originals parked on the CPU, so the load fits in 64 GB cards | always | yes |
| `dsv32_kernels.py` | `models/deepseek_v32/common/kernels.py` | sm_80 has no `fp8e4nv` convert: software E4M3 encode and `uint8` views for the sparse-attention KV/indexer writers | always | base only (replaced by the next row) |
| `kvfp8_dsv32_kernels.py` | same path | `dsv32_kernels.py` + an int8 latent payload for the `fp8_ds_mla` KV layout | `CMP_KV_DSMLA_INT8=1` | yes (`compose.kvfp8i8.yaml`) |
| `kvfp8_sparse_mla.py` | `v1/attention/ops/kvfp8_sparse_mla.py` (new) | Triton read path for 8-bit latent KV on sm_80 (bytes expanded to BF16/FP16 in-kernel, per-tile scales, split-KV decode) | used when KV is 8-bit | yes (`compose.kvfp8.yaml`) |
| `xpu_mla_sparse.py` | `v1/attention/backends/mla/xpu_mla_sparse.py` | `TRITON_MLA_SPARSE` backend: accepts `fp8`/`fp8_ds_mla` KV and routes it to the read path above; metadata fix for the current fork | always | yes |
| `mqa_logits_triton.py` | `v1/attention/ops/mqa_logits_triton.py` | indexer logits: clamp the width captured into a FULL CUDA graph (an uncapped capture caused Xid 31) | always | yes |
| `v32_mtp.py` | `models/deepseek_v32/nvidia/mtp.py` | **MTP drafter embedding fix under pipeline parallelism** (see [docs/mtp.md](../docs/mtp.md)) | always | yes |
| `p2p_custom_all_reduce.py` | `distributed/device_communicators/custom_all_reduce.py` | lets vLLM's custom all-reduce run for TP > 2 over PCIe P2P | `CMP_PCIE_CUSTOM_AR=1` | yes (`compose.p2p.yaml`) |
| `pipespec_core.py` | `v1/cmp_pipespec.py` (new) | PipeSpec policy, flags and counters | `CMP_PIPESPEC=1` | yes |
| `pipespec_scheduler.py` | `v1/core/sched/scheduler.py` | `scheduler.py` below + PipeSpec chunk-A/B scheduling and dropped-chunk rollback | `CMP_PIPESPEC=1` | yes (`compose.pipespec.yaml`) |
| `pipespec_pp_utils.py` | `v1/worker/gpu/pp_utils.py` | PipeSpec stage-to-stage message formats | `CMP_PIPESPEC=1` | yes |
| `pipespec_model_runner.py` | `v1/worker/gpu/model_runner.py` | PipeSpec epoch logic on the first and last stage | `CMP_PIPESPEC=1` | yes |
| `pipespec_ar_speculator.py` | `v1/worker/gpu/spec_decode/autoregressive/speculator.py` | longer MTP draft chain (`continue_chain`) for PipeSpec | `CMP_PIPESPEC=1` | yes |
| `scheduler.py` | `v1/core/sched/scheduler.py` | prefill throttle hook; **experimental fair chunked prefill** | `CMP_PREFILL_INFLIGHT_CAP`, `CMP_FAIR_CHUNK` | base only (contained in `pipespec_scheduler.py`) |
| `engine_core.py` | `v1/engine/core.py` | optional cap on in-flight PP batches that carry prefill (wall-power limiting) | `CMP_PREFILL_INFLIGHT_CAP=N` | mounted, off |
| `config_vllm.py` | `config/vllm.py` | optional cap on concurrent PP batches | `CMP_MAX_CONCURRENT_BATCHES=N` | mounted, off |
| `indexer.py` | `v1/attention/backends/mla/indexer.py` | indexer prefill buffer `max_model_len x 40` made configurable; 4 frees ~2.3 GiB per GPU at 512K | `CMP_INDEXER_PREFILL_BUF_FACTOR` | yes (4) |
| `vocab_parallel_embedding.py` | `model_executor/layers/vocab_parallel_embedding.py` | Triton GEMV for `lm_head` at batch <= 16 (faster at TP1, no gain at TP2) | `CMP_LMHEAD_TRITON=1` | mounted, off |

## Porting to another image

The diffs are small (about 1,600 lines in total, most of them PipeSpec). To move to a newer
`vllm-backport` image: extract the originals from the new image
(`docker create` + `docker cp`), apply each diff with `patch -p1`, resolve any rejects, and
regenerate the full copies. Check the mount paths too: the fork moves files between releases.
