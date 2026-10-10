# 8-bit latent KV cache on Ampere (int8-tile)

Overlays: `engine/compose.kvfp8.yaml` (read path) + `engine/compose.kvfp8i8.yaml` (int8
payload, sets `CMP_KV_DSMLA_INT8=1`), with `KV_DTYPE=fp8_ds_mla`.

## Why

GLM-5.3's attention stores one 576-value latent per token per layer (512 compressed KV + 64
RoPE), plus a 132-byte indexer key on every fourth layer. In BF16 that is 1,152 bytes per token
per layer. On TP2 x PP4 with MTP, the BF16 pool was 171K tokens: not enough for 350K, let alone
512K. Halving the latent is the only lever that does not cost speed elsewhere.

Ampere has no FP8 arithmetic, and the fork's sparse-MLA read path is BF16 only. The Flash recipe
deliberately keeps a full-precision KV cache, so there was nothing to port; this was built here.

## Formats

All are selectable with `KV_DTYPE`; the read kernel handles each.

| `KV_DTYPE` | Bytes / token / layer | Latent | RoPE | Scale |
|---|---:|---|---|---|
| `bfloat16` | 1,152 | BF16 | BF16 | - |
| `fp8_ds_mla` | 656 | e4m3 | BF16 | fp32 per token per 128 values (amax/448, at write) |
| `fp8_ds_mla` + `CMP_KV_DSMLA_INT8=1` (**production**) | 656 | **int8** | BF16 | fp32 per token per 128 values (amax/127) |
| `fp8` | 576 | e4m3 | e4m3 | one per-tensor scale (1.0: the checkpoint has none) |

`fp8_ds_mla` is DeepSeek's V3.2 sparse cache layout (FlashMLA on Hopper maps `fp8` to it).
The int8 option keeps that layout and vLLM's page accounting and changes only the latent bytes,
so the writer (`kvfp8_dsv32_kernels.py`) and reader (`kvfp8_sparse_mla.py`) must agree; the
reader checks the writer module at the first forward and raises if they do not.

Plain `fp8` with scale 1.0 is **unsafe** for this model: layers 0-9 have latent RMS of
0.003-0.05, so 99% of their e4m3 values are subnormal (4.5% latent error at layer 3).

## The read kernel

`patches/vllm/kvfp8_sparse_mla.py` (new file), called from the `TRITON_MLA_SPARSE` backend:

- final (prefill) and split-KV (decode) Triton kernels;
- bytes -> exact BF16 with inline PTX (`prmt`/`lop3` plus one exact x2^120 `fma.rn.bf16x2`,
  4 values per instruction); int8 via `xor`/`prmt` and one `sub.f16x2` per 2 values, dots in
  FP16;
- per-tile scales are applied to the [heads x keys] score and probability tiles, not per
  element; the latent is loaded once per tile for both QK and PV;
- rows addressed by (block, offset) with the real block stride, so any page layout works;
  CUDA-graph safe;
- split policy: keep splitting the 2,048 selected keys until about 2 waves of programs fill the
  SMs (`CMP_KVFP8_SPLIT_WAVES=2`). The BF16 kernel stops splitting early, which is why
  12-query batches are 2.5x faster on the 8-bit path.

## Accuracy

Kernel correctness (GPU, `kvfp8/test_kvfp8.py`): every decoder bit-exact on all 256 codes;
kernels vs an fp32 reference on the same dequantized cache 1.6-2.3e-3 relative (the BF16 kernel:
1.7-4.5e-3); padding, fully masked rows, strided pages and CUDA-graph replay all checked.

Real activations: relative L2 error of the attention block output (after V up-projection and
`o_proj`) against a BF16 cache, 256 queries over 2,048 causal keys:

| Layer | Latent RMS | BF16-cache floor | `fp8` (576 B) | `fp8_ds_mla` (e4m3) | **int8-tile** |
|---|---:|---:|---:|---:|---:|
| 78, MTP (exact inputs) | 0.38 | 0.35% | 1.86% | 1.23% | **0.48%** |
| 3 (proxy input) | 0.004 | 0.21% | 1.35% | 0.24% | 0.15% |
| 60 (proxy input) | 0.53 | 0.46% | 0.67% | 0.60% | 0.37% |
| 77 (proxy input) | 0.59 | 0.30% | 0.40% | 0.35% | 0.23% |

Main layers used a proxy input (near-uniform attention), so their errors look smaller than in
the engine; the MTP layer had exact inputs.

In the engine: chat speed with int8-tile KV at 350K matched BF16 KV at 128K (64.7 / 67.0 / 58.4
vs 65.3 / 67.8 / 58.4 tok/s), and the same stack with PipeSpec scored 84.8% on GSM8K-250,
inside the 83-88% band of BF16-KV runs.

## Speed (one GPU, TP2 shapes: 32 heads, CUDA-graph replay, L2 flushed, 2,048 keys)

Microseconds per layer call; context 4K-512K gives the same numbers (random gather).

| Queries | BF16 kernel | int8-tile | `fp8_ds_mla` |
|---|---:|---:|---:|
| 1 (MTP draft step) | 25-29 | 27 | 27 |
| 3 (k=2 verify, one stream) | 38 | 36 | 38 |
| 4 | 40 | 38 | 40 |
| 12 (verify, 4 streams) | 275 | 104 | 115 |
| prefill chunk, 2,048 tokens | 10.4 ms | 8.9 ms | 9.9 ms |

## Capacity

The MLA latent is replicated on every TP rank, so a TP4 stage holds twice the per-GPU KV of a
TP2 stage. `kvfp8/kv_capacity.py` models the pool per stage (calibrated on startup logs to
within 0.1%):

| Layout, layer split | BF16 | 8-bit (656 B) |
|---|---:|---:|
| TP2 x PP4, 21,20,20,17 | 171K (measured) | 293K (stage 4 binds) |
| **TP2 x PP4, 22,20,20,16** | 303K | **520K** (measured 520,384) |
| TP2 x PP4, 22,20,20,16, util 0.97 | 331K | **570K** (measured 569,792) |
| TP4 x PP2, 42,36 | 140K (measured) | 240K |

Moving one layer from the last stage (which also carries the 18.5 GiB MTP layer) to the first
balances the stages for 8-bit KV; the slowest TP2 stage still has 20 MoE layers, so speed is
unchanged.

## Testing

`GPU=0 MODEL_DIR=... bash kvfp8/run_test.sh --model /model --bench --quick --sweep` runs the unit
tests, the microbench and a tunable sweep in the serving image on one GPU (engine stopped).
Tunables: `CMP_KVFP8_DECODE` (1 = PTX, 0 = lookup table, 2 = e4m3 -> fp16 without the FMA),
`CMP_KVFP8_BLOCK_H`, `CMP_KVFP8_SPLIT_BN`, `CMP_KVFP8_SPLIT_WAVES` and others in the file header.
