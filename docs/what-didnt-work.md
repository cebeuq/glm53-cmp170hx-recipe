# What didn't work

Everything here was measured on the same server; "chat" is coding / browse / prose tok/s
([methodology](methodology.md)). Most rows are one A/B against the configuration of the day,
so small differences are within noise.

## Weights

| Idea | Result |
|---|---|
| Requantize the NVFP4 experts to INT4-g128 (RTN, from the NVFP4 values) | +5% decode over the hybrid at the time (11.64 vs 11.06 tok/s, same graph mode, 100 W / 450 MHz), but teacher-forced top-1 agreement 93.65% vs the original (gate: 95%). Rejected; the hybrid keeps the NVFP4 bytes. |
| Per-output-channel INT8 residue (to use the AllSpark W8A16 kernel) | slower: 10.42 vs 11.0 tok/s; Marlin g128 is faster on sm_80 |
| INT4 MTP experts | built to save ~6.6 GiB per GPU on the last stage; only tested before the MTP embedding fix, not since |

## Speculative decoding

| Idea | Result |
|---|---|
| MTP before the embedding fix | 22% acceptance at k=1; slower than no speculation ([MTP](mtp.md)) |
| Self-distilled MTP head (offline top-1 0.806 -> 0.850) | no gain in the engine; the offline holdout leaked |
| Chained-step trained MTP head (offline accepted length 1.851 -> 1.985 per k=3 draft) | no gain in the engine |
| k=3 | equal to k=2 for one stream, worse at 4 streams |
| Adaptive draft depth 1..4 per request | 64.7 / 69.7 / 57.2 vs fixed k=2 at 67.0 / 70.5 / 61.9 |

## Layouts and interconnect

| Idea | Result |
|---|---|
| TP8 | without P2P: 28.7 tok/s and a 93K-token KV pool; with custom all-reduce across sockets: 15.1 tok/s |
| TP4 x PP2 | with P2P as fast as TP2 x PP4 (61.1 tok/s, chat 67.0 / 70.5 / 61.9), but the KV pool tops out around 240K even with 8-bit KV |
| Host-staged all-reduce | faster than NCCL only while P2P is off ([comm](comm.md)) |
| NCCL `LL` protocol on PP8 | no gain (11.005 vs 11.0 tok/s); on TP2 x PP4 it is a clear win |
| PCIe Gen3 | fuse-blocked; estimated decode gain under 1% anyway |
| HBM overclock | locked (offset range 0..0) |

## Engine settings

| Idea | Result |
|---|---|
| `FULL_DECODE_ONLY` CUDA graphs | 10.04 vs 11.0 tok/s with `FULL_AND_PIECEWISE` |
| vLLM V1 model runner | single stream +0.7%, 4 streams 21.2 vs ~29-35 tok/s |
| `--no-async-scheduling` | +4.3 ms fixed cost per step |
| `--max-num-batched-tokens 4096` | prefill 497 vs 607 tok/s (13K prompt), smaller KV pool |
| Caps on in-flight prefill batches | only useful to limit wall power; turning the cap off gave +18% at 4 streams and +13% prefill |
| `--max-num-seqs 8` on PP8 | 71.0 tok/s at 8 streams vs 82-97 at 4: the pipeline schedules a request's decode steps `pp_size` steps apart, so extra streams queue. (On the current TP2 x PP4 stack 8 sequences reach 135.5.) |
| Plain `fp8` KV with scale 1.0 | unsafe: 99% of the early layers' latent values are subnormal ([KV cache](kv-cache.md)) |

## Kernels

Custom MoE GEMV, a port of the Flash recipe's decode MoE kernel, a fused router, a thin-batch
INT8 GEMV and a Triton `lm_head`: none beat Marlin/cuBLAS end to end at the shapes production
runs (3-token verify at TP2). Details in [kernels](kernels.md).

## Lessons

- Measure acceptance in the engine, not offline. Two trained drafter heads looked 5-7% better
  offline and were no better in the engine.
- Check the drafter under the exact parallel layout you serve. The same head accepted 85% at
  TP8 and 20% under PP.
- Greedy output from this engine is not reproducible run to run; do not use exact match as a
  correctness gate. Use teacher-forced agreement against goldens, NLL on fixed text, GSM8K,
  long-context ladders and multi-stream corruption checks (`bench/`).
