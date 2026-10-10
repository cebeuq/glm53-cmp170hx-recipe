# Kernels: what limits decode, and what we tried

There is no profiler on these cards: CUPTI is fused off, so `torch.profiler`, Nsight Systems and
Nsight Compute all fail. Everything below is CUDA-event timing of single kernels (under CUDA
graphs, L2 flushed where noted) or end-to-end A/B tests.

## Decode kernels, one GPU, one token (M=1), CUDA graphs

| Kernel | Time | Effective bandwidth |
|---|---:|---:|
| Routed MoE, Marlin NVFP4 (8 experts) | 154.5 us / layer | 1.10 TB/s |
| Attention INT8-g128 projections (Marlin W8A16) | 140 us / layer | `qkv_a` 0.72, `q_b` 0.92, `o_proj` 1.28 TB/s |
| `lm_head`, cuBLAS BF16 | 1,677 us | 1.13 TB/s |
| `lm_head`, Triton `tl.dot` GEMV | 1,125 us | 1.69 TB/s |
| Indexer top-k, `persistent_topk` at 4K context | 33.5 us | - |
| Indexer top-k, `top_k_per_row` | 13.2 us at 4K, 244 us at 131K | - |

Routed MoE at TP2 shapes and engine-like duty (other kernels in between, as in a real step):
96 / 154 / 226 / 299 us per layer for 1 / 2 / 3 / 4 tokens, i.e. 80-95% of the bandwidth
these cards reach in practice. The k=2 verify step is M=3.

Measured HBM bandwidth (`tools/gemv_bw.py`): about 1.37-1.41 TB/s on decode-shaped BF16
matrix-vector products at a 1,200 MHz SM clock; the copy/reduction peak is ~1.7 TB/s. The
HBM clock is fixed at 1,728 MHz.

## Tried, not adopted

| Idea | Result |
|---|---|
| Custom MoE GEMV reading Marlin's layout | correct, but 494 us vs Marlin 438 us (same conditions) |
| Port of the Flash recipe's decode MoE kernel | wins only at M=1 (-3 to -5 us per layer) and TP4 M<=2; production runs M=3 |
| INT4-g128 experts instead of NVFP4 | MoE kernel 10-12% faster, but the checkpoint failed the quality gate (93.65% agreement) |
| Thin-batch INT8 GEMV for the residue | slower than Marlin W8A16 |
| Per-channel INT8 residue (AllSpark kernel) | 10.42 vs 11.0 tok/s (slower than Marlin g128) |
| Fused sm_80 MoE router kernel | no end-to-end gain |
| Triton `lm_head` GEMV (`CMP_LMHEAD_TRITON=1`) | kernel 33% faster on the full vocabulary, but no end-to-end gain at TP2 (33.9 vs 34.4 tok/s), where each GPU holds half of it |

Marlin is close to what these cards can read, so the remaining decode time is mostly outside
the big matrix kernels: collectives, pipeline hand-offs, attention and indexer work, sampling,
the drafter, and gaps between kernels ([roofline](roofline.md)).

## Clocks and power

- At a 450 MHz SM clock lock a single GPU reached only 0.55-0.70 TB/s on the same products; at
  ~1,200 MHz 1.37-1.41 TB/s. Memory-bound decode still needs SM clock to issue enough loads.
- Memory overclocking is not possible: NVML's memory-clock offset range is 0..0 and 1,728 MHz
  is the only supported memory clock. The GPC offset range is +-1000, which does not help a
  memory-bound decode.
- Under back-to-back load the 160 W cap pulls the SM clock to 750-1,100 MHz, so multi-stream
  decode and prefill are power-limited at this cap; single-stream decode is not.
- PCIe Gen3 is fuse-blocked on these cards (an experimental cmpunlocker Gen3 branch knocked a
  card off the bus), and the estimated decode gain was under 1% anyway.
