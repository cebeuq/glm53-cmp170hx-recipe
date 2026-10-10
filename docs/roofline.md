# Roofline: how fast could this be?

Estimates, not measurements, unless marked "measured". Byte counts come from the model
config (hidden 6,144, 78 layers, q_lora 2,048, kv_lora 512, 64 heads, 256 experts with 8 active,
expert width 2,048, vocabulary 154,880); `tools/layer_bytes.py` computes the same census from a
checkpoint's safetensors headers.

## Bytes read per generated token (hybrid checkpoint)

| Part | Format | Per token |
|---|---|---:|
| MLA projections (`q_a`, `q_b`, `kv_a`, `kv_b`, `o_proj`), 78 layers | INT8-g128 | ~13.1 GB |
| Routed experts, 8 of 256 per layer, 75 layers | NVFP4 (4.5 bits) | ~12.7 GB |
| Shared experts + 3 dense MLP layers | INT8-g128 | ~3.6 GB |
| `lm_head` | BF16 | ~1.9 GB |
| Indexer, router, norms | BF16 | ~0.6 GB |
| **Total** | | **~32 GB** |

For comparison the original NVFP4 checkpoint (BF16 residue) reads ~49 GB per token; the INT8
residue is the single biggest bandwidth lever, which is why the hybrid exists.

## One token at a time (no speculation)

Single stream, TP2 x PP4: the four stages run one after another, and inside a stage two GPUs
read their halves in parallel. So per token each GPU on the critical path reads about
32 / 2 = 16 GB, plus ~1-2 GB of small projections that both GPUs of a pair read in full
(replicated low-rank and indexer weights).

| Per-GPU read bandwidth | Time per token | Upper bound |
|---|---:|---:|
| 1.70 TB/s (measured copy/reduction peak) | ~9.5-10.5 ms | **~95-105 tok/s** |
| 1.40 TB/s (best measured decode-shaped GEMV) | ~11.5-13 ms | ~77-87 tok/s |

Ignoring the replicated part, the bound is ~106 tok/s. Measured without speculation: 29.0 ms per
token (34.4 tok/s) before P2P, about 3x the floor.

## Where a step's time goes (production, k=2)

A k=2 verify step pushes 3 tokens through the model. Non-expert weights are read once; the
routed experts are read for up to 3 x 8 = 24 distinct experts per layer (about 23 if the three
tokens chose independently, fewer in practice), so ~19 GB + up to ~37 GB of experts, i.e. up to
~56 GB, or ~28-30 GB per GPU on the critical path. Add the drafter (two MTP passes on the last
stage, ~1.6 GB per GPU each):

| | Time |
|---|---:|
| Weight-read floor at 1.70 TB/s | ~17-19 ms (less if the 3 tokens share experts) |
| **Measured step, P2P + custom all-reduce** | **36.2 ms** |
| Measured step, before P2P | 42.2 ms |

That step yields ~2.4 tokens on average (more on code, fewer on prose), hence 60-75 tok/s.
PipeSpec adds a second chunk per step when the first is fully accepted.

What fills the gap between the floor and the measured step:

- ~166 tensor-parallel collectives per step: ~16 us each with custom all-reduce (2.6 ms);
  ~40-75 us each before P2P, which matches the 6 ms that P2P removed.
- 3 pipeline hand-offs + 1 broadcast back (~0.6 ms each, host-staged before P2P).
- `lm_head` and rejection sampling on the last stage (~1.5 ms).
- Kernels below peak: Marlin MoE runs at 80-95% of practical bandwidth, the INT8 attention
  projections at 0.7-1.3 TB/s ([kernels](kernels.md)).
- Sparse attention and the indexer (compute, not weights; grows with context).
- Kernel launch gaps that CUDA graphs do not remove (the drafter rebuilds attention metadata on
  the host between draft steps).

## Many streams

With 8 streams all four stages work at once and expert reads are shared between streams. The
measured 8-stream aggregate (135.5 tok/s) is close to the 4-stream one (128-132 tok/s); the
last stage (verify, sampling and drafting for every stream) and the 160 W cap
(SM clock drops to 750-1,100 MHz under sustained load) appear to be the shared ceiling.
