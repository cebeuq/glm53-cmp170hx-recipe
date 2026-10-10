# All-reduce on PCIe Gen2: measurements

Decode on TP2 x PP4 runs about 158 tensor-parallel collectives per step (2 per layer for 78
layers, plus the embedding and `lm_head`), ~166 with MTP k=2, one after another. Each carries
6,144 BF16 values per token: 12 KB for one token, 36 KB for a 3-token verify step, ~150 KB at
4 streams. At these sizes latency, not bandwidth, decides.

## Before P2P (inferred from end-to-end runs)

No direct microbenchmark exists for this state; these are inferred from step times:

- **TP2:** PP8 decodes at 32.3 ms/token, TP2 x PP4 + NCCL LL at 29.0 ms. Halving each GPU's
  layer work should have landed near 17-19 ms + hops, so roughly 10 ms/step is collective
  cost, i.e. ~40-75 us per call. Switching NCCL from `Simple` to `LL` alone saved ~5 ms/step
  (29.4 -> 34.4 tok/s).
- **TP4:** TP4 x PP2 + MTP decoded at ~74 ms/step vs ~43 ms for TP2 x PP4 despite half the
  per-GPU work: >= ~200 us per 4-GPU call.
- **TP8:** ~150 us per 8-GPU call.

## With P2P

nccl-tests `all_reduce_perf`, 12 KB, LL protocol, graph-launched:

| GPUs | P2P off | P2P on |
|---|---:|---:|
| 2 (same switch) | 22.8 us | 11.4 us |
| 4 (same switch) | 58 us | 22 us |
| 8 (both sockets) | 80 us | 74 us (`PXB`) / 65 us (`SYS`) |

TP2 under CUDA graphs, P2P on, microseconds per call (`comm/` microbench, results bit-equal to
NCCL):

| Tokens (bytes) | NCCL | host-staged LL | P2P push | **vLLM custom all-reduce** |
|---|---:|---:|---:|---:|
| 1 (12 KB) | 17.3 | 10.9 | 9.3 | 11.2 |
| 3 (36 KB, k=2 verify) | 27.7 | 21.1 | 21.6 | **16.0** |

Production verifies 3 tokens per step, where vLLM's custom all-reduce is fastest, so it stays.

End to end (TP2 x PP4, MTP k=2, chat): NCCL over P2P +4-7%; plus custom all-reduce +12-16%.
Step time 42.1 -> 39.6 ms (NCCL P2P) -> 36.2 ms (custom all-reduce).

## Across sockets

`ForceP2P` advertises cross-socket pairs, and they pass the byte-exact probe at the same
5.79 GB/s. But TP8 with vLLM's custom all-reduce across the two sockets decoded at 15.1 tok/s
(152 ms per step), against 28.7 tok/s for TP8 on plain NCCL without P2P and ~60 tok/s for TP4
inside one switch. Keep custom all-reduce groups inside one PCIe switch. (PP hand-offs across
sockets over NCCL P2P are fine: `NCCL_P2P_LEVEL=SYS` is the production setting.)

## Host-staged all-reduce (fallback without P2P)

We ported the Flash recipe's host-shared-memory all-reduce (one round trip through pinned host
memory instead of NCCL's multi-step SHM ring) plus a flags-in-data LL variant and a P2P push
variant, as a vLLM all-reduce backend. Without P2P it is the fastest path (estimated +7-9% end to
end over NCCL LL); with P2P, vLLM's custom all-reduce wins at the 3-token size, so it is not
part of this recipe's production stack. Fusing the following RMSNorm into it saved nothing
(21.5 vs 20.6 us). If you cannot install the P2P driver, Morrowmake's
[vLLM fork](https://github.com/Morrowmake/vllm-cmp170hx) has the original implementation.
