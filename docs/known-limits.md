# Status and known limits

- **One server, one set of cards.** Everything was measured on a single 8-GPU machine
  ([hardware](hardware.md)). Other boards, BIOS settings, PCIe topologies or driver builds may
  behave differently, especially for P2P.
- **The production configuration's chat speed is inferred.** Chat numbers were measured at
  358K context / 4 sequences; production (524K / 8 sequences / util 0.97) measured 4% lower on
  the raw-prose harness and was not re-measured on chat.
- **Greedy output is not reproducible run to run** on this engine, even with an identical
  configuration. This predates our patches; we have not tracked down the source.
- **PipeSpec** runs only when exactly one request is decoding. Without `CMP_PIPESPEC_REFILL=1`
  it leaves the MTP layer's KV and indexer keys stale at some positions, which may lower draft
  acceptance at long agentic contexts (not measured yet). One 4-stream measurement with PipeSpec
  enabled was 26% lower than without; a later one was 3% lower ([PipeSpec](pipespec.md)).
- **Fair chunked prefill** (`CMP_FAIR_CHUNK`) is experimental and off by default; a clean A/B is
  pending.
- **P2P** relies on a modified NVIDIA kernel module and on `ForceP2P`, which advertises every
  pair; only the byte-exact probe tells you whether your machine actually moves bytes correctly.
  Custom all-reduce across CPU sockets is unusably slow.
- **Patches are tied to one image.** The `patches/vllm/` files are whole-file copies for
  `lazymio/vllm-backport@sha256:2627d2a9...`. Mounting them over any other image will break it;
  port with the diffs instead.
- **Power.** The production stack peaks at ~2.3 kW at the wall. Plan the power supply and
  circuits accordingly, and run a guard that stops the engine on a wall-power limit.
- **Startup is slow**: loading ~420 GB of weights, repacking NVFP4 for Marlin and capturing CUDA
  graphs takes many minutes (the compose health check allows 45).
- **The API has no authentication.** It listens on 127.0.0.1 only by default.
- **The checkpoint is abliterated** (its refusal behavior was removed by the checkpoint author);
  plan any safeguards your use needs at the application level.

## The pipeline hand-off at TP2 (fixed)

This section used to be an open question found by reading the fork's source. It is confirmed
and fixed by `patches/vllm/ppfix_gpu_worker.py`, which `engine/compose.yaml` mounts.

In the fork's DeepSeek-V3.2 model (`models/deepseek_v32/nvidia/model.py`) the MoE layers run
with `reduce_results=False` (around line 96): each tensor-parallel rank's MoE output is a
partial sum, and the all-reduce is fused into the next layer's input norm. A non-last stage
returns these un-reduced `hidden_states` together with the residual (around line 288). The GPU
worker passed `all_gather_tensors = {}` to the pipeline send/receive
(`v1/worker/gpu_worker.py`, around line 1174), so `_should_use_all_gather`
(`distributed/parallel_state.py`, around lines 1054-1066) returned true: each TP rank sent only
its own slice of the tensor (around line 1208) and the receiving stage all-gathered the slices.
For the residual, which is identical on both ranks, that is correct. For the partial sums it
reassembled [rank 0's first half; rank 1's second half], and the next layer's all-reduce then
produced 2 x that instead of rank 0 + rank 1. This affected the MoE output of the last layer of
each of the first three stages (layers 21, 41 and 61 with the 22,20,20,16 split).

The fix is two lines: `"hidden_states": False` in both places where the worker builds
`all_gather_tensors`, so every rank sends its whole `hidden_states` to its peer on the next
stage and the residual keeps its existing rule. At TP1 the tensor is whole either way.

Measured 2026-11-04 on a second 8x CMP 170HX server (a contributed measurement, not the machine
in [hardware](hardware.md)): the production stack (TP2 x PP4, P2P, int8 KV, MTP k=2 + PipeSpec,
524K context) plus a grammar bitmask fix for drafts that was on in both runs (proposed separately),
same build and weights, with the original and with the patched `gpu_worker.py`.
The gate is `bench/quality_gate.py compare` against goldens captured from the same checkpoint
at TP8, where there is no pipeline and so no hand-off.

| | original | patched |
|---|---:|---:|
| Teacher-forced top-1 agreement | 86.98% | 90.13% |
| NLL on the fixed text, against the goldens | +0.38% | -0.65% |
| Log-prob difference per token, mean / p99 | 0.164 / 1.92 | 0.055 / 0.71 |
| Accepted MTP drafts per 2-draft chunk (temperature 0) | 1.21 | 1.30 |
| One stream, JSON / code / prose prompt, tok/s | 83.3 / 94.9 / 57.8 | 94.7 / 93.1 / 60.7 |

Those goldens are not the NVFP4 goldens behind the 93.9% and 96.4% in [results](results.md), so
read the difference (+3.2 points), not the level. No speed change can be resolved: code and prose
are equal within this engine's run-to-run spread, and the JSON prompt's 83.3 sits just under the
85.6 to 94.7 tok/s that reloads of the patched build gave with identical math. Draft acceptance is
lower with the original file (1.21 against 1.25 to 1.31 across patched reloads), which fits
corrupted hidden states also hurting the drafter. Not re-measured: the 93.9% row of the quality
table and GSM8K on the machine in [hardware](hardware.md); both predate the fix.
