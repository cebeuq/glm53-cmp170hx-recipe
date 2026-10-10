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

## Open question: the pipeline hand-off at TP2 (unverified)

Found by reading the fork's source; **not yet confirmed or A/B tested.**

In the fork's DeepSeek-V3.2 model (`models/deepseek_v32/nvidia/model.py`) the MoE layers run
with `reduce_results=False` (around line 96): each tensor-parallel rank's MoE output is a
partial sum, and the all-reduce is fused into the next layer's input norm. A non-last stage
returns these un-reduced `hidden_states` together with the residual (around line 288). The GPU
worker passes `all_gather_tensors = {}` to the pipeline send/receive
(`v1/worker/gpu_worker.py`, around line 1174), so `_should_use_all_gather`
(`distributed/parallel_state.py`, around lines 1054-1066) returns true: each TP rank sends only
its own slice of the tensor (around line 1208) and the receiving stage all-gathers the slices.
For the residual, which is identical on both ranks, that is correct. For the partial sums it
would reassemble [rank 0's first half; rank 1's second half], and the next layer's all-reduce
would then produce 2 x that instead of rank 0 + rank 1. This would affect the MoE output of the
last layer of each of the first three stages (layers 21, 41 and 61 with the 22,20,20,16 split).

What fits this reading: on the same weights, TP2 x PP4 showed lower teacher-forced top-1
agreement (93.9%) than PP8 (96.4%), which we had attributed to all-reduce rounding. What does
not obviously fit: GSM8K-250 on TP2 x PP4 (84.8%) was not lower than PP8 (83.2%), and outputs
are coherent. A fix would be about ten lines (send full tensors at the PP boundary, or reduce
before sending); the test is NLL on fixed text and teacher-forced agreement against PP8,
plus GSM8K. If you run this stack and can check it, please open an issue.
