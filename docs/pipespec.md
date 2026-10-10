# PipeSpec: pipelined speculative decoding across pipeline stages

Switch: `CMP_PIPESPEC=1` with `engine/compose.pipespec.yaml`. Optional: `CMP_PIPESPEC_REFILL=1`
(see risks), `CMP_PIPESPEC_LOG=N` (log counters every N epochs).

## The idle-stage problem

Production is TP2 x PP4 with MTP k=2, so one request sends one 3-token chunk (the last token
plus 2 drafts) through the pipeline per step. A step walks stage 1 -> 2 -> 3 -> 4; stage 4
verifies, samples, runs the MTP drafter and broadcasts the tokens and new drafts back. Only
then can stage 1 start the next chunk. With a single request, three of the four stages are
idle at any moment. (vLLM's V1 pipeline scheduler deliberately keeps one request in only one
in-flight micro-batch; the fork's "ring cadence" spaces a request's decode steps `pp_size`
steps apart.)

## The idea

Let stage 1 start the next chunk before stage 4 has verified the current one, betting that the
current chunk is fully accepted:

```
step e   (chunk A): stage 1 runs [last, d1, d2]     drafts from the previous epoch (P1)
step e+1 (chunk B): stage 1 runs [d3, d4, d5]       further drafts (P2), right behind A
step e+4          : next epoch (normal ring cadence)

stage 4, after verifying A (one host sync on a 1-int flag):
  B valid  <=>  A accepted d1 and d2  AND  A's bonus token == d3
  valid  : verify B with drafts d4, d5; propose the next P1 (2 drafts) and broadcast it,
           then chain 3 more drafts (P2) and broadcast those
  invalid: skip B's forward; propose from A's end instead; report B as "0 tokens"
```

Two details make it pay:

- **Two-phase broadcast.** Stage 4 sends the next chunk A's 2 drafts as soon as they exist,
  then keeps chaining 3 more for chunk B. Stage 1 needs those only ~8 ms later, after it has
  run chunk A, so the extra drafting is off the critical path.
- **A failed bet costs almost nothing.** Stages 1-3 would have been idle anyway; stage 4 skips
  chunk B's forward and only receives its hidden states.

## Correctness

- **No copies, no erasing.** As in stock speculative decoding, KV at positions at or after the
  committed length is "don't care": every chunk writes its own positions before attending and
  starts at the first uncommitted position, so a dropped chunk B's KV is overwritten before
  anyone reads it.
- **A valid chunk B computes exactly what a normal step would**: same positions, same KV
  context (A was fully accepted), same 1x3 CUDA graph, drafts through the unchanged rejection
  sampler. At temperature 0 the output is token-identical in principle; at T > 0 it is
  identical in distribution (drafts are greedy, validity is token equality with the sampled
  bonus).
- **Message shapes never depend on the decision.** Every stage-to-stage message has a format
  fixed by the step's marker (normal / A / B); only the contents change, so the ranks cannot
  fall out of NCCL lockstep.
- **Scheduler accounting.** A dropped chunk B reports zero tokens and the scheduler subtracts its
  3 optimistic tokens, keeping the "1 + k new tokens per decode" invariant. The KV lookahead is
  raised to 2k + 1 = 5 slots.
- `pipespec/check.py` simulates the protocol at token level on a toy model (6 seeds, pauses,
  gating flips) and checks: committed tokens equal greedy decoding, first and last stage agree
  on every chunk's positions and inputs, no stage reads uncommitted KV, and the scheduler
  invariant. Five deliberately injected protocol bugs are each caught.
  `docker compose -f pipespec/compose.yaml run --rm check` (CPU only).

Gate: PipeSpec runs only when exactly one request is running and nothing is waiting, with
exactly k=2 greedy drafts, no structured output / logprobs / LoRA, and at least 16 tokens left
before `max_tokens` and `max_model_len`. Otherwise every patched file behaves like the original.

## Throughput model

`pipespec/model.py` is a discrete-event simulation of the four stages, calibrated on the
pre-P2P step time (derived, not measured per stage):

| Item | ms |
|---|---:|
| one stage forward, 1-token chunk | 6.3 |
| one stage forward, 3-token chunk | 8.2 |
| stage-to-stage hop, and the 4 -> 1 broadcast | 0.6 each |
| `lm_head` + sampling on stage 4 | 1.5 |
| MTP propose, 2 drafts | 5.6 |
| check: 4 x 8.2 + 3 x 0.6 + 0.6 + 1.5 + 5.6 = | 42.3 (measured 42.2) |

The model predicted +12% to +23% at one stream depending on how well the chained drafts are
accepted, and a loss with two streams (stage 4 becomes the bottleneck), which is why it is
gated to one running request.

## Measured

Same session, flag off vs on, TP2 x PP4 + P2P custom all-reduce, k=2, 128K:

| | Coding | Browse | Prose | Harness 1-stream | Harness 4-stream |
|---|---:|---:|---:|---:|---:|
| PipeSpec off | 65.8 | 67.8 | 60.3 | 59.7 | 132.8 |
| PipeSpec on | 69.6 | 76.3 | 63.2 | 58.0 | 99 |
| change | +6% | +13% | +5% | | |

With int8 KV at 350K (production stack), on vs off: coding 68.6 vs 64.7, browse 72.8 vs 67.0,
prose 59.9 vs 58.4 (+6% / +9% / +3%), 4-stream harness 128.5 vs 132.0 (the off run also used
the drafter's local argmax, which PipeSpec does not).

In the engine log, chunk B was valid in about 20% of epochs on the raw-prose harness
(2.55 tokens per epoch); acceptance is higher on code and agent text, which is where PipeSpec
gains most.

Greedy exact match: with PipeSpec on, 36 of 36 prompts diverged from the PipeSpec-off
reference somewhere (median first difference at token 29). Two runs of the same configuration
without PipeSpec diverged just as much (36 of 36, median token 34), so this engine is not
run-to-run deterministic and the comparison cannot isolate PipeSpec. GSM8K-250 stayed in its
usual band (84.8%), and the texts are coherent.

## Risks and open items

1. **Stale MTP state at chunk-A positions.** On a valid chunk B, the MTP layer never runs on
   chunk A's own positions, so they keep KV written by the previous draft chain, and the MTP's
   indexer keys there are never written. Below 2,048 tokens of context this is only an
   approximation; at long agentic contexts stale indexer keys could push good positions out of
   the MTP's top-2,048 and lower acceptance. `CMP_PIPESPEC_REFILL=1` always re-runs the
   drafter on chunk A (exact, but gives back part of the gain). Not yet measured at long
   context.
2. **Four-stream aggregate.** The first paired run showed 99 vs 132.8 tok/s at 4 streams; the
   later production-stack pair showed 128.5 vs 132.0. PipeSpec should be inactive whenever more
   than one request runs, so the first number is suspect, but it has not been re-measured.
3. **Exposed host work.** After the host sync on stage 4, chunk B's launch is on the critical
   path; each extra millisecond costs about one point of gain.
4. **Metrics.** vLLM's spec-decode counters mix chunk A and B; use the `PipeSpec:` log lines
   (A-full rate, B-valid rate, tokens per epoch).
5. **Ported files.** `pipespec_model_runner.py`, `pipespec_pp_utils.py` and
   `pipespec_ar_speculator.py` are the image's files plus marked PipeSpec blocks; re-port them
   for a new image ([patches/diffs](../patches/diffs)).

## Related work

- PipeSpec (arXiv 2505.01572): asynchronous cascade of separate models with rollback; same
  "don't wait for the verifier" idea, but its stages are models, not layer slices.
- PipeDec / SpecPipe (2504.04104), FlowSpec (2507.02620): draft trees feeding a deep
  pipeline for a single request.
- Speculative Pipeline Decoding (2605.30852), PPSD (2509.19368): drafting from intermediate
  layers inside a pipeline.
- PipeSwift (2609.16491): pipeline parallelism + MTP for agent serving.
- vLLM RFC #11945: V1 pipeline parallelism schedules a decoding request in only one in-flight
  micro-batch "until speculative decoding is handled"; PipeSpec relaxes that for a lone request.
