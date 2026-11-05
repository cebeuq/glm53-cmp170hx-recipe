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
  identical in distribution (validity is token equality with the sampled bonus; drafts are
  greedy unless [probabilistic drafts](#probabilistic-drafts-sampled-requests) are on).
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
exactly k=2 drafts (greedy or probabilistic), no structured output / logprobs / LoRA, and at least 16 tokens left
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

## Probabilistic drafts (sampled requests)

Switch: `"draft_sample_method":"probabilistic"` in the speculative config, for example
`--speculative-config '{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}'`.
Unset means greedy drafts, and every file behaves as before.

With greedy drafts the verifier treats a draft as a one-hot proposal: at temperature T > 0 it
accepts draft d with probability p(d) and otherwise resamples from p without d. That is exact,
but wasteful where p is flat, because the drafter can only ever propose its argmax. The fork
already has the lossless alternative outside PipeSpec: the drafter samples d from its own
distribution q at the request temperature (Gumbel), and the rejection sampler gets q, accepts
with probability min(1, p(d) / q(d)) and resamples from max(p - q, 0). PipeSpec's gates excluded
it (`draft_sample_method == "greedy"`, `draft_logits is None`). Three changes lift that:

- `pipespec_core.py`: the gate accepts `probabilistic`. Chunk B's validity rule (A fully
  accepted and A's bonus token == d3) does not depend on how the drafts were drawn.
- `pipespec_ar_speculator.py`: the draft-logits cache gets k + (k + 1) = 5 columns, so the
  chain steps d3, d4, d5 keep their logits (8 x 5 x vocabulary in bf16, about 12 MB).
- `pipespec_model_runner.py`: before chunk B is verified, the logits of its drafts (chain steps
  k+1..2k) are moved to the columns the rejection sampler reads. When an A-without-B step
  restores stashed drafts, which were drawn for other positions, they get one-hot logits
  (q(d) = 1): the greedy rule, which is exact for a fixed draft.

Chunk A and chunk B are then both verified against the distribution that produced their
drafts, and the committed tokens are distributed as without speculation. The model-runner part
is needed for that: an overlay that mounts another `model_runner.py` after
`compose.pipespec.yaml` has to carry the same two hunks, otherwise chunk B is verified against
the wrong draft logits and sampling is no longer exact.
`pipespec/check_sampling.py` (CPU, stdlib) checks this by Monte Carlo on a toy model: greedy
and sampled drafts, standard and block verification, under the chunk A/B protocol; the
frequency of every committed sequence has to match its target probability (chi-square and
maximum error against bounds). `docker compose -f pipespec/compose.yaml run --rm check-sampling`

`CMP_SQ_GREEDY_D3=1` (optional, default off) draws the chain's bonus guess d3 as the argmax
while d1, d2, d4 and d5 stay sampled; d3 is never verified, so this is exact as well. It
measured no gain (pooled acceptance 1.130 against 1.124 for greedy drafts) and stays off.

### Measured (contributed)

Not measured on the machine in [hardware](hardware.md). Measured 2026-11-04 on a second 8x CMP
170HX server: the production stack (TP2 x PP4, P2P, int8 KV, MTP k=2 + PipeSpec, 524K context,
8 sequences) plus the pipeline hand-off fix and the grammar bitmask fix for drafts (both proposed
separately, on in both arms), greedy against probabilistic drafts on the same build and weights. "Sampled" means temperature 0.9, top_p 0.9,
top_k 80, presence penalty 0.5, repetition penalty 1.02. Acceptance is accepted draft tokens
per verified 2-draft chunk, from the engine's spec-decode counters.

| | greedy drafts | probabilistic drafts |
|---|---:|---:|
| Acceptance, sampled, 6 chat prompts x 1,024 tokens | 1.191 | 1.252 |
| Acceptance, sampled, long-context chat (83K-token prompt, 2 turns x 2,000 tokens) | 1.226 | 1.385 |
| Acceptance, sampled, raw prose completions (6 x 384 tokens) | 0.829 | 0.929 |
| Acceptance, the three pooled (about 5,700 chunks per arm) | 1.124 | 1.223 (+8.8%) |
| One stream, sampled, code prompt, tok/s (3 runs) | 92.5 / 91.4 / 92.0 | 94.6 / 94.6 / 93.3 |
| One stream, sampled, prose prompt, tok/s (3 runs) | 61.8 / 58.6 / 59.4 | 62.0 / 61.3 / 61.8 |
| Raw prose completions, tok/s | 50.0 | 56.9 |
| Long-context chat, mean of the two turns, tok/s (one run) | 68.4 | 77.2 |
| Temperature 0: acceptance / code tok/s | 1.303 / 93.1 | 1.305 / 93.9 |
| Quality gate at temperature 0, teacher-forced agreement | 90.13% | 90.07% |
| NLL of the sampled text (6 x 384 tokens) | 0.632 +- 0.045 | 0.699 +- 0.040 |

Read it with the noise in mind. On this engine a reload of the same configuration moves
acceptance by about +-4% and one single-stream tok/s number by about +-5% (greedy output is not
reproducible run to run), every tok/s row is three runs or fewer, and the sampled-text NLL
probe itself moves between 0.61 and 0.70 run to run. The acceptance gain has the same sign on
all three workloads. The tok/s rows are within that noise, except the raw-prose pair (+14%),
which is one sample per arm and did not repeat in the second measurement below.

A second measurement, on another layout of the same engine (a pair of RTX PRO 6000 Blackwell
cards as an extra first stage in front of the eight CMPs, TP2 x PP5, greedy and probabilistic
arms back to back): pooled sampled acceptance 1.059 against 1.147 (+8%), raw prose +12%
accepted drafts, sampled chat prompts unchanged (1.212 against 1.207), nothing on code, nothing
at four concurrent streams. Long-context chat turns read +8 to +20% there, but that workload
alone spreads further than that between reruns of one configuration. Every tok/s difference was
inside the noise (raw prose 59.3 tok/s greedy, 57.5 probabilistic). On that layout the three
files of this section without the grammar bitmask fix passed the quality gate (90.38%) and ran
the same single-stream speeds as with it.

In short: about 8 to 9% more accepted drafts on sampled requests, pooled, in two separate
comparisons, coming from prose-like text, and no throughput gain that either comparison can
separate from the noise. It does not change anything at temperature 0 (a Gumbel draw at T = 0
is the argmax) or the chunk B valid rate (0.32-0.34 either way). On prose the limit is the
drafter itself, which accepts 0.72 / 0.47 per position against 0.75 / 0.55 on code.

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
