# MTP speculative decoding

GLM-5.3 ships one multi-token-prediction layer (layer 78): a full DSA attention + MoE layer
that takes the model's last hidden state and the embedding of the next token and predicts the
token after it. vLLM uses it as a drafter: it proposes k tokens, the main model verifies
1 + k tokens in one forward pass, and every accepted draft is a token for free. In this
checkpoint the MTP layer is BF16 (18.5 GiB, mostly its 256 experts) and lives on the last
pipeline stage.

## The bug: the drafter had no embedding table under pipeline parallelism

First measurements were bad: k=1 accepted 22% of drafts (break-even was ~35%), so speculation
made decoding slower. We worked through the possible causes one at a time:

| Test | Acceptance (k=1) | Conclusion |
|---|---:|---|
| PP8, MTP experts quantized to INT4 | 22% | |
| PP8, stock BF16 MTP layer | 21.7% | not the INT4 experts |
| Offline: the same MTP layer in PyTorch on captured hidden states | 80.6% top-1 | not the abliteration; the head itself is fine |
| PP8, eager mode (no CUDA graphs) | 19% | not CUDA graphs |
| **TP8, no pipeline parallelism** | **85.2%** | **the bug is specific to PP** |

Root cause, in the fork's `models/deepseek_v32/nvidia/mtp.py`: GLM-5.x has no copy of
`embed_tokens` inside layer 78; the table exists once, at the top level of the checkpoint. The
MTP `load_weights` skipped top-level weights. Normally that is harmless, because
`maybe_share_target_embed()` points the drafter at the target model's embedding. Under PP,
however, the last stage does not hold the embedding (it is a `PPMissingLayer` there), so the
sharing function returns early, and the drafter embedded every draft position with an
uninitialized table.

The fix (`patches/vllm/v32_mtp.py`, 11 lines) loads `model.embed_tokens.weight` into the
drafter's own table, as upstream vLLM's `deepseek_mtp.py` does. Cost: a second embedding copy
on the last stage (~0.95 GB per GPU at TP2). Result on TP2 x PP4: acceptance 86.2% at k=1.

If you run another DeepSeek-V3.2-style model with MTP under pipeline parallelism, check the
acceptance rate in `/metrics` (`vllm:spec_decode_num_accepted_tokens_total` /
`..._draft_tokens_total`): a value near 20% is this bug.

## Choosing k

TP2 x PP4, before P2P, one stream, raw prose (measured with the self-distilled head described
below, which performed like the stock head in the engine):

| k | Step time | Tokens per step | tok/s |
|---|---:|---:|---:|
| 0 | 29.0 ms | 1 | 34.4 |
| 1 | 37.0 ms | 1.79 | 47.0 |
| 2 | 42.2 ms | 2.29 | 54.2 |
| 3 | 47.3 ms | 2.56 | 54.1 |

The first draft costs ~8 ms per step, each further one ~5 ms (mostly verifying one more token:
up to 8 more routed experts per MoE layer). k=3 tied k=2 on one stream and was worse at 4
streams; on chat prompts with P2P, k=3 was also no better than k=2 (coding 67.1 vs 67.0,
browse 70.5 vs 70.5 tok/s, TP4 x PP2). Production uses k=2.

Per-position acceptance with k=2 (stock head, chat, T=0, conditional on the previous draft
being accepted): coding 0.81 / 0.73, browse 0.84 / 0.81, prose 0.81 / 0.68.

## Things that did not help

- **Self-distilled head.** We captured ~790K rows of the served model's own hidden states on
  synthetic agentic coding and browsing episodes (prompts built from public benchmarks: DeepSWE, Terminal-Bench, DeepSearchQA, AA-Omniscience, GDPval) and retrained the small parts of layer 78 (routed
  experts frozen) with a KL loss against the model's own next-token distribution. Offline
  top-1 went from 0.806 to 0.850. In the engine it gained nothing (distilled vs stock: coding
  56.3 vs 56.4, browse 57.9 vs 59.4 tok/s). The offline holdout was split by episode but shared a corpus and
  near-duplicate tasks with training, so the offline gain was leakage.
- **Chained-step training** (training draft steps 2..3 on the head's own previous output, as
  vLLM runs it for k>1): offline expected accepted length per k=3 draft 1.851 -> 1.985; in the
  engine no gain over the stock head (chained vs stock at k=2: coding 67.3 vs 67.0, browse 69.7
  vs 70.5, prose 61.0 vs 61.9).
- **Adaptive draft depth** (per-request acceptance statistics choose k in 1..4 every step,
  one k per batch so CUDA graphs stay uniform): coding 64.7 / browse 69.7 / prose 57.2 vs fixed
  k=2 at 67.0 / 70.5 / 61.9. Not adopted.
- **INT4 MTP experts** (to free ~6.6 GiB per GPU on the last stage): tested only before the
  embedding fix; not re-tested since.

The drafter's own cost matters: one MTP forward reads ~1.6 GB per GPU at TP2, of which
`lm_head` (the full 154,880-row vocabulary, half per GPU) is ~0.95 GB.
`use_local_argmax_reduction` (each rank takes the argmax over its own vocabulary half instead of
all-gathering the logits) is a free fork option; it was on for most measurements above, but
PipeSpec production runs without it.
