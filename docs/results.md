# Results

All numbers below were measured on one server with 8x CMP 170HX (64 GB each, PCIe Gen2 x16),
GPUs capped at 160 W with the SM clock at up to 1,695 MHz, between 2026-10-08 and 2026-10-10,
with the hybrid checkpoint (exact NVFP4 routed experts + INT8-g128 residue) unless noted.
How each number is produced: [Methodology](methodology.md).

Two speed measurements appear throughout:

- **Chat** (`bench/accept_eval.py`): 36 chat prompts in three categories (16 coding,
  16 web-browsing agent tasks, 4 prose), one at a time, temperature 0, thinking on,
  384 tokens each. Reported as mean streaming tok/s per category. This is the primary
  single-stream metric.
- **Harness** (`bench/harness.py`): a raw-completion prose prompt ("a long story about a
  lighthouse keeper"), 256 tokens, median of 3; plus 4 concurrent streams and cold prefill.
  Prose is the hardest category for speculation, so harness single-stream numbers sit
  below the chat numbers, and they are noisy (repeats of one config spread by up to ±10%).

## Production

TP2 x PP4 (layers 22,20,20,16) + PCIe P2P with vLLM custom all-reduce + int8-tile latent KV
+ stock MTP k=2 + PipeSpec ([engine/prod.vars.example](../engine/prod.vars.example)).

| | 524,288 ctx, 8 seqs, util 0.97 (production) | 358,400 ctx, 4 seqs, util 0.96 |
|---|---:|---:|
| Chat, 1 stream: coding / browse / prose | not re-measured (see note) | **68.3-68.6 / 72.8-74.9 / 59.9-60.0 tok/s** |
| Harness, 1 stream (raw prose) | 52.1 tok/s | 54.2 tok/s |
| Harness, 4 streams, aggregate | - | 128.5 tok/s |
| Harness, 8 streams, aggregate | **135.5 tok/s** (18-22 per stream) | - |
| Time to first token, 16K prompt | 8.2 s | - |
| KV pool | **569,792 tokens** | 520,384 tokens |
| Longest prompt served (context ladder) | 519,451 tokens, answered in 415 s | 358,027 tokens, 245 s |
| GSM8K-250 (5-shot, greedy) | - | 84.8%, 82.8%, 87.6% (three runs) |
| Wall power, mean / peak | 1,784 / 2,236 W (harness) | 1,928 / 2,340 W (soak) |

Ranges are two measurements of the same configuration (candidate run and a later soak).
The production column changes only the context length, sequence count and memory target;
its chat speed was not measured separately. On the raw-prose harness it was 4% lower
(52.1 vs 54.2 tok/s), so expect chat numbers a few percent below the right-hand column.

## How we got there (single stream)

Each row adds one change to the row above. Chat = coding / browse / prose tok/s.

| Step | Context | Harness 1-stream | Chat | Note |
|---|---|---:|---|---|
| PP8, no speculation, no P2P | 512K | 31.0 | - | first fast baseline |
| TP2 x PP4 + NCCL LL protocol, no speculation | 350K | **34.4** | - | production before the work below |
| + MTP k=1, stock head, **PP embedding fix** | 128K | 47.9 | - | acceptance 86.2% (22% before the fix) |
| MTP k=2 | 128K | 51.7 | 56.4 / 59.4 / 51.9 | |
| + drafter local argmax (P2P driver installed, P2P not used) | 128K | 54.9 | 58.2 / 58.5 / 53.1 | |
| + NCCL over P2P | 128K | 57.6 | 60.6 / 62.4 / 52.3 | |
| + vLLM custom all-reduce over P2P | 128K | 60.2 | 65.3 / 67.8 / 58.4 | KV pool 170,688 |
| + int8-tile latent KV, layers 22,20,20,16 | 350K | 58.2 | 64.7 / 67.0 / 58.4 | KV pool 520,384; no speed cost |
| + PipeSpec | 350K | 54.2 | **68.6 / 72.8 / 59.9** | GSM8K-250 84.8%, 358K ladder pass |
| same, soak run | 350K | - | 68.3 / 74.9 / 60.0 | |
| 512K, 8 seqs, util 0.97 | 512K | 52.1 | - | production; 8-stream aggregate 135.5 |

From 34.4 tok/s to roughly 60-75 tok/s on chat depending on content, i.e. about 2x, while the
context went from 350K to 512K.

## Where the single-stream gain came from

| Change | Effect on chat tok/s | Paired comparison |
|---|---|---|
| MTP embedding fix under PP | k=1 acceptance 22% -> 86%; enables speculation at all | TP8 (no PP) already had 85% |
| P2P, NCCL only | +4-7% | vs same driver, P2P disabled |
| P2P + custom all-reduce | +12-16% | vs same driver, P2P disabled |
| int8-tile latent KV (vs BF16) | 0% (64.7/67.0/58.4 vs 65.3/67.8/58.4) | 350K instead of 128K |
| PipeSpec | +6% coding / +9-13% browse / +3-5% prose | flag on vs off, same boot |

## Layout comparison

No speculation, no P2P, 160 W, 32K context unless noted (harness numbers):

| Layout | 1 stream | 4 streams | Prefill (13K prompt) | KV pool |
|---|---:|---:|---:|---:|
| PP8 (512K) | 31.0 | 82.5 | 1,517 tok/s | 599,488 |
| PP8, prefill throttle off (512K) | 31.0 | 96.9 | 1,728 tok/s | 599,488 |
| TP8 | 28.7 | 82.2 | 542 tok/s | 93K |
| TP2 x PP4, NCCL Simple | 29.4 | 72.9 | 1,137 tok/s | 400K |
| TP2 x PP4, NCCL LL (350K) | 34.4 | 76.6 | 1,597 tok/s | 399,680 |

With MTP k=2 at 128K:

| Layout | Interconnect | 1 stream | Chat | 4 streams | KV pool |
|---|---|---:|---|---:|---:|
| TP4 x PP2 (42,36) | no P2P | 30.1 * | - | 81.6 | - |
| TP4 x PP2 (42,36) | P2P + custom all-reduce | 61.1 | 67.0 / 70.5 / 61.9 | 124.3 | 140,032 |
| TP2 x PP4 (21,20,20,17) | P2P + custom all-reduce | 60.2 | 65.3 / 67.8 / 58.4 | - | 170,688 |
| TP8 | P2P + custom all-reduce across sockets | 15.1 | 16.0 (coding) | - | 32K only |

\* measured with a self-distilled MTP head, which performed the same as the stock head in the engine ([MTP](mtp.md)).

TP4 x PP2 is as fast as TP2 x PP4 once P2P works, but each TP4 stage replicates the MLA latent
on 4 GPUs, so its KV pool tops out around 240K tokens even with 8-bit KV. TP2 x PP4 is the
layout that reaches 512K. TP8 needs all-reduces across the two CPU sockets, which are very slow
with custom all-reduce ([comm](comm.md)).

## Quality

| Check | Result |
|---|---|
| Hybrid checkpoint vs original NVFP4, teacher-forced top-1 agreement (PP8) | 96.4% (gate: >= 95%); NLL on fixed text within the +2% budget |
| INT4-g128 experts + INT8 residue vs original NVFP4 (rejected) | 93.65% agreement: fail |
| Hybrid, TP2 x PP4 vs NVFP4 goldens | 93.9% agreement (see the open question in [known limits](known-limits.md)) |
| GSM8K-250, hybrid PP8 / TP2 x PP4 / TP8 (no speculation) | 83.2% / 84.8% / 86.8% |
| GSM8K-250, production stack at 350K | 84.8%, 82.8%, 87.6% |
| Context ladder | 65K, 131K, 262K, 519K prompts answered coherently, no Xid |

GSM8K-250 has a standard error of about ±2.4 points; differences inside 83-88% are noise.
Greedy output is not reproducible run to run on this engine, even with the same
configuration (36 of 36 prompts diverged at some token, median first difference at token 34),
so exact-match comparisons cannot be used as a gate; PipeSpec on vs off diverged no more than
two runs of the same configuration (median token 29).

## Power regimes

The same software at two power settings (PP8, hybrid, 512K, no speculation):

| GPU cap | SM clock | 1 stream | 4 streams | Prefill (13K) |
|---|---|---:|---:|---:|
| 100 W | locked at 450 MHz | 11.0 | 35.3 | 607 tok/s |
| 160 W | up to 1,695 MHz | 31.0 | 82.5 | 1,517 tok/s |

At 450 MHz a single GPU reaches only 0.55-0.70 TB/s on decode-shaped matrix-vector products,
against 1.37-1.41 TB/s at 1,200 MHz: these cards need SM clock, not just power, to read HBM at
full speed. The HBM clock is fixed at 1,728 MHz either way. See [hardware](hardware.md).

## Fair chunked prefill (experimental)

`CMP_FAIR_CHUNK` ([how it works](how-it-works.md#fair-chunked-prefill-experimental)) is being
validated with `bench/multiagent_stall.py` (3 streams decoding when a ~17K-token prompt
arrives; worst stall per stream and the new prompt's time to first token).

Without the cap, on an otherwise idle server, the three decoding streams fell from 13.2 to 0.42
streamed chunks per second while the long prompt prefilled, with a worst gap of 5.6 s between
chunks (the long prompt's own time to first token: 14.5 s). A clean A/B with the cap is
pending; a first run with `CMP_FAIR_CHUNK=384` was taken under unrelated live load and is not
comparable.

With `CMP_FAIR_CHUNK=384`, measured under heavy unrelated load (4-8 other requests running), the
three decoding streams still got 1.59 chunks per second while the long prompt prefilled, and the
worst gap was 1.5 s (vs 5.6 s without the cap on an idle server). The cap only applies while
something is decoding, so a lone prompt and single-stream decode are unaffected. A clean A/B on an
idle server is still to do.
