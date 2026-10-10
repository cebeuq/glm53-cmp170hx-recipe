# Methodology

All tools are in [`bench/`](../bench) and use only the Python standard library; run them from
any machine that can reach the API. `GLM53_MODEL` sets the model id (default `glm-5.3`).

## Speed

**`bench/accept_eval.py BASE bench/accept_prompts.jsonl OUT.json`** (the chat numbers):

- 36 prompts in three categories: 16 coding tasks written for this benchmark, 16 web-browsing
  agent tasks (taken from the public DeepSearchQA, AA-Omniscience and GDPval sets), 4 prose
  prompts.
- `/v1/chat/completions`, one request at a time, temperature 0, the model's default chat
  template (thinking on), `ignore_eos`, 384 tokens.
- Per request: (completion tokens - 1) / (end of stream - first streamed chunk).
  Reported: the mean per category.
- Speculation statistics per category from the `/metrics` counter deltas
  (`vllm:spec_decode_num_accepted_tokens_per_pos_total` and friends): acceptance rate, tokens
  per step and per-position acceptance. Do not trust `tokens_per_step` under adaptive depth or
  PipeSpec (both change the drafts per step).

**`bench/harness.py BASE --tag NAME`** (the harness numbers), one JSON line:

- one discarded warm-up request;
- single stream: raw `/v1/completions` "Write a long story about a lighthouse keeper.",
  256 tokens, temperature 0, `ignore_eos`, median of 3, the first 5 streamed tokens dropped,
  token counts from the server's usage accounting, inter-token latency p50 / p99;
- `--conc 4` (default): 4 different prompts at once, aggregate = completion tokens / wall time;
- cold prefill at ~2K and ~16K tokens (cache-busted), reported as time to first token and
  prompt tokens / TTFT.

Prompts carry a random tag so prefix caching never hits.

**`bench/multiagent_stall.py BASE OUT.json`**: N streams (default 3) decode; after 6 s a
~17K-token prompt arrives. Reports per stream the streamed chunks per second before and during
the long prefill and the worst gap between chunks, plus the long prompt's time to first token
alone and under contention. Used for fair chunked prefill.

## Quality

- **`bench/gsm8k_eval.py BASE DATA OUT --n 250 --conc 4`**: GSM8K, 5-shot raw completions,
  greedy, exact final-number match. Problems 0-4 of the test split are the fixed exemplars,
  5-254 are scored. Get the data with `python3 bench/fetch_gsm8k.py` (first 300 problems of
  `openai/grade-school-math` `test.jsonl`, in file order). Standard error at n=250 is about
  ±2.4 points.
- **`bench/quality_gate.py capture|compare`**: teacher-forced comparison against goldens
  captured once from a reference server: top-1 agreement on greedy continuations of code,
  math and reasoning prompts (pass >= 95%), NLL on fixed text (within +2%), free-run collapse
  (distinct 4-grams), no NaN/inf. It scores the server's logprobs, so it compares weights and
  kernels, not sampling.
- **`bench/conc_gate.py BASE`**: rounds of concurrent requests, all answers checked for errors,
  emptiness and corruption. Run it after any change to speculation, caching or kernels: a past
  speculative-decoding bug corrupted output only under concurrency.
- **`bench/ctx_crash_ladder.py BASE 65536 262144 520000`**: nested long prompts (each rung
  extends the previous one, so prefix caching keeps the total cost near the largest rung);
  reports tokens, TTFT and the answer, and fails on any server error.
- **`pipespec/exact_match.py`**: greedy completions compared token by token between two
  servers. On this engine two runs of one configuration already differ (median first difference
  at token 34), so it is informative only.

## Running candidates

[`loop/`](../loop) runs one configuration at a time on the server, always the same way:

```bash
sudo bash loop/guard.sh &                 # better: its own terminal, or setsid
bash loop/run_candidate.sh NAME engine/variants/no-pipespec.vars --no-quality --accept --gsm8k 250
```

`run_candidate.sh` validates the compose config, restarts the engine with the vars file, waits
for `/health`, records the KV pool size and startup time from the log, runs the harness, then
the optional gates (`--accept`, `--gsm8k N`, `--capture` / quality compare, `--conc-gate`,
`--ladder`, `--exact`), checks the kernel log for new Xid messages, and appends one JSON row to
`RESULTS.ndjson`. `loop/queue.sh` runs a list of candidates (`loop/queues/example.txt`);
`loop/devlock.sh` runs a GPU microbenchmark on chosen GPUs while the engine is down;
`loop/guard.sh` stops everything on Xid / bus errors and, optionally, on a wall-power limit read
from the BMC (`POWER_CMD`, `KILL_W`).

## Hygiene

- Change one thing per candidate and compare against a control measured in the same session.
- Single-stream harness numbers vary by up to ±10% between repeats of one configuration; chat
  numbers by about ±2-3%. Small differences are noise.
- Keep the power cap and clock settings fixed and recorded; they move every number.
- Discard the first request after a boot (CUDA graph and JIT warm-up).
