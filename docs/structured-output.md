# Structured output with speculative decoding

Overlay: `engine/compose.grammar.yaml`, after `compose.pipespec.yaml` in `COMPOSE_FILE`. There
is no switch: the mounted files act only on requests that carry a grammar (`response_format`
with `json_schema` or `json_object`, a named `tool_choice`).

## Symptom

With a draft method active (MTP k=2, with or without PipeSpec), requests with
`response_format: json_schema` or a named `tool_choice` could end in HTTP 500. The engine stayed
up; the request ended as `FINISHED_ERROR` and the log said
`Unexpected: grammar rejected tokens [...]`. On the contributor's machine it was 4 of 24 requests
of a structured-output test set with PipeSpec, 3 of 24 without it and none with speculation off.

## Cause

Async scheduling (on by default with MTP) schedules a decode step with `[-1] * k` placeholder
drafts. The engine core defers sampling, and fills in the real, grammar-validated drafts, only
while the request still has output placeholders at schedule time. Under pipeline parallelism a
request's decode steps are `pp_size` steps apart (`next_decode_eligible_step = step + pp_size`,
batch queue depth = pp), so the previous output of a lone request is already consumed when it
is scheduled again. Sampling is then not deferred and the grammar bitmask is built from the
placeholders: the draft rows allow every token and the bonus row carries the grammar state from
before the drafts. The worker verifies its real drafts against those rows and can commit tokens
the grammar would not have allowed; when it does, the next bitmask fill rejects them and the
request fails.

This is the race of vLLM issue #54437 (structured output, speculative decoding, async
scheduling and PP together).

## Fix

A port of vLLM #54442 ("Don't let a structured-output request sample from an unmasked row")
onto this image, and onto the recipe's PipeSpec scheduler and model runner:

- `GrammarOutput.num_acceptable_drafts` (`grammar_sched_output.py`): per request, how many
  leading drafts the bitmask really constrained. `grammar_scheduler.py` fills it with the
  number of non-placeholder drafts it scheduled.
- `grammar_invalid_drafts` (`grammar_gpu_structured_outputs.py`, carried through
  `grammar_batch_shard.py` and `grammar_model_runner.py`): a mask over the logit rows whose
  draft met a permissive bitmask row.
- `grammar_rejection_sampler.py` verifies a copy of the drafts in which those are -1, so the
  accepted length stops before the first unconstrained row and the token sampled there comes
  from a constrained row.
- `grammar_structured_output_init.py`: when the grammar rejects a draft it was handed, the
  bitmask fill logs the request and reuses the last valid row for the rest of the window. The
  image's file raised `AssertionError` in the engine core there.

`grammar_scheduler.py` and `grammar_model_runner.py` are `pipespec_scheduler.py` and
`pipespec_model_runner.py` plus 14 changed lines each, so `compose.pipespec.yaml` has to be in
the chain; set `CMP_PIPESPEC=0` there if PipeSpec itself is not wanted.

## Measured

A contributed measurement, 2026-11-04, on a second 8x CMP 170HX server (not the machine in
[hardware](hardware.md)): the production stack (TP2 x PP4, P2P, int8 KV, MTP k=2 + PipeSpec,
524K context, 8 sequences) plus the pipeline hand-off fix (proposed separately).

- `bench/struct_probe.py` (json_schema strict, named tool_choice, json_object and an
  unconstrained control), 10 requests at concurrency 1 and 24 at concurrency 8: no 5xx, and every
  constrained reply parses and matches its schema. (The failures above were seen with an earlier
  test set; this probe was not run without the overlay.)
- Requests without a grammar are unchanged within run-to-run noise: one stream on a code
  prompt, three runs, 94.5 / 89.7 / 94.3 tok/s with the overlay and 86.5 / 87.3 / 88.2 without;
  17K-token cold prefill 2,234 and 2,210 tok/s; PipeSpec counters as before (chunk A fully
  accepted 0.54-0.55, chunk B valid 0.37-0.39, 3.15-3.23 tokens per epoch).
- Cost: a constrained request whose bitmask came from placeholders gets its drafts invalidated
  for that step and advances one token. At concurrency 1 the constrained replies ran at 42 tok/s
  (named tool_choice), 49 and 70 (two json_schema prompts) and 60 (json_object); an unconstrained
  one ran at 75.

## Limits

- The cost above is the known price of this fix under PP. vLLM #63219 later reverted #54442
  for it and fixed the race at its source (per-request draft retention in
  `DraftTokensHandler`). That change is not ported here; it would give constrained requests
  their drafts back.
- Not measured: grammar requests with `CMP_FAIR_CHUNK`, logprobs together with drafts, and
  anything on the machine in [hardware](hardware.md).

## Check

```bash
python3 bench/struct_probe.py http://127.0.0.1:8000 glm-5.3 1 10 c1
python3 bench/struct_probe.py http://127.0.0.1:8000 glm-5.3 8 3 c8
```

Exit code 0 and `verdict=PASS` mean no 5xx and every constrained reply valid. To disable, remove
`:compose.grammar.yaml` from `COMPOSE_FILE` and restart.
