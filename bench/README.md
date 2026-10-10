# Benchmarks and gates

Standard-library Python; point them at any OpenAI-compatible endpoint. `GLM53_MODEL` sets the
model id (default `glm-5.3`). Protocols and caveats: [docs/methodology.md](../docs/methodology.md).

| Script | What it measures |
|---|---|
| `harness.py [BASE]` | single-stream decode (raw prose), 4-stream aggregate, cold prefill / TTFT |
| `accept_eval.py BASE accept_prompts.jsonl OUT` | chat tok/s and MTP acceptance per category (coding / browse / prose) |
| `gsm8k_eval.py BASE DATA OUT --n 250` | GSM8K, 5-shot, greedy (get DATA with `fetch_gsm8k.py`) |
| `quality_gate.py capture\|compare\|canary` | teacher-forced agreement and NLL against goldens |
| `conc_gate.py BASE` | corruption under concurrent requests |
| `ctx_crash_ladder.py BASE RUNG...` | nested long prompts up to the context limit |
| `multiagent_stall.py BASE OUT` | decode stalls while a long prompt prefills |
| `streamlib.py` | shared SSE timing helpers |

`accept_prompts.jsonl`: 16 coding prompts written for this benchmark, 16 web-browsing agent
tasks from the public DeepSearchQA, AA-Omniscience and GDPval sets (their ids are kept in the
file), and 4 prose prompts.
