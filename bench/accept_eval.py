#!/usr/bin/env python3
"""In-engine MTP acceptance + single-stream speed per prompt category (stdlib only).

usage: accept_eval.py BASE PROMPTS_JSONL OUT_JSON [--max-tokens 384]
Prompts run one at a time (chat template, thinking on by default, temperature 0, ignore_eos), so the
/metrics spec-decode counter deltas per category belong to that category alone.
Prompt set (bench/accept_prompts.jsonl): fresh coding prompts, browse tasks from public benchmarks (held out of our MTP distillation set), prose.
"""
import argparse
import json
import os
import re
import time
import urllib.request

MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")
METRIC = re.compile(r'^vllm:spec_decode_num_(draft_tokens|accepted_tokens|accepted_tokens_per_pos)_total\{([^}]*)\}\s+(\S+)$')


def counters(base):
    out = {}
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            m = METRIC.match(line)
            if m:
                pos = re.search(r'position="(\d+)"', m.group(2))
                key = m.group(1) + (f"_{pos.group(1)}" if pos else "")
                out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def chat(base, prompt, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first, n = None, 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                n = ev["usage"]["completion_tokens"]
            ch = ev.get("choices") or []
            if ch and first is None and (ch[0].get("delta") or {}):
                first = time.perf_counter()
    end = time.perf_counter()
    return n, (n - 1) / (end - first) if first and n > 1 and end > first else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("prompts"); ap.add_argument("out")
    ap.add_argument("--max-tokens", type=int, default=384)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.prompts)]
    chat(a.base, "Say hi.", 8)  # warmup
    res = {}
    for cat in dict.fromkeys(r["cat"] for r in rows):
        c0, tps = counters(a.base), []
        for r in (r for r in rows if r["cat"] == cat):
            tps.append(chat(a.base, r["prompt"], a.max_tokens)[1])
        c1 = counters(a.base)
        d = {k: c1.get(k, 0.0) - c0.get(k, 0.0) for k in c1}
        drafts = d.get("draft_tokens", 0.0)
        k = sum(1 for key in d if key.startswith("accepted_tokens_per_pos_")) or 1
        steps = drafts / k if drafts else 0.0
        res[cat] = {
            "n": len(tps), "decode_tps_mean": round(sum(tps) / len(tps), 2),
            "accept_rate": round(d.get("accepted_tokens", 0.0) / drafts, 4) if drafts else None,
            "tokens_per_step": round(1 + d.get("accepted_tokens", 0.0) / steps, 3) if steps else None,
            "per_pos": [round(d.get(f"accepted_tokens_per_pos_{i}", 0.0) / steps, 4) if steps else None
                        for i in range(k)],
        }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
