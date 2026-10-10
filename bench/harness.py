#!/usr/bin/env python3
"""Frozen speed harness for GLM-5.3 serving candidates (stdlib only).

usage: harness.py [BASE] [--tag NAME] [--reps 3] [--decode-tokens 256] [--prefill 2048,16384]
Emits one JSON verdict line on stdout:
  decode_c1_tps (median of reps), itl_p50_ms / itl_p99_ms, decode_c4_agg_tps,
  prefill_<N>_tps + ttft_<N>_s per prompt size (cold: cache-busted).
Token counts come from server usage accounting; the first 5 streamed tokens are
dropped from decode timing (streamlib default). One warmup request is discarded.
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streamlib import build_request, cache_bust_tag, measure_sse  # noqa: E402

MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")
UNIT = ("The observatory stood above a frozen valley; its instruments measured "
        "starlight, wind and ice through the long polar night. ")


def complete(base, prompt, max_tokens, ignore_eos=True):
    body = build_request({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0.0, "ignore_eos": ignore_eos})
    req = urllib.request.Request(f"{base}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        s = measure_sse(r, started_at=t0, emit_warnings=False)
    return s


def prompt_tokens(base, text):
    req = urllib.request.Request(f"{base}/tokenize", data=json.dumps(
        {"model": MODEL, "prompt": text}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["count"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--prefill", default="2048,16384")
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args()
    out = {"tag": a.tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    complete(a.base, f"[{cache_bust_tag('warm')}] Say hi.", 8)  # discarded warmup

    tps, itls = [], []
    for _ in range(a.reps):
        s = complete(a.base, f"[{cache_bust_tag('dec')}] Write a long story about a lighthouse keeper.\n\n",
                     a.decode_tokens)
        tps.append(s.decode_tps)
        itls.extend(s.itls[5:])
    out["decode_c1_tps_reps"] = [round(x, 3) for x in tps]
    out["decode_c1_tps"] = round(statistics.median(tps), 3)
    itls.sort()
    out["itl_p50_ms"] = round(1e3 * itls[len(itls) // 2], 2)
    out["itl_p99_ms"] = round(1e3 * itls[min(len(itls) - 1, int(0.99 * len(itls)))], 2)

    if a.conc > 1:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(a.conc) as ex:
            rs = list(ex.map(lambda i: complete(
                a.base, f"[{cache_bust_tag('conc')}] Story {i}: a ship in a storm.\n\n", a.decode_tokens),
                range(a.conc)))
        wall = time.perf_counter() - t0
        out[f"decode_c{a.conc}_agg_tps"] = round(sum(r.completion_tokens for r in rs) / wall, 3)
        out[f"decode_c{a.conc}_per_stream_tps"] = [round(r.decode_tps, 2) for r in rs]

    for n in [int(x) for x in a.prefill.split(",") if x]:
        text = f"[{cache_bust_tag('pf')}] " + UNIT * max(1, n // 30)
        n_tok = prompt_tokens(a.base, text)
        s = complete(a.base, text + "\nSummarize in one word:", 1)
        out[f"prefill_{n}_prompt_tokens"] = n_tok
        out[f"ttft_{n}_s"] = round(s.ttft, 3) if s.ttft else None
        out[f"prefill_{n}_tps"] = round(n_tok / s.ttft, 1) if s.ttft else None
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
