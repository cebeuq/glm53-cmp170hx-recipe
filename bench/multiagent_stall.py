#!/usr/bin/env python3
"""Multi-agent prefill/decode interference: N streams decode, then one long prompt arrives.

usage: multiagent_stall.py BASE OUT_JSON [--decoders 3] [--prompt-tokens 17000] [--decode-max 900]
Reports per decoder: tok/s before the long prompt, worst gap between streamed chunks and tok/s while the
long prompt prefills; the long prompt's TTFT; and the same long prompt's TTFT alone (solo, no decoders).
"""
import argparse, json, os, threading, time, urllib.request, uuid

MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")
UNIT = ("The observatory stood above a frozen valley; its instruments measured "
        "starlight, wind and ice through the long polar night. ")


def stream(base, messages, max_tokens, rec):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "reasoning_effort": "low"}
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    rec["t_send"] = time.perf_counter()
    rec["chunks"] = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            ch = ev.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") or (ch and (ch[0].get("delta") or {}).get("reasoning")):
                rec["chunks"].append(time.perf_counter())
    rec["t_end"] = time.perf_counter()


def long_prompt(n_tokens):
    return f"[{uuid.uuid4().hex}] " + UNIT * max(1, n_tokens // 19) + "\nSummarize the text above in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--decoders", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=17000)
    ap.add_argument("--decode-max", type=int, default=900)
    a = ap.parse_args()
    res = {}
    # Solo long-prompt TTFT (no decoders).
    solo = {}
    stream(a.base, [{"role": "user", "content": long_prompt(a.prompt_tokens)}], 8, solo)
    res["solo_ttft_s"] = round(solo["chunks"][0] - solo["t_send"], 2) if solo["chunks"] else None
    time.sleep(2)
    # Decoders.
    recs = [dict() for _ in range(a.decoders)]
    ths = [threading.Thread(target=stream, args=(a.base, [{"role": "user", "content": f"[{uuid.uuid4().hex}] Write a very long story about lighthouse keeper number {i}."}], a.decode_max, recs[i])) for i in range(a.decoders)]
    for t in ths:
        t.start()
    while not all(r.get("chunks") for r in recs):
        time.sleep(0.05)
    time.sleep(6)
    longrec = {}
    lt = threading.Thread(target=stream, args=(a.base, [{"role": "user", "content": long_prompt(a.prompt_tokens)}], 8, longrec))
    lt.start(); lt.join()
    t0 = longrec["t_send"]; t1 = longrec["chunks"][0] if longrec.get("chunks") else longrec["t_end"]
    res["contended_ttft_s"] = round(t1 - t0, 2)
    for t in ths:
        t.join()
    per = []
    for r in recs:
        c = r["chunks"]
        before = [x for x in c if t0 - 5 <= x < t0]
        during = [x for x in c if t0 <= x <= t1]
        win = [x for x in c if t0 - 0.0 <= x <= t1 + 0.5]
        gaps = [b - a_ for a_, b in zip(win, win[1:])] or [t1 - t0]
        if during:
            first_gap = during[0] - max([x for x in c if x < t0] or [t0])
            gaps.append(first_gap)
        per.append({"chunks_per_s_before": round(len(before) / 5, 2),
                    "chunks_per_s_during_prefill": round(len(during) / max(t1 - t0, 1e-6), 2),
                    "worst_gap_s": round(max(gaps), 2)})
    res["decoders"] = per
    res["worst_gap_s"] = max(p["worst_gap_s"] for p in per)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
