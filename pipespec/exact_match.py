#!/usr/bin/env python3
"""PipeSpec correctness check: greedy completions must be identical with and
without PipeSpec (stdlib only; run against a live server, one request at a time).

  python3 -I pipespec/exact_match.py run BASE PROMPTS_JSONL OUT_JSON [--max-tokens 256] [--n 36]
  python3 -I pipespec/exact_match.py compare REF_JSON NEW_JSON

T=0 target sampling is argmax, so the committed tokens do not depend on which
drafts were verified, only on the KV/positions being right. A divergence means
a PipeSpec state bug (or, rarely, a non-row-invariant kernel: the report shows
the first differing token for inspection). Exit 1 on any mismatch.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")


def complete(base, prompt, max_tokens):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "logprobs": False,
        "return_token_ids": True,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    dt = time.perf_counter() - t0
    ch = out["choices"][0]
    msg = ch.get("message") or {}
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\x00" + (
        msg.get("content") or ""
    )
    ids = ch.get("token_ids")
    n = (out.get("usage") or {}).get("completion_tokens")
    return {"text": text, "token_ids": ids, "completion_tokens": n, "seconds": dt}


def run(args):
    prompts = [json.loads(line) for line in open(args.prompts) if line.strip()]
    prompts = prompts[: args.n]
    res = []
    for i, p in enumerate(prompts):
        r = complete(args.base, p["prompt"], args.max_tokens)
        r["i"] = i
        r["cat"] = p.get("cat")
        res.append(r)
        tps = (r["completion_tokens"] or 0) / max(r["seconds"], 1e-9)
        print(f"{i:3d} {p.get('cat', ''):8s} {r['completion_tokens']} tok {tps:6.1f} tok/s",
              file=sys.stderr)
    json.dump(res, open(args.out, "w"))


def compare(args):
    a = {r["i"]: r for r in json.load(open(args.ref))}
    b = {r["i"]: r for r in json.load(open(args.new))}
    bad = 0
    for i in sorted(set(a) & set(b)):
        ra, rb = a[i], b[i]
        ta, tb = ra.get("token_ids"), rb.get("token_ids")
        if ta is not None and tb is not None:
            same = ta == tb
            pos = next((j for j, (x, y) in enumerate(zip(ta, tb)) if x != y),
                       min(len(ta), len(tb)))
        else:
            same = ra["text"] == rb["text"]
            pos = next((j for j, (x, y) in enumerate(zip(ra["text"], rb["text"])) if x != y),
                       min(len(ra["text"]), len(rb["text"])))
        if not same:
            bad += 1
            unit = "token" if ta is not None and tb is not None else "char"
            print(f"MISMATCH prompt {i} ({ra.get('cat')}): first differing {unit} {pos}")
    n = len(set(a) & set(b))
    ta = sum(r["completion_tokens"] or 0 for r in a.values())
    tb = sum(r["completion_tokens"] or 0 for r in b.values())
    sa = sum(r["seconds"] for r in a.values())
    sb = sum(r["seconds"] for r in b.values())
    print(json.dumps({"prompts": n, "mismatches": bad,
                      "ref_tok_s": round(ta / sa, 2), "new_tok_s": round(tb / sb, 2)}))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("base")
    r.add_argument("prompts")
    r.add_argument("out")
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--n", type=int, default=36)
    c = sub.add_parser("compare")
    c.add_argument("ref")
    c.add_argument("new")
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
