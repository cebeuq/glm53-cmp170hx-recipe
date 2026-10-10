#!/usr/bin/env python3
"""Cold long-context prefill ladder to localize a context-length crash.

Sends nested prompts (each rung extends the previous one, so prefix caching
makes the total cost ~ the largest rung) and reports per rung: prompt tokens,
time to first token, and whether the server survived.

usage: ctx_crash_ladder.py BASE_URL rung1 rung2 ...   (rungs in tokens)
"""
import json, os, sys, time, urllib.request, urllib.error

BASE = sys.argv[1]
RUNGS = [int(x) for x in sys.argv[2:]]
MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")
UNIT = "The observatory stood above a frozen valley; its instruments measured starlight, wind and ice. "

def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def ntok(text):
    return len(post("/tokenize", {"model": MODEL, "prompt": text})["tokens"])

unit_tok = ntok(UNIT * 50) / 50.0
print(f"unit ~{unit_tok:.2f} tok", flush=True)
for rung in RUNGS:
    reps = int(rung / unit_tok)
    prompt = UNIT * reps + f"\n\nSummarize the passage in five words."
    n = ntok(prompt)
    t0 = time.time()
    try:
        r = post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 8, "temperature": 0})
        dt = time.time() - t0
        print(f"rung {rung}: prompt_tokens={r['usage']['prompt_tokens']} ok in {dt:.1f}s -> {r['choices'][0]['text']!r}", flush=True)
    except urllib.error.HTTPError as e:
        print(f"rung {rung}: prompt_tokens~{n} HTTP {e.code} after {time.time()-t0:.1f}s  <-- CRASH?", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"rung {rung}: prompt_tokens~{n} error {type(e).__name__}: {e} after {time.time()-t0:.1f}s", flush=True)
        sys.exit(1)
