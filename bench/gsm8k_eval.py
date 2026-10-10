#!/usr/bin/env python3
"""GSM8K task-level eval (stdlib only): 5-shot raw completions, greedy, exact final-number match.

usage: gsm8k_eval.py BASE DATA_JSON OUT_JSON [--n 250] [--conc 8]
Problems 0-4 are the fixed few-shot exemplars; problems 5..5+n-1 are scored. Same prompts for
every config, so accuracy deltas compare engine/quant changes, not prompt variance.
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.request

MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")
CALC = re.compile(r"<<[^>]*>>")
FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def norm(x):
    x = x.replace(",", "").strip().rstrip(".")
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return x


def extract(text):
    m = FINAL.search(text)
    if m:
        return norm(m.group(1))
    nums = NUM.findall(text)
    return norm(nums[-1]) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("data"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--conc", type=int, default=4)
    a = ap.parse_args()
    rows = json.load(open(a.data))
    shots = "".join(f"Question: {r['q']}\nAnswer: {CALC.sub('', r['a'])}\n\n" for r in rows[:5])
    items = rows[5:5 + a.n]

    def run(i_r):
        i, r = i_r
        body = {"model": MODEL, "prompt": shots + f"Question: {r['q']}\nAnswer:", "max_tokens": 320,
                "temperature": 0.0, "stop": ["\n\nQuestion:", "Question:"]}
        req = urllib.request.Request(f"{a.base}/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    text = json.load(resp)["choices"][0]["text"]
                break
            except Exception as e:  # noqa: BLE001
                text = f"ERROR {e}"
                time.sleep(5)
        gold = norm(FINAL.search(r["a"]).group(1))
        pred = extract(text)
        return {"i": i, "gold": gold, "pred": pred, "ok": pred == gold, "text": text[-300:]}

    t0 = time.time()
    with cf.ThreadPoolExecutor(a.conc) as ex:
        res = sorted(ex.map(run, enumerate(items)), key=lambda x: x["i"])
    acc = sum(r["ok"] for r in res) / len(res)
    out = {"n": len(res), "correct": sum(r["ok"] for r in res), "accuracy": round(acc, 4),
           "errors": sum(r["text"].startswith("ERROR") for r in res), "seconds": round(time.time() - t0, 1),
           "results": res}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n", "correct", "accuracy", "errors", "seconds")}))


if __name__ == "__main__":
    main()
