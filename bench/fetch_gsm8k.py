#!/usr/bin/env python3
"""Download the GSM8K test split and write the first N problems in gsm8k_eval.py's format.

usage: fetch_gsm8k.py [OUT_JSON] [--n 300]
Source: openai/grade-school-math (MIT), grade_school_math/data/test.jsonl.
The published GSM8K-250 numbers use problems 0-4 as few-shot exemplars and score 5-254 of
this file (the first 300 test problems, in file order). Stdlib only.
"""
import argparse
import json
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/openai/grade-school-math/master/"
       "grade_school_math/data/test.jsonl")


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("out", nargs="?", default=os.path.join(here, "data", "gsm8k_test_300.json"))
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    with urllib.request.urlopen(URL, timeout=120) as r:
        rows = [json.loads(line) for line in r.read().decode().splitlines() if line.strip()]
    items = [{"q": x["question"], "a": x["answer"]} for x in rows[:a.n]]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(items, f)
    print(f"wrote {len(items)} problems to {a.out}")


if __name__ == "__main__":
    main()
