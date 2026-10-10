"""Strict concurrent text gate, compatible with BASE N ROUNDS MAX_TOKENS.

`4 25` means 25 rounds of four requests. Request errors, malformed/empty/short
answers, and corruption fail the process. Only successful answers are scored.
This does not replace the separate tool, image, or long-context quality gates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.request
import uuid

MODEL = os.environ.get("GLM53_MODEL", "glm-5.3")
PROMPTS = [
    "Explain in detail how a modern jet engine works, section by section.",
    "Write a thorough explanation of how the TCP three-way handshake works.",
    "Describe how a refrigerator moves heat, step by step, with the thermodynamics.",
    "Explain how public-key cryptography works and why it is secure.",
]


class GateError(ValueError):
    """The endpoint did not return a valid successful generation."""


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def temperature_value(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("temperature must be finite and nonnegative")
    return number


def sampling_settings(temperature=0.0):
    return {"temperature": temperature_value(temperature), "top_p": .95, "reasoning_effort": "max"}


def request_payload(prompt, max_tokens, model=MODEL, temperature=0.0):
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, **sampling_settings(temperature), "stream": False}


def validate_usage(usage):
    if not isinstance(usage, dict):
        raise GateError("missing server usage accounting")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        number = usage.get(key)
        if type(number) is not int or number < 0:
            raise GateError(f"invalid usage.{key}")
    if usage["completion_tokens"] <= 0:
        raise GateError("server reports zero completion tokens")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise GateError("inconsistent server usage total")
    return usage


def validate_response(body, *, min_words=8, min_completion_tokens=8):
    if not isinstance(body, dict) or body.get("error") is not None:
        raise GateError("error or non-object response")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise GateError("expected exactly one response choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") not in ("stop", "length"):
        raise GateError("missing or unsuccessful finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise GateError("missing assistant message")
    if message.get("tool_calls") or message.get("function_call"):
        raise GateError("unexpected tool/function call in a text-only gate")
    fields = {}
    for key in ("content", "reasoning", "reasoning_content"):
        value = message.get(key)
        if value is not None and not isinstance(value, str):
            raise GateError(f"message.{key} is not text")
        fields[key] = value or ""
    reasoning = fields["reasoning"] or fields["reasoning_content"]
    text = reasoning + fields["content"]
    if not text.strip():
        raise GateError("empty generated answer")
    usage = validate_usage(body.get("usage"))
    if usage["completion_tokens"] < min_completion_tokens:
        raise GateError("answer has too few server-counted completion tokens")
    if len(text.split()) < min_words:
        raise GateError("answer has too few words for the corruption gate")
    return {"text": text, "content": fields["content"], "reasoning": reasoning,
            "usage": usage, "finish_reason": choice["finish_reason"]}


def chat(base, prompt, max_tokens, *, model=MODEL, timeout=180.0,
         min_words=8, min_completion_tokens=8, temperature=0.0):
    body = request_payload(prompt, max_tokens, model, temperature)
    request = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=min(timeout, 30.0)) as response:
        if response.status != 200:
            raise GateError(f"HTTP {response.status}")
        chunks, received = [], 0
        while True:
            if time.monotonic() - started > timeout:
                raise GateError("response exceeded total request deadline")
            chunk = response.read1(64 * 1024)
            if not chunk:
                break
            received += len(chunk)
            if received > 16 * 1024 * 1024:
                raise GateError("response exceeds 16 MiB")
            chunks.append(chunk)
        raw = b"".join(chunks)
    result = validate_response(json.loads(raw), min_words=min_words,
                               min_completion_tokens=min_completion_tokens)
    result["wall_s"] = time.monotonic() - started
    result["request"] = body
    return result


def d4(text):
    words = text.split()
    if len(words) < 8:
        raise GateError("text too short to compute meaningful diversity")
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / len(grams)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def run_gate(base, streams, rounds, max_tokens, *, model=MODEL, timeout=180.0,
             min_words=8, min_completion_tokens=8, tag=None, out=None, request_fn=None, temperature=0.0):
    request_fn = request_fn or chat
    tag = tag or uuid.uuid4().hex
    summary = {"status": "running", "tag": tag, "base": base, "model": model,
               "streams_per_round": streams, "requested_rounds": rounds,
               "max_tokens": max_tokens, "rounds": [], "errors": 0,
               "corrupt": 0, "completed_streams": 0, "successful_streams": 0,
               "threshold": 0.90, "sampling_settings": sampling_settings(temperature)}
    started = time.monotonic()
    if out:
        write_json(out, summary)
    with ThreadPoolExecutor(max_workers=streams) as pool:
        for rnd in range(rounds):
            futures = {}
            for index in range(streams):
                prompt = f"[gate {tag}] {PROMPTS[index % len(PROMPTS)]} (round {rnd} stream {index})"
                future = pool.submit(request_fn, base, prompt, max_tokens, model=model,
                                     timeout=timeout, min_words=min_words,
                                     min_completion_tokens=min_completion_tokens,
                                     temperature=temperature)
                futures[future] = (index, request_payload(prompt, max_tokens, model, temperature))
            records = []
            for future in as_completed(futures):
                index, request = futures[future]
                record = {"stream": index, "request": request}
                summary["completed_streams"] += 1
                try:
                    result = future.result()
                    score = d4(result["text"])
                    record.update(result)
                    record.update(status="pass" if score >= 0.90 else "corrupt", d4=score)
                    summary["successful_streams"] += 1
                    summary["corrupt"] += score < 0.90
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    summary["errors"] += 1
                records.append(record)
            records.sort(key=lambda record: record["stream"])
            summary["rounds"].append({"round": rnd, "streams": records})
            print("round %d: %s" % (rnd, " ".join(
                f"{r['d4']:.3f}" if "d4" in r else f"ERROR[{r['error']}]" for r in records)), flush=True)
            if out:
                write_json(out, summary)
            if summary["errors"] or summary["corrupt"]:
                break
    summary["wall_s"] = time.monotonic() - started
    summary["status"] = "fail" if summary["errors"] or summary["corrupt"] else "pass"
    scores = [r["d4"] for item in summary["rounds"] for r in item["streams"] if "d4" in r]
    summary["worst_d4"] = min(scores) if scores else None
    if out:
        write_json(out, summary)
    print(f"RESULT status={summary['status']} streams={summary['completed_streams']} "
          f"errors={summary['errors']} bad(<0.90)={summary['corrupt']} "
          f"worst_d4={summary['worst_d4']}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8000")
    parser.add_argument("streams", nargs="?", type=positive_int, default=4)
    parser.add_argument("rounds", nargs="?", type=positive_int, default=6)
    parser.add_argument("max_tokens", nargs="?", type=positive_int, default=300)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=positive_float, default=180.0)
    parser.add_argument("--temperature", type=temperature_value, default=0.0)
    parser.add_argument("--min-words", type=positive_int, default=8)
    parser.add_argument("--min-completion-tokens", type=positive_int, default=8)
    parser.add_argument("--tag")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.streams > 4:
        parser.error("S9 gate supports at most four simultaneous streams")
    if args.min_words < 8:
        parser.error("diversity gate needs at least eight words")
    if args.out and args.out.exists():
        parser.error("output already exists; choose a new filename")
    try:
        result = run_gate(args.base, args.streams, args.rounds, args.max_tokens,
                          model=args.model, timeout=args.timeout, min_words=args.min_words,
                          min_completion_tokens=args.min_completion_tokens, tag=args.tag, out=args.out,
                          temperature=args.temperature)
    except KeyboardInterrupt:
        print("Gate interrupted; no pass result.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Gate failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
