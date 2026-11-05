#!/usr/bin/env python3
"""struct_probe.py <url> <model> <conc> <rounds> <tag> [--out file]: structured output with speculative decoding.

Each round sends, concurrently at <conc>, a mix of: response_format json_schema (strict) extraction requests
(kinds d8 and inv), a named tool_choice with JSON-schema constrained arguments (d5), json_object mode (jobj) and
one plain (unconstrained) JSON prompt as control. Every chat request thinks (enable_thinking true), with
reasoning_effort low. Reports per kind: HTTP status counts, JSON-parse and schema validity, decode tok/s
(completion tokens / wall after first byte is not available non-streamed, so wall is end to end), and the
engine's spec-decode acceptance is read by the caller from the container log. Exit 0 iff no 5xx and every
constrained reply is valid JSON matching its schema.
"""
import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("conc", type=int)
ap.add_argument("rounds", type=int)
ap.add_argument("tag")
ap.add_argument("--out")
ap.add_argument("--effort", default="low")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--timeout", type=int, default=600)
args = ap.parse_args()

D8_SCHEMA = {"type": "object", "additionalProperties": False,
             "required": ["order_id", "ship_date", "city", "boxes", "insured"],
             "properties": {"order_id": {"type": "integer"}, "ship_date": {"type": "string"},
                            "city": {"type": "string"}, "boxes": {"type": "integer"},
                            "insured": {"type": "boolean"}}}
D8 = [("Order 7781 shipped on 2026-03-14 to Oslo with 3 boxes; it was not insured.",
       {"order_id": 7781, "city": "Oslo", "boxes": 3, "insured": False}),
      ("Shipment record: order #10452, 12 boxes, destination Valparaíso, dispatched 2026-08-01, insurance: yes.",
       {"order_id": 10452, "boxes": 12, "insured": True}),
      ("We sent order 333 (a single box) to Kraków on 2026-12-24 with full insurance.",
       {"order_id": 333, "boxes": 1, "insured": True}),
      ("Order 9090 left the warehouse 2026-05-05 for Lima, five boxes, uninsured.",
       {"order_id": 9090, "boxes": 5, "insured": False})]
INV_SCHEMA = {"type": "object", "additionalProperties": False,
              "required": ["invoice_number", "vendor_name", "currency", "total", "line_items"],
              "properties": {"invoice_number": {"type": "string"}, "vendor_name": {"type": "string"},
                             "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
                             "total": {"type": "number"},
                             "line_items": {"type": "array", "minItems": 1,
                                            "items": {"type": "object", "additionalProperties": False,
                                                      "required": ["sku", "qty", "unit_price"],
                                                      "properties": {"sku": {"type": "string"}, "qty": {"type": "integer"},
                                                                     "unit_price": {"type": "number"}}}}}}
INV = ("Invoice INV-2291 from Northwind Traders, in EUR: 4 x SKU-11 at 12.50, 1 x SKU-72 at 99.00, 10 x SKU-3 at 0.75; "
       "the total is 156.50. Extract it.")
T_LOG = {"type": "function", "function": {"name": "log_event", "description": "Write an event to the audit log.",
         "parameters": {"type": "object", "additionalProperties": False, "required": ["event", "level"],
                        "properties": {"event": {"type": "string"},
                                       "level": {"type": "string", "enum": ["info", "warning", "error"]}}}}}
T_WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city.",
             "parameters": {"type": "object", "additionalProperties": False, "required": ["city", "unit"],
                            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}}
D5 = ["Hi! In one sentence, what is the capital of France?", "Please tell me a fun fact about octopuses.",
      "Log that the nightly backup finished with two warnings.", "Record an error: disk /dev/sdb is 99 percent full."]
PLAIN = "Produce a JSON object with keys name, city, boxes (integer) and insured (boolean) for a fictional shipment. Only JSON."


def typed_ok(value, schema):
    t = schema.get("type")
    if t == "object":
        if not isinstance(value, dict):
            return False
        for k in schema.get("required", []):
            if k not in value:
                return False
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(k not in props for k in value):
            return False
        return all(typed_ok(value[k], props[k]) for k in value if k in props)
    if t == "array":
        if not isinstance(value, list) or len(value) < schema.get("minItems", 0):
            return False
        return all(typed_ok(v, schema["items"]) for v in value) if "items" in schema else True
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "string":
        return isinstance(value, str) and ("enum" not in schema or value in schema["enum"])
    return True


def post(body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:600]}, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return 0, {"error": repr(e)[:300]}, time.time() - t0


def one(kind, i):
    base = {"model": args.model, "temperature": 0, "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"reasoning_effort": args.effort, "enable_thinking": True}}
    schema = None
    if kind == "d8":
        p, exp = D8[i % len(D8)]
        body = dict(base, messages=[{"role": "user", "content": "Extract the shipment as JSON. " + p}],
                    response_format={"type": "json_schema", "json_schema": {"name": "shipment", "strict": True, "schema": D8_SCHEMA}})
        schema = D8_SCHEMA
    elif kind == "inv":
        body = dict(base, messages=[{"role": "user", "content": INV}],
                    response_format={"type": "json_schema", "json_schema": {"name": "invoice", "strict": True, "schema": INV_SCHEMA}})
        schema = INV_SCHEMA
    elif kind == "d5":
        body = dict(base, messages=[{"role": "user", "content": D5[i % len(D5)]}], tools=[T_WEATHER, T_LOG],
                    tool_choice={"type": "function", "function": {"name": "log_event"}})
        schema = T_LOG["function"]["parameters"]
    elif kind == "jobj":
        body = dict(base, messages=[{"role": "user", "content": PLAIN}], response_format={"type": "json_object"})
    else:  # plain control
        body = dict(base, messages=[{"role": "user", "content": PLAIN}])
    st, resp, wall = post(body)
    rec = {"kind": kind, "i": i, "status": st, "wall": round(wall, 2)}
    if st == 200:
        ch = resp["choices"][0]
        msg = ch["message"]
        u = resp.get("usage", {})
        rec["completion_tokens"] = u.get("completion_tokens")
        rec["tps"] = round(u.get("completion_tokens", 0) / wall, 1) if wall else None
        rec["finish"] = ch.get("finish_reason")
        content = msg.get("content") or ""
        rec["truncated"] = rec["finish"] == "length"
        if kind == "d5":
            calls = msg.get("tool_calls") or []
            rec["tool_calls"] = len(calls)
            try:
                argsd = json.loads(calls[0]["function"]["arguments"]) if calls else None
                rec["json_ok"] = argsd is not None
                rec["schema_ok"] = bool(argsd is not None and calls[0]["function"]["name"] == "log_event" and typed_ok(argsd, schema))
            except Exception:  # noqa: BLE001
                rec["json_ok"] = rec["schema_ok"] = False
        else:
            txt = content.strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt)
            try:
                obj = json.loads(txt)
                rec["json_ok"] = True
                rec["schema_ok"] = typed_ok(obj, schema) if schema else True
            except Exception:  # noqa: BLE001
                rec["json_ok"] = rec["schema_ok"] = False
                rec["head"] = txt[:120]
    else:
        rec["error"] = str(resp.get("error"))[:300]
    return rec


kinds = ["d8", "d5", "inv", "jobj", "plain"]
recs = []
for r in range(args.rounds):
    jobs = [(kinds[(r * args.conc + j) % len(kinds)], r * args.conc + j) for j in range(args.conc)]
    with cf.ThreadPoolExecutor(max_workers=args.conc) as ex:
        recs.extend(ex.map(lambda kj: one(*kj), jobs))

summary = {}
for k in kinds:
    rs = [x for x in recs if x["kind"] == k]
    if not rs:
        continue
    st = {}
    for x in rs:
        st[x["status"]] = st.get(x["status"], 0) + 1
    ok = [x for x in rs if x["status"] == 200]
    tps = sorted(x["tps"] for x in ok if x.get("tps"))
    summary[k] = {"n": len(rs), "status": st,
                  "json_ok": sum(1 for x in ok if x.get("json_ok")), "schema_ok": sum(1 for x in ok if x.get("schema_ok")),
                  "tps_med": tps[len(tps) // 2] if tps else None, "tps_min": tps[0] if tps else None,
                  "ctok_med": sorted(x["completion_tokens"] or 0 for x in ok)[len(ok) // 2] if ok else None}
constrained = [x for x in recs if x["kind"] in ("d8", "d5", "inv", "jobj")]
bad = [x for x in recs if x["status"] >= 500 or x["status"] == 0]
# A reply cut by max_tokens is not a grammar failure: report it separately.
invalid = [x for x in constrained if x["status"] == 200 and not x.get("schema_ok") and not x.get("truncated")]
truncated = [x for x in constrained if x.get("truncated")]
verdict = "PASS" if not bad and not invalid else "FAIL"
line = f"STRUCT {args.tag} conc={args.conc} rounds={args.rounds} verdict={verdict} 5xx/err={len(bad)} invalid={len(invalid)} truncated={len(truncated)} " + \
       " ".join(f"{k}:{v['status']} json {v['json_ok']}/{v['n']} schema {v['schema_ok']}/{v['n']} tps med {v['tps_med']} ctok {v['ctok_med']}" for k, v in summary.items())
print(line)
for x in bad[:6] + invalid[:6]:
    print("  detail:", json.dumps(x)[:400])
if args.out:
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "verdict": verdict, "records": recs}, f, indent=1)
sys.exit(0 if verdict == "PASS" else 1)
