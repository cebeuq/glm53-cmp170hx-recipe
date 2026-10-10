#!/usr/bin/env python3
"""Per-layer byte census of a safetensors checkpoint from headers only (no tensor reads).

Usage: layer_bytes.py MODEL_DIR  -> JSON on stdout: per-layer bytes, per-category bytes,
and the active-per-token bytes estimate (routed experts scaled by topk/n_experts).
"""
import json, re, struct, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
idx = json.loads((d / "model.safetensors.index.json").read_text())
layer_re = re.compile(r"^model\.layers\.(\d+)\.")
per_layer = defaultdict(int)
cat = defaultdict(int)
dtypes = defaultdict(int)
for shard in sorted(set(idx["weight_map"].values())):
    with open(d / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        b = meta["data_offsets"][1] - meta["data_offsets"][0]
        dtypes[meta["dtype"]] += b
        m = layer_re.match(name)
        key = int(m.group(1)) if m else "nonlayer:" + name.split(".")[0 if not name.startswith("model.") else 1]
        per_layer[key] += b
        if ".mlp.experts." in name:
            c = "routed_experts"
        elif ".shared_experts." in name:
            c = "shared_experts"
        elif ".indexer." in name or ".indexers_proj" in name:
            c = "indexer"
        elif ".self_attn." in name:
            c = "attention"
        elif ".mlp.gate." in name:
            c = "router"
        elif ".mlp." in name:
            c = "dense_mlp"
        elif "embed_tokens" in name:
            c = "embed"
        elif "lm_head" in name:
            c = "lm_head"
        else:
            c = "other"
        if m and int(m.group(1)) >= cfg["num_hidden_layers"]:
            c = "mtp:" + c
        cat[c] += b
topk, ne = cfg.get("num_experts_per_tok", 8), cfg.get("n_routed_experts", 256)
active = sum(v for k, v in cat.items() if not k.startswith("mtp") and k != "embed" and k != "routed_experts")
active += cat["routed_experts"] * topk / ne
print(json.dumps({
    "num_hidden_layers": cfg["num_hidden_layers"],
    "total_bytes": sum(per_layer.values()),
    "category_GiB": {k: round(v / 2**30, 2) for k, v in sorted(cat.items())},
    "dtype_GiB": {k: round(v / 2**30, 2) for k, v in dtypes.items()},
    "active_per_token_GB": round(active / 1e9, 2),
    "per_layer_GiB": {str(k): round(v / 2**30, 3) for k, v in sorted(per_layer.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))},
}, indent=1))
