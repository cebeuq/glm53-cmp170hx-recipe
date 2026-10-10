#!/usr/bin/env python3
"""KV-token capacity per PP stage for GLM-5.3 (+MTP) by KV dtype, layout and layer partition.

Calibrated on startup logs (util 0.96, BF16 KV, stock BF16 MTP layer 78 on the last stage):
  tp2pp4 (P2P custom AR, MTP k=2), partition 21,20,20,17, KV in use 10.46/6.68/6.68/3.40 GiB, pool 170,688
          (same without P2P: 10.31/6.55/6.55/3.26 GiB, 163,712)
  tp4pp2 (P2P custom AR, MTP k=2), partition 42,36, KV in use 6.69/5.73 GiB, pool 140,032
Per token per stage = MLA layers x row + indexer layers x 132 B (layers 0-2, 6, 10, ..., 74, and MTP).
The MLA latent is NOT sharded by TP, so TP4 stages hold ~2x the layers' KV of TP2 stages.
Moving one MoE layer between stages shifts 53.29/20 GiB (TP2) or half that (TP4) of weights per rank.

usage: kv_capacity.py [--layout tp2pp4|tp4pp2] [--partition 22,20,20,16] [--extra-gib 0,0,0,0]
                      [--util-delta 0.01]
"""

import argparse

GIB = 1024**3
ROW = {"bf16": 1152, "fp8": 576, "fp8_ds_mla / int8-tile": 656}
IDX = 132  # indexer K cache: 128 fp8 + 4 B scale
NLAYERS = 78
GPU_GIB = 63.4
LAYOUTS = {
    "tp2pp4": dict(part=[21, 20, 20, 17], kv=[10.46, 6.68, 6.68, 3.40], layer_gib=53.29 / 20),
    "tp4pp2": dict(part=[42, 36], kv=[6.69, 5.73], layer_gib=53.29 / 40),
}


def has_indexer(i):
    return i < 3 or (i - 3) % 4 == 3 or i >= NLAYERS  # MTP layer always has one


def capacity(layout, part, extra, util_delta, mtp=True, verbose=True):
    L = LAYOUTS[layout]
    assert sum(part) == NLAYERS and len(part) == len(L["part"])
    start = 0
    res = {k: [] for k in ROW}
    for s, n in enumerate(part):
        layers = list(range(start, start + n))
        last = s == len(part) - 1
        if last and mtp:
            layers.append(NLAYERS)
        start += n
        kv = L["kv"][s] - (n - L["part"][s]) * L["layer_gib"] + extra[s] + util_delta * GPU_GIB
        nidx = sum(has_indexer(i) for i in layers)
        for k, row in ROW.items():
            per_tok = len(layers) * row + nidx * IDX
            res[k].append(int(kv * GIB / per_tok) // 64 * 64)
        if verbose:
            print(f"  PP{s}: {n} layers{' + MTP' if last and mtp else ''}, indexers {nidx}, "
                  f"KV mem {kv:.2f} GiB")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="tp2pp4", choices=list(LAYOUTS))
    ap.add_argument("--partition", default=None)
    ap.add_argument("--extra-gib", default=None, help="freed GiB per stage (e.g. MTP quant)")
    ap.add_argument("--util-delta", type=float, default=0.0, help="util change vs 0.96")
    a = ap.parse_args()
    L = LAYOUTS[a.layout]
    part = [int(x) for x in a.partition.split(",")] if a.partition else L["part"]
    extra = [float(x) for x in a.extra_gib.split(",")] if a.extra_gib else [0.0] * len(part)
    print(f"{a.layout} partition {','.join(map(str, part))} util {0.96 + a.util_delta:.2f}")
    res = capacity(a.layout, part, extra, a.util_delta)
    for k, v in res.items():
        print(f"    {k:22s}: per-stage {v} -> pool {min(v):,}")


if __name__ == "__main__":
    main()
