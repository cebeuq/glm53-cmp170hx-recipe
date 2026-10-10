#!/usr/bin/env python3
"""Batch-1 decode-shaped bandwidth probe for one GPU (CUDA events only; CUPTI is fused off).

Reports effective HBM GB/s for:
  copy   : device-to-device tensor copy (read+write counted)
  gemv   : bf16 [1,K] x [K,N] matmul (weight bytes counted) - the residue/attention path
  sum    : bf16 reduction over a large tensor (pure read)
Shapes are GLM-5.3 decode shapes (hidden 6144). Run once per SM-clock setting.
"""
import json
import sys

import torch

dev = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = False
REPS, WARM = 50, 10


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPS):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / REPS / 1e3  # seconds per call


out = {"tag": sys.argv[1] if len(sys.argv) > 1 else "", "gpu": torch.cuda.get_device_name(0)}

a = torch.empty(512 * 2**20, dtype=torch.uint8, device=dev)  # 512 MiB
b = torch.empty_like(a)
t = timed(lambda: b.copy_(a))
out["copy_GBps"] = round(2 * a.numel() / t / 1e9, 1)

big = torch.randn(256 * 2**20, dtype=torch.bfloat16, device=dev)  # 512 MiB
t = timed(lambda: big.sum())
out["sum_GBps"] = round(big.numel() * 2 / t / 1e9, 1)

for name, (k, n) in {
    "gemv_q_b_6144x24576": (6144, 24576),     # q-proj-like, 302 MB
    "gemv_o_16384x6144": (16384, 6144),        # o_proj-like, 201 MB
    "gemv_shared_6144x4096": (6144, 4096),     # shared-expert-like, 50 MB
    "gemv_lmhead_6144x154880": (6144, 154880), # lm_head, 1.9 GB
}.items():
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
    x = torch.randn(1, k, dtype=torch.bfloat16, device=dev)
    t = timed(lambda: torch.nn.functional.linear(x, w))
    out[name + "_GBps"] = round(w.numel() * 2 / t / 1e9, 1)
    out[name + "_us"] = round(t * 1e6, 1)
    del w

print(json.dumps(out))
