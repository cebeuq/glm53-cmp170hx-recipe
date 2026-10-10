#!/usr/bin/env python3
"""Decode-size all-reduce latency: NCCL (PyNccl) vs vLLM CustomAllreduce.

One process per GPU (like vLLM TP workers). Each op is captured K times in one CUDA graph and
replayed R times (decode runs under CUDA graphs), so the number is per-call latency including
the in-graph launch gap. Every replay is checked for exact sums.

Run inside the p2ptools image (engine image + tools), engine DOWN:
  python3 /tests/ar_bench.py --gpus 0,1
  python3 /tests/ar_bench.py --gpus 0,1,2,3 --force-pcie-custom-ar
NCCL transport follows the caller's env (NCCL_P2P_DISABLE=1 = today's host-staged path).
--force-pcie-custom-ar mimics the engine override needed for TP>2 on PCIe (see docs/p2p.md):
vLLM otherwise disables CustomAllreduce for >2 GPUs without NVLink.
"""
import argparse
import contextlib
import json
import os
import socket
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 6144  # GLM-5.3 hidden size; bf16 bytes per token = 12288
DEFAULT_SIZES = [12288 * t for t in (1, 2, 3, 4, 6, 8, 12, 16, 32)]  # 12 KiB .. 384 KiB


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def time_graph(fn, inp, k, r, capture_ctx=None):
    """Capture k calls in one graph (inside capture_ctx, e.g. CustomAllreduce.capture(), which
    registers the graph's IPC buffers when it exits), then replay r times outside it."""
    g = torch.cuda.CUDAGraph()
    outs = []
    with capture_ctx if capture_ctx is not None else contextlib.nullcontext():
        with torch.cuda.graph(g):
            for _ in range(k):
                outs.append(fn(inp))
    torch.cuda.synchronize()
    g.replay()  # warm
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(r):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / (k * r), outs


def worker(rank, world, gpus, sizes, k, r, port, force_fc, no_custom, queue):
    try:
        os.environ.setdefault("VLLM_SKIP_P2P_CHECK", "0")  # real IPC write/read check, not just the driver bit
        torch.cuda.set_device(gpus[rank])
        dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
        group = dist.group.WORLD
        from vllm.platforms import current_platform
        if force_fc:
            type(current_platform).is_fully_connected = classmethod(lambda cls, ids: True)
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        dev = torch.device(f"cuda:{gpus[rank]}")
        nccl = PyNcclCommunicator(group, device=dev)
        ca = None if no_custom else CustomAllreduce(group, dev)
        can_peer = [torch.cuda.can_device_access_peer(gpus[rank], g) for g in gpus if g != gpus[rank]]
        rows = []
        for size in sizes:
            n = size // 2
            pattern = (torch.arange(n, device=dev) % 7).to(torch.bfloat16) / 8
            inp = pattern * (rank + 1)
            expect = pattern * (world * (world + 1) // 2)
            row = {"world": world, "gpus": gpus, "bytes": size, "tokens": size // (2 * HIDDEN)}
            dist.barrier()
            us, outs = time_graph(lambda x: nccl.all_reduce(x, stream=torch.cuda.current_stream()), inp, k, r)
            row["nccl_us"] = round(us, 2)
            row["nccl_ok"] = all(torch.equal(o, expect) for o in outs)
            del outs
            if ca is not None and not ca.disabled and ca.should_custom_ar(inp):
                dist.barrier()
                us, outs = time_graph(lambda x: ca.custom_all_reduce(x), inp, k, r, capture_ctx=ca.capture())
                row["custom_us"] = round(us, 2)
                row["custom_ok"] = all(torch.equal(o, expect) for o in outs)
                del outs
            else:
                row["custom_us"] = None
                row["custom_ok"] = None
            rows.append(row)
        meta = {"rank": rank, "can_access_peer": can_peer, "custom_disabled": True if ca is None else ca.disabled,
                "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE"), "nccl_proto": os.environ.get("NCCL_PROTO"),
                "force_pcie_custom_ar": force_fc}
        queue.put((rank, meta, rows))
        dist.barrier()
    except Exception:
        queue.put((rank, {"error": traceback.format_exc()}, []))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", required=True, help="visible CUDA ordinals, e.g. 0,1")
    ap.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)), help="bytes, multiples of 16")
    ap.add_argument("--per-graph", type=int, default=50)
    ap.add_argument("--replays", type=int, default=40)
    ap.add_argument("--force-pcie-custom-ar", action="store_true")
    ap.add_argument("--no-custom", action="store_true", help="measure NCCL only (P2P not verified)")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = free_port()
    procs = [ctx.Process(target=worker, args=(i, len(gpus), gpus, sizes, args.per_graph, args.replays, port,
                                               args.force_pcie_custom_ar, args.no_custom, q)) for i in range(len(gpus))]
    for p in procs:
        p.start()
    results = {}
    for _ in procs:
        rank, meta, rows = q.get(timeout=900)
        results[rank] = (meta, rows)
    for p in procs:
        p.join(timeout=60)
    errors = [m for m, _ in results.values() if "error" in m]
    if errors:
        print(json.dumps({"error": errors[0]["error"]}))
        sys.exit(1)
    meta0, rows0 = results[0]
    print(json.dumps({"meta": meta0}))
    for i, row in enumerate(rows0):
        # report the slowest rank (an all-reduce finishes when its slowest member does)
        for rk in results:
            other = results[rk][1][i]
            for key in ("nccl_us", "custom_us"):
                if other.get(key) is not None and row.get(key) is not None:
                    row[key] = max(row[key], other[key])
            for key in ("nccl_ok", "custom_ok"):
                if other.get(key) is False:
                    row[key] = False
        print(json.dumps(row))


if __name__ == "__main__":
    main()
