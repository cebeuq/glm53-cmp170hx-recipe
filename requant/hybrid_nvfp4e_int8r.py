#!/usr/bin/env python3
"""Build the hybrid GLM-5.3 checkpoint: exact NVFP4 routed experts + INT8-g128 residue.

usage: hybrid_nvfp4e_int8r.py NVFP4_DIR REQUANT_DIR OUT_DIR [--workers N] [--only SHARD]

REQUANT_DIR is requant_full.py output (INT4 experts + INT8 residue, source shard names kept).
For each shard: keep every requant tensor except the INT4 expert packs, and re-insert the
original ModelOpt NVFP4 expert tensors in compressed-tensors NVFP4A16 naming:
  .weight          (uint8 E2M1 pairs, low nibble = even k) -> .weight_packed   (bytes unchanged)
  .weight_scale    (fp8 e4m3 per 16)                       -> .weight_scale    (bytes unchanged)
  .weight_scale_2  (fp32, ModelOpt: w = q*s*scale_2)       -> .weight_global_scale = 1/scale_2
  .input_scale     (fp32)                                  -> .input_global_scale  = 1/input_scale
(vLLM CT NVFP4 MoE feeds 1/global_scale into the same Marlin FP4 path as ModelOpt scale_2:
 compressed_tensors_moe_w4a4_nvfp4.py:229.) Experts of layers 3-77 only; layer 78 (MTP) untouched.
"""
import argparse
import concurrent.futures as cf
import json
import re
import shutil
import sys
from pathlib import Path

EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(\w+)$")
NVFP4_GROUP = {
    "format": "nvfp4-pack-quantized",
    "input_activations": None,
    "output_activations": None,
    "targets": [r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
    "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 16,
                "num_bits": 4, "observer": "memoryless_minmax", "observer_kwargs": {},
                "scale_dtype": None, "strategy": "tensor_group", "symmetric": True,
                "type": "float", "zp_dtype": None},
}
RENAME = {"weight": "weight_packed", "weight_scale": "weight_scale",
          "weight_scale_2": "weight_global_scale", "input_scale": "input_global_scale"}


INT4_LAYERS = set()  # layers whose routed experts stay INT4-g128 (from the requant); others NVFP4


def is_moe_expert(name):
    m = EXPERT.match(name)
    return m if (m and 3 <= int(m.group(1)) <= 77 and int(m.group(1)) not in INT4_LAYERS) else None


def do_shard(src, rq, out, shard, int4_layers=()):
    import torch
    INT4_LAYERS.update(int4_layers)
    from safetensors import safe_open
    from safetensors.torch import save_file
    tensors, dropped, added = {}, 0, 0
    with safe_open(rq / shard, "pt") as f:
        for k in f.keys():
            if is_moe_expert(k):
                dropped += 1
                continue
            tensors[k] = f.get_tensor(k)
    src_path = src / shard
    if src_path.exists():
        with safe_open(src_path, "pt") as f:
            for k in f.keys():
                m = is_moe_expert(k)
                if not m:
                    continue
                kind = m.group(3)
                if kind not in RENAME:
                    raise RuntimeError(f"{shard}: unexpected expert tensor {k}")
                t = f.get_tensor(k)
                if kind in ("weight_scale_2", "input_scale"):
                    t = (1.0 / t.float()).to(torch.float32)
                tensors[k[: -len(kind)] + RENAME[kind]] = t.contiguous()
                added += 1
    save_file(tensors, str(out / shard), metadata={"format": "pt"})
    return shard, {k: v.numel() * v.element_size() for k, v in tensors.items()}, dropped, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("rq"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only")
    ap.add_argument("--int4-layers", default="", help="comma list of MoE layers to keep INT4 (e.g. 5,8,11)")
    a = ap.parse_args()
    int4 = sorted({int(x) for x in a.int4_layers.split(",") if x})
    INT4_LAYERS.update(int4)
    src, rq, out = Path(a.src), Path(a.rq), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rq_index = json.loads((rq / "model.safetensors.index.json").read_text())
    shards = sorted(set(rq_index["weight_map"].values()))
    if a.only:
        shards = [a.only]
    for p in rq.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
                "model.safetensors.index.json", "config.json", "REQUANT-REPORT.json"):
            shutil.copy2(p, out / p.name)
    weight_map, total, nd, na = {}, 0, 0, 0
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(do_shard, src, rq, out, s, int4) for s in shards if a.only or not (out / s).exists()]
        done_existing = [s for s in shards if not a.only and (out / s).exists()]
        for i, fu in enumerate(cf.as_completed(futs), 1):
            shard, sizes, d, ad = fu.result()
            nd += d; na += ad
            for k, b in sizes.items():
                weight_map[k] = shard; total += b
            print(f"[{i}/{len(futs)}] {shard}: dropped {d} int4, added {ad} nvfp4", flush=True)
    if done_existing:
        from safetensors import safe_open
        for s in done_existing:
            with safe_open(out / s, "pt") as f:
                for k in f.keys():
                    weight_map[k] = s
    if a.only:
        print(json.dumps({"only": a.only, "dropped": nd, "added": na}))
        return
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    cfg = json.loads((rq / "config.json").read_text())
    qc = cfg["quantization_config"]
    nv_layers = [L for L in range(3, 78) if L not in int4]
    g0 = json.loads(json.dumps(NVFP4_GROUP))
    g0["targets"] = [r"re:.*\.layers\.(?:" + "|".join(map(str, nv_layers)) + r")\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]
    qc["config_groups"]["group_0"] = g0
    if int4:
        g2 = json.loads(json.dumps(qc["config_groups"]["group_1"]))
        g2["weights"]["num_bits"] = 4
        g2["targets"] = [r"re:.*\.layers\.(?:" + "|".join(map(str, int4)) + r")\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"]
        qc["config_groups"]["group_2"] = g2
    qc["format"] = "mixed-precision"
    cfg["quantization_config"] = qc
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"shards": len(shards), "tensors": len(weight_map), "dropped_int4": nd, "added_nvfp4": na}))


if __name__ == "__main__":
    sys.exit(main())
