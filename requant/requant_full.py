#!/usr/bin/env python3
"""Stream GLM-5.3 ModelOpt NVFP4 weights into compressed-tensors packs.

The converter keeps the source shard names.  It converts routed experts to
INT4/group-128, converts the selected BF16 residue linears to INT8/group-128,
and copies every other tensor without changing its bytes or dtype.

NVFP4 nibble convention
-----------------------
vLLM's ModelOpt/Marlin FP4 path follows the natural packed-E2M1 convention:
the even input feature is the low nibble and the following odd input feature
is the high nibble.  In other words, for source byte ``b``::

    code[2*j + 0] = b & 0x0f
    code[2*j + 1] = b >> 4

This is also the order consumed when vLLM reinterprets the ModelOpt bytes as
INT32 words before its Marlin repack.  E2M1 bit 3 is the sign bit.  Bits 2:0
select ``[0, .5, 1, 1.5, 2, 3, 4, 6]``.

The output integer pack follows vLLM's compressed-tensors convention (checked
by ``--selftest``): the first input feature occupies the least-significant
bits of each INT32 word.  INT4 therefore stores eight
features per word and INT8 stores four.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import datetime as dt
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable


GROUP_SIZE = 128
NVFP4_GROUP_SIZE = 16
CHUNK_VALUES = 1 << 20
CONFIG_VERSION = "0.18.1.a20260827"
REQUANT_FORMAT_VERSION = "1"

EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
EXPERT_AUX_RE = re.compile(
    r"^(?P<module>model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj))\."
    r"(?P<kind>weight_scale|weight_scale_2|input_scale)$"
)
ATTENTION_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.self_attn\."
    r"(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj)\.weight$"
)
SHARED_EXPERT_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.shared_experts\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
DENSE_MLP_WEIGHT_RE = re.compile(
    r"^model\.layers\.(?P<layer>[0-2])\.mlp\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
ANY_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")

EXPERT_CONFIG_TARGET = (
    r"re:.*mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"
)
LAYER_0_77_CONFIG_RE = r"(?:[0-9]|[1-6][0-9]|7[0-7])"
RESIDUE_CONFIG_TARGETS = [
    rf"re:.*\.layers\.{LAYER_0_77_CONFIG_RE}\.self_attn\."
    r"(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_b_proj|o_proj|"
    r"fused_qkv_a_proj)$",
    rf"re:.*\.layers\.{LAYER_0_77_CONFIG_RE}\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj|gate_up_proj)$",
    r"re:.*\.layers\.[0-2]\.mlp\."
    r"(gate_proj|up_proj|down_proj|gate_up_proj)$",
]

_torch: Any | None = None
_safe_open: Any | None = None
_save_file: Any | None = None


def log(message: str) -> None:
    """Print one timestamped progress line."""

    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def _load_torch() -> Any:
    global _torch
    if _torch is None:
        import torch

        _torch = torch
    return _torch


def _load_safetensors() -> tuple[Any, Any]:
    global _safe_open, _save_file
    if _safe_open is None or _save_file is None:
        from safetensors import safe_open
        from safetensors.torch import save_file

        _safe_open = safe_open
        _save_file = save_file
    return _safe_open, _save_file


def _worker_init() -> None:
    """Keep each process single-threaded; the process pool owns parallelism."""

    torch = _load_torch()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this value only once in a process.
        pass
    _load_safetensors()


def e2m1_lut(*, device: Any = None) -> Any:
    """Return the signed E2M1 decode table indexed by a four-bit code."""

    torch = _load_torch()
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def unpack_e2m1(weight_u8: Any) -> Any:
    """Decode natural-order packed E2M1 bytes to an FP32 tensor."""

    torch = _load_torch()
    if weight_u8.dtype != torch.uint8 or weight_u8.ndim != 2:
        raise ValueError(
            f"NVFP4 weight must be 2-D uint8, got {weight_u8.dtype} "
            f"shape={tuple(weight_u8.shape)}"
        )
    rows, packed_in = weight_u8.shape
    codes = torch.empty((rows, packed_in * 2), dtype=torch.uint8)
    # vLLM/ModelOpt natural order: even K is low, odd K is high.
    codes[:, 0::2] = weight_u8 & 0x0F
    codes[:, 1::2] = weight_u8 >> 4
    return e2m1_lut()[codes.to(torch.int64)]


def dequantize_nvfp4(weight_u8: Any, weight_scale: Any, weight_scale_2: Any) -> Any:
    """Dequantize one complete (normally synthetic/small) NVFP4 tensor."""

    torch = _load_torch()
    values = unpack_e2m1(weight_u8)
    rows, in_features = values.shape
    expected_scale_shape = (rows, in_features // NVFP4_GROUP_SIZE)
    if in_features % NVFP4_GROUP_SIZE or tuple(weight_scale.shape) != expected_scale_shape:
        raise ValueError(
            f"NVFP4 scale shape mismatch: weight values {(rows, in_features)}, "
            f"scale {tuple(weight_scale.shape)}, expected {expected_scale_shape}"
        )
    scales = weight_scale.float().repeat_interleave(NVFP4_GROUP_SIZE, dim=1)
    global_scale = torch.as_tensor(weight_scale_2, dtype=torch.float32).reshape(())
    return values * scales * global_scale


def quantize(weight: Any, bits: int) -> tuple[Any, Any, Any]:
    """RTN symmetric group-128 quantization with compressed-tensors packing.

    The arithmetic and packing sequence match what vLLM's compressed-tensors
    loader expects (``--selftest`` pins it).  Do not replace it with a different
    signed representation or a most-significant-bit-first pack.
    """

    torch = _load_torch()
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % GROUP_SIZE:
        raise ValueError(
            f"in_features={in_features} is not divisible by group size {GROUP_SIZE}"
        )

    wf = weight.float().view(out_features, in_features // GROUP_SIZE, GROUP_SIZE)
    qmax = (1 << (bits - 1)) - 1
    scale = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = (
        torch.round(wf / scale)
        .clamp(-qmax - 1, qmax)
        .to(torch.int8)
        .view(out_features, in_features)
    )
    pack_factor = 32 // bits
    unsigned = (
        q.to(torch.int16) + (1 << (bits - 1))
    ).to(torch.int32).view(out_features, in_features // pack_factor, pack_factor)
    shifts = torch.arange(pack_factor, dtype=torch.int32) * bits
    packed = (unsigned << shifts).sum(dim=-1, dtype=torch.int64).to(torch.int32)
    return (
        packed.contiguous(),
        scale.view(out_features, in_features // GROUP_SIZE)
        .to(torch.bfloat16)
        .contiguous(),
        torch.tensor([out_features, in_features], dtype=torch.int64),
    )


def dequantize_packed(packed: Any, scale: Any, shape: Any, bits: int) -> Any:
    """Reference unpacker used by the self-test."""

    torch = _load_torch()
    out_features, in_features = (int(v) for v in shape.tolist())
    pack_factor = 32 // bits
    shifts = torch.arange(pack_factor, dtype=torch.int32) * bits
    mask = (1 << bits) - 1
    codes = ((packed.unsqueeze(-1) >> shifts) & mask).view(
        out_features, in_features
    )
    signed = codes.to(torch.int16) - (1 << (bits - 1))
    expanded_scale = scale.float().repeat_interleave(GROUP_SIZE, dim=1)
    return signed.float() * expanded_scale


def _row_chunk_size(in_features: int) -> int:
    return max(1, CHUNK_VALUES // max(1, in_features))


def _tensor_to_ipc_arrays(packed: Any, scale: Any) -> tuple[Any, Any]:
    """Return NumPy arrays so standard multiprocessing avoids torch /dev/shm."""

    # Torch's multiprocessing reducer moves tensors through shared memory.  A
    # default Docker /dev/shm can be only 64 MiB, far below one source shard.
    # NumPy arrays use the ProcessPool pipe and keep the converter independent
    # of that limit.  BF16 has no NumPy dtype, so transport its uint16 bits.
    return packed.numpy(), scale.view(_load_torch().uint16).numpy()


def _worker_quantize_residue(task: tuple[str, str, int]) -> tuple[Any, ...]:
    shard_path, name, bits = task
    torch = _load_torch()
    safe_open, _ = _load_safetensors()

    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        source = handle.get_slice(name)
        out_features, in_features = source.get_shape()
        pack_factor = 32 // bits
        packed = torch.empty(
            (out_features, in_features // pack_factor), dtype=torch.int32
        )
        scales = torch.empty(
            (out_features, in_features // GROUP_SIZE), dtype=torch.bfloat16
        )
        row_chunk = _row_chunk_size(in_features)
        for start in range(0, out_features, row_chunk):
            end = min(out_features, start + row_chunk)
            chunk = source[start:end, :]
            chunk_packed, chunk_scale, _ = quantize(chunk, bits)
            packed[start:end].copy_(chunk_packed)
            scales[start:end].copy_(chunk_scale)

    packed_array, scale_bits_array = _tensor_to_ipc_arrays(packed, scales)
    return name, bits, out_features, in_features, packed_array, scale_bits_array


def _worker_quantize_expert(
    task: tuple[str, str, str, str, str, str]
) -> tuple[Any, ...]:
    weight_path, scale_path, scale_2_path, weight_name, scale_name, scale_2_name = task
    torch = _load_torch()
    safe_open, _ = _load_safetensors()

    paths = {weight_path, scale_path, scale_2_path}
    with contextlib.ExitStack() as stack:
        handles = {
            path: stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
            for path in paths
        }
        weight_slice = handles[weight_path].get_slice(weight_name)
        scale_slice = handles[scale_path].get_slice(scale_name)
        global_scale_tensor = handles[scale_2_path].get_tensor(scale_2_name)
        global_scale = global_scale_tensor.float().reshape(()).item()

        out_features, packed_in = weight_slice.get_shape()
        in_features = packed_in * 2
        expected_scale_shape = (out_features, in_features // NVFP4_GROUP_SIZE)
        if tuple(scale_slice.get_shape()) != expected_scale_shape:
            raise ValueError(
                f"{weight_name}: scale shape {tuple(scale_slice.get_shape())}, "
                f"expected {expected_scale_shape}"
            )
        if in_features % GROUP_SIZE:
            raise ValueError(
                f"{weight_name}: in_features={in_features} is not divisible by "
                f"{GROUP_SIZE}"
            )

        packed = torch.empty(
            (out_features, in_features // 8), dtype=torch.int32
        )
        output_scales = torch.empty(
            (out_features, in_features // GROUP_SIZE), dtype=torch.bfloat16
        )
        row_chunk = _row_chunk_size(in_features)
        for start in range(0, out_features, row_chunk):
            end = min(out_features, start + row_chunk)
            source_bytes = weight_slice[start:end, :]
            source_scales = scale_slice[start:end, :]
            dequantized = dequantize_nvfp4(
                source_bytes, source_scales, global_scale
            )
            chunk_packed, chunk_scale, _ = quantize(dequantized, 4)
            packed[start:end].copy_(chunk_packed)
            output_scales[start:end].copy_(chunk_scale)

    packed_array, scale_bits_array = _tensor_to_ipc_arrays(packed, output_scales)
    return (
        weight_name,
        4,
        out_features,
        in_features,
        packed_array,
        scale_bits_array,
    )


def _numpy_result_to_tensors(result: tuple[Any, ...]) -> tuple[str, int, dict[str, Any]]:
    torch = _load_torch()
    name, bits, out_features, in_features, packed_array, scale_bits_array = result
    module = name[: -len(".weight")]
    tensors = {
        module + ".weight_packed": torch.from_numpy(packed_array),
        module + ".weight_scale": torch.from_numpy(scale_bits_array).view(
            torch.bfloat16
        ),
        module + ".weight_shape": torch.tensor(
            [out_features, in_features], dtype=torch.int64
        ),
    }
    return module, bits, tensors


def _dtype_code(dtype: Any) -> str:
    value = str(dtype).upper().replace("TORCH.", "")
    aliases = {
        "UINT8": "U8",
        "INT8": "I8",
        "INT16": "I16",
        "INT32": "I32",
        "INT64": "I64",
        "FLOAT16": "F16",
        "HALF": "F16",
        "FLOAT32": "F32",
        "FLOAT64": "F64",
        "BFLOAT16": "BF16",
        "FLOAT8_E4M3FN": "F8_E4M3",
        "FLOAT8_E5M2": "F8_E5M2",
    }
    return aliases.get(value, value)


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F8_E8M0FNU": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _shape_numel(shape: Iterable[int]) -> int:
    return math.prod(int(v) for v in shape)


def _slice_nbytes(tensor_slice: Any) -> int:
    code = _dtype_code(tensor_slice.get_dtype())
    try:
        item_size = DTYPE_BYTES[code]
    except KeyError as exc:
        raise ValueError(f"unsupported safetensors dtype {code!r}") from exc
    return _shape_numel(tensor_slice.get_shape()) * item_size


def _tensor_nbytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()


def _empty_stats() -> dict[str, Any]:
    return {
        "tensor_counts": {
            "expert_int4": 0,
            "residue_int8": 0,
            "copied": 0,
            "dropped": 0,
        },
        "tensor_bytes": {
            "expert_int4": 0,
            "residue_int8": 0,
            "copied": 0,
        },
        "bf16_residue_modules": [],
    }


def _add_stats(target: dict[str, Any], source: dict[str, Any]) -> None:
    for section in ("tensor_counts", "tensor_bytes"):
        for key, value in source[section].items():
            target[section][key] += int(value)
    target["bf16_residue_modules"] = sorted(
        set(target["bf16_residue_modules"])
        | set(source.get("bf16_residue_modules", []))
    )


def _module_is_expert(module: str) -> bool:
    return EXPERT_WEIGHT_RE.fullmatch(module + ".weight") is not None


def _inspect_output_shard(path: Path) -> tuple[dict[str, int], dict[str, Any]]:
    """Read only a safetensors header and recover its map, bytes, and stats."""

    safe_open, _ = _load_safetensors()
    weight_bytes: dict[str, int] = {}
    stats = _empty_stats()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("requant_full_version") != REQUANT_FORMAT_VERSION:
            raise ValueError(
                f"{path}: file exists but is not a completed requant_full.py "
                "output shard; move it aside or remove it before resuming"
            )
        if metadata.get("source_shard") != path.name:
            raise ValueError(
                f"{path}: requant metadata names source shard "
                f"{metadata.get('source_shard')!r}, not {path.name!r}"
            )
        keys = list(handle.keys())
        key_set = set(keys)
        quant_modules = {
            key[: -len(".weight_packed")]
            for key in keys
            if key.endswith(".weight_packed")
        }
        for module in quant_modules:
            required = {
                module + ".weight_packed",
                module + ".weight_scale",
                module + ".weight_shape",
            }
            if not required.issubset(key_set):
                missing = sorted(required - key_set)
                raise ValueError(f"{path}: incomplete quantized module {module}: {missing}")

        quant_key_tiers: dict[str, str] = {}
        for module in quant_modules:
            tier = "expert_int4" if _module_is_expert(module) else "residue_int8"
            for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
                quant_key_tiers[module + suffix] = tier

        for key in keys:
            nbytes = _slice_nbytes(handle.get_slice(key))
            weight_bytes[key] = nbytes
            tier = quant_key_tiers.get(key)
            if tier is None:
                stats["tensor_bytes"]["copied"] += nbytes
            else:
                stats["tensor_bytes"][tier] += nbytes

        for module in quant_modules:
            tier = "expert_int4" if _module_is_expert(module) else "residue_int8"
            stats["tensor_counts"][tier] += 1
        stats["tensor_counts"]["copied"] = len(keys) - 3 * len(quant_modules)
        # Each converted expert replaces weight/weight_scale and removes the
        # source global and activation scales.  Count all three source aux
        # tensors as dropped for a consistent source-action report.
        stats["tensor_counts"]["dropped"] = (
            3 * stats["tensor_counts"]["expert_int4"]
        )
        stats["bf16_residue_modules"] = sorted(
            key[: -len(".weight")]
            for key in keys
            if key.endswith(".weight") and _residue_match(key) is not None
        )
    return weight_bytes, stats


def _residue_match(name: str) -> re.Match[str] | None:
    for pattern in (
        ATTENTION_WEIGHT_RE,
        SHARED_EXPERT_WEIGHT_RE,
        DENSE_MLP_WEIGHT_RE,
    ):
        match = pattern.fullmatch(name)
        if match is not None and int(match.group("layer")) <= 77:
            return match
    return None


def _expert_is_convertible(name: str) -> bool:
    match = EXPERT_WEIGHT_RE.fullmatch(name)
    return match is not None and 3 <= int(match.group("layer")) <= 77


def _expert_aux_is_dropped(name: str) -> bool:
    match = EXPERT_AUX_RE.fullmatch(name)
    return match is not None and 3 <= int(match.group("layer")) <= 77


def _source_shard_selection(
    weight_map: dict[str, str], layer_range: tuple[int, int] | None
) -> set[str]:
    all_shards = set(weight_map.values())
    if layer_range is None:
        return all_shards
    first, last = layer_range
    selected: set[str] = set()
    for name, shard in weight_map.items():
        match = ANY_LAYER_RE.match(name)
        if match is not None and first <= int(match.group("layer")) <= last:
            selected.add(shard)
    return selected


def _expert_task(
    in_dir: Path, weight_map: dict[str, str], weight_name: str
) -> tuple[str, str, str, str, str, str]:
    module = weight_name[: -len(".weight")]
    scale_name = module + ".weight_scale"
    scale_2_name = module + ".weight_scale_2"
    input_scale_name = module + ".input_scale"
    missing = [
        name
        for name in (scale_name, scale_2_name, input_scale_name)
        if name not in weight_map
    ]
    if missing:
        raise ValueError(f"{weight_name}: missing ModelOpt tensors {missing}")
    return (
        str(in_dir / weight_map[weight_name]),
        str(in_dir / weight_map[scale_name]),
        str(in_dir / weight_map[scale_2_name]),
        weight_name,
        scale_name,
        scale_2_name,
    )


def _process_shard(
    in_dir: Path,
    out_dir: Path,
    shard: str,
    weight_map: dict[str, str],
    executor: concurrent.futures.ProcessPoolExecutor,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Convert one source shard and atomically install the output shard."""

    safe_open, save_file = _load_safetensors()
    source_path = in_dir / shard
    output_path = out_dir / shard
    if not source_path.is_file():
        raise FileNotFoundError(f"source shard does not exist: {source_path}")

    copy_names: list[str] = []
    expert_tasks: list[tuple[str, str, str, str, str, str]] = []
    residue_tasks: list[tuple[str, str, int]] = []
    stats = _empty_stats()

    with safe_open(str(source_path), framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        for name in source_keys:
            if _expert_is_convertible(name):
                source_slice = handle.get_slice(name)
                if _dtype_code(source_slice.get_dtype()) != "U8":
                    raise ValueError(
                        f"{name}: expected uint8 NVFP4 weight, got "
                        f"{source_slice.get_dtype()}"
                    )
                expert_tasks.append(_expert_task(in_dir, weight_map, name))
                stats["tensor_counts"]["expert_int4"] += 1
                continue
            if _expert_aux_is_dropped(name):
                stats["tensor_counts"]["dropped"] += 1
                continue

            residue = _residue_match(name)
            if residue is not None:
                source_slice = handle.get_slice(name)
                shape = tuple(source_slice.get_shape())
                dtype = _dtype_code(source_slice.get_dtype())
                if len(shape) == 2 and dtype == "BF16" and shape[1] % GROUP_SIZE == 0:
                    residue_tasks.append((str(source_path), name, 8))
                    stats["tensor_counts"]["residue_int8"] += 1
                    continue
                reason = (
                    f"in_features={shape[1]} is not divisible by {GROUP_SIZE}"
                    if len(shape) == 2 and shape[1] % GROUP_SIZE
                    else f"expected 2-D BF16, got dtype={dtype} shape={shape}"
                )
                log(f"{shard}: keep BF16 {name}: {reason}")
                stats["bf16_residue_modules"].append(name[: -len(".weight")])

            copy_names.append(name)
            stats["tensor_counts"]["copied"] += 1

    futures: dict[concurrent.futures.Future[Any], str] = {}
    for task in expert_tasks:
        future = executor.submit(_worker_quantize_expert, task)
        futures[future] = task[3]
    for task in residue_tasks:
        future = executor.submit(_worker_quantize_residue, task)
        futures[future] = task[1]

    output_tensors: dict[str, Any] = {}
    try:
        for future in concurrent.futures.as_completed(futures):
            source_name = futures[future]
            try:
                module, bits, converted = _numpy_result_to_tensors(future.result())
            except Exception as exc:
                raise RuntimeError(f"failed to convert {source_name}: {exc}") from exc
            tier = "expert_int4" if bits == 4 else "residue_int8"
            for output_name, tensor in converted.items():
                if output_name in output_tensors:
                    raise ValueError(f"duplicate output tensor {output_name}")
                output_tensors[output_name] = tensor
                stats["tensor_bytes"][tier] += _tensor_nbytes(tensor)

        # Copy only after conversions complete.  This keeps worker transients
        # separate from the source shard's full set of passthrough tensors.
        with safe_open(str(source_path), framework="pt", device="cpu") as handle:
            for name in copy_names:
                tensor = handle.get_tensor(name)
                output_tensors[name] = tensor
                stats["tensor_bytes"]["copied"] += _tensor_nbytes(tensor)

        metadata = {
            "format": "pt",
            "requant_full_version": REQUANT_FORMAT_VERSION,
            "source_shard": shard,
            "requant_stats": json.dumps(stats, sort_keys=True, separators=(",", ":")),
        }
        temp_path = output_path.with_name(
            f".{output_path.name}.requant-tmp-{os.getpid()}"
        )
        try:
            if temp_path.exists():
                temp_path.unlink()
            save_file(output_tensors, str(temp_path), metadata=metadata)
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    finally:
        for future in futures:
            if not future.done():
                future.cancel()

    weight_bytes = {name: _tensor_nbytes(tensor) for name, tensor in output_tensors.items()}
    return weight_bytes, stats


def _quant_group(targets: list[str], bits: int) -> dict[str, Any]:
    return {
        "format": "pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": targets,
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": GROUP_SIZE,
            "num_bits": bits,
            "observer": "memoryless_minmax",
            "observer_kwargs": {},
            "scale_dtype": None,
            "strategy": "group",
            "symmetric": True,
            "type": "int",
            "zp_dtype": None,
        },
    }


def compressed_tensors_config(
    extra_ignore: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the replacement quantization_config block."""

    return {
        "config_groups": {
            "group_0": _quant_group([EXPERT_CONFIG_TARGET], 4),
            "group_1": _quant_group(RESIDUE_CONFIG_TARGETS, 8),
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": [
            "lm_head",
            "model.embed_tokens",
            r"re:.*mlp\.gate$",
            r"re:.*indexer.*",
            r"re:model\.layers\.78\..*",
            *sorted(set(extra_ignore)),
        ],
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": CONFIG_VERSION,
    }


def _atomic_write_json(path: Path, data: Any) -> None:
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_config(
    in_dir: Path, out_dir: Path, extra_ignore: Iterable[str] = ()
) -> None:
    with (in_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["quantization_config"] = compressed_tensors_config(extra_ignore)
    _atomic_write_json(out_dir / "config.json", config)


def _copy_model_sidecars(in_dir: Path, out_dir: Path) -> None:
    """Copy small tokenizer/custom-code files needed beside the new weights."""

    allowed_suffixes = {".json", ".jinja", ".txt", ".py", ".model", ".tiktoken"}
    excluded = {"config.json", "model.safetensors.index.json", "REQUANT-REPORT.json"}
    for source in in_dir.iterdir():
        if (
            source.is_file()
            and source.name not in excluded
            and source.suffix.lower() in allowed_suffixes
        ):
            shutil.copy2(source, out_dir / source.name)


def _build_index(
    source_index: dict[str, Any],
    shard_maps: dict[str, dict[str, int]],
) -> dict[str, Any]:
    output_index = copy.deepcopy(source_index)
    output_weight_map: dict[str, str] = {}
    total_size = 0
    for shard in sorted(shard_maps):
        for name, nbytes in sorted(shard_maps[shard].items()):
            previous = output_weight_map.get(name)
            if previous is not None:
                raise ValueError(
                    f"output tensor {name} occurs in both {previous} and {shard}"
                )
            output_weight_map[name] = shard
            total_size += int(nbytes)
    metadata = dict(output_index.get("metadata") or {})
    metadata["total_size"] = total_size
    output_index["metadata"] = metadata
    output_index["weight_map"] = output_weight_map
    return output_index


def _report(
    *,
    started: float,
    workers: int,
    source_shards: list[str],
    selected_shards: set[str],
    shard_maps: dict[str, dict[str, int]],
    shard_stats: dict[str, dict[str, Any]],
    skipped_existing: int,
    layer_range: tuple[int, int] | None,
) -> dict[str, Any]:
    totals = _empty_stats()
    for stats in shard_stats.values():
        _add_stats(totals, stats)
    total_tensor_bytes = sum(
        nbytes for weight_bytes in shard_maps.values() for nbytes in weight_bytes.values()
    )
    # The real file-byte total is filled by the caller.  Keep this helper free
    # of output-directory state so it remains straightforward to unit test.
    return {
        "status": "complete" if len(shard_maps) == len(source_shards) else "partial",
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "workers": workers,
        "layers": None if layer_range is None else f"{layer_range[0]}-{layer_range[1]}",
        "shards": {
            "source": len(source_shards),
            "selected_this_run": len(selected_shards),
            "written_or_present": len(shard_maps),
            "skipped_existing": skipped_existing,
        },
        "tensor_counts_by_tier": totals["tensor_counts"],
        "tensor_bytes_by_tier": totals["tensor_bytes"],
        "bf16_residue_modules": totals["bf16_residue_modules"],
        "total_bytes": total_tensor_bytes,
        "total_tensor_bytes": total_tensor_bytes,
        "total_file_bytes": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def _bf16_residue_ignores(
    shard_stats: dict[str, dict[str, Any]],
) -> list[str]:
    return sorted(
        {
            module
            for stats in shard_stats.values()
            for module in stats.get("bf16_residue_modules", [])
        }
    )


def _parse_layers(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)-(\d+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("layers must use inclusive form a-b")
    first, last = (int(match.group(1)), int(match.group(2)))
    if first > last or first < 0 or last > 78:
        raise argparse.ArgumentTypeError("layers must satisfy 0 <= a <= b <= 78")
    return first, last


def run_selftest() -> int:
    try:
        torch = _load_torch()
    except ImportError as exc:
        print(
            "SELFTEST SKIPPED: PyTorch is not installed in this environment; "
            "run this command inside the vLLM image.",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 0

    torch.manual_seed(20260901)

    # Hand-computed code table plus an order-sensitive packed byte check.
    expected = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
    )
    codes = torch.arange(16, dtype=torch.uint8).view(1, 16)
    packed_codes = codes[:, 0::2] | (codes[:, 1::2] << 4)
    decoded = unpack_e2m1(packed_codes).view(-1)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    order_probe = unpack_e2m1(torch.tensor([[0x21]], dtype=torch.uint8))
    torch.testing.assert_close(
        order_probe, torch.tensor([[0.5, 1.0]]), rtol=0, atol=0
    )

    # Pin the proven compressed-tensors word order with values whose RTN
    # scales are exactly 1.0.  The first K value must occupy the low bits.
    known4 = torch.zeros((1, GROUP_SIZE), dtype=torch.float32)
    known4[0, :15] = torch.arange(-7, 8, dtype=torch.float32)
    known4_packed, known4_scale, _ = quantize(known4, 4)
    assert int(known4_packed[0, 0]) == -2023406815  # uint32 0x87654321
    torch.testing.assert_close(
        known4_scale, torch.ones_like(known4_scale), rtol=0, atol=0
    )

    known8 = torch.zeros((1, GROUP_SIZE), dtype=torch.float32)
    known8[0, :4] = torch.arange(-7, -3, dtype=torch.float32)
    known8[0, -1] = 127.0
    known8_packed, known8_scale, _ = quantize(known8, 8)
    assert int(known8_packed[0, 0]) == 2088467065  # uint32 0x7c7b7a79
    torch.testing.assert_close(
        known8_scale, torch.ones_like(known8_scale), rtol=0, atol=0
    )

    out_features, in_features = 9, 256
    random_codes = torch.randint(0, 16, (out_features, in_features), dtype=torch.uint8)
    fake_weight = random_codes[:, 0::2] | (random_codes[:, 1::2] << 4)
    fake_scale = (
        0.01
        + torch.rand(out_features, in_features // NVFP4_GROUP_SIZE) * 0.25
    )
    if hasattr(torch, "float8_e4m3fn"):
        try:
            fake_scale = fake_scale.to(torch.float8_e4m3fn)
        except (RuntimeError, TypeError):
            # Some CPU-only PyTorch builds expose the dtype but do not
            # implement CPU conversion to it.  FP32 still tests the exact
            # scale placement and the full dequantize/requantize path.
            pass
    fake_global_scale = torch.tensor(1.75, dtype=torch.float32)
    dequantized = dequantize_nvfp4(fake_weight, fake_scale, fake_global_scale)
    packed, scales, shape = quantize(dequantized, 4)
    roundtrip = dequantize_packed(packed, scales, shape, 4)

    assert packed.dtype == torch.int32
    assert packed.shape == (out_features, in_features // 8)
    assert scales.dtype == torch.bfloat16
    assert scales.shape == (out_features, in_features // GROUP_SIZE)
    assert shape.dtype == torch.int64
    assert shape.tolist() == [out_features, in_features]

    error = (roundtrip - dequantized).abs()
    group_amax = dequantized.abs().view(
        out_features, in_features // GROUP_SIZE, GROUP_SIZE
    ).amax(dim=-1)
    # Ideal RTN error is at most half an FP32 quantization step.  The saved
    # scale is BF16, so add the worst reconstruction displacement caused by
    # its measured rounding error instead of relying on an arbitrary epsilon.
    ideal_scale = group_amax / 7.0
    stored_scale = scales.float()
    allowed_per_group = (
        ideal_scale / 2.0 + 7.0 * (stored_scale - ideal_scale).abs() + 1e-6
    )
    allowed = allowed_per_group.repeat_interleave(GROUP_SIZE, dim=1)
    if not bool((error <= allowed).all()):
        raise AssertionError(
            f"INT4 round-trip error exceeded bound: max error={error.max().item():.6g}, "
            f"max bound={allowed.max().item():.6g}"
        )

    # Exercise INT8's four-values-per-word convention too.
    packed8, scales8, shape8 = quantize(dequantized.to(torch.bfloat16), 8)
    assert packed8.dtype == torch.int32
    assert packed8.shape == (out_features, in_features // 4)
    assert scales8.dtype == torch.bfloat16
    assert shape8.tolist() == [out_features, in_features]

    print(
        "SELFTEST PASS: low-nibble-first E2M1 decode, INT4/INT8 natural-order "
        "packing, dtypes, shapes, and RTN error bound"
    )
    return 0


def convert(args: argparse.Namespace) -> int:
    try:
        _load_torch()
        _load_safetensors()
    except ImportError as exc:
        raise SystemExit(
            f"conversion requires torch and safetensors (run inside the vLLM image): {exc}"
        ) from exc

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if in_dir == out_dir:
        raise SystemExit("input and output directories must be different")
    if not in_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {in_dir}")
    for required in ("model.safetensors.index.json", "config.json"):
        if not (in_dir / required).is_file():
            raise SystemExit(f"missing input file: {in_dir / required}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with (in_dir / "model.safetensors.index.json").open(
        "r", encoding="utf-8"
    ) as handle:
        source_index = json.load(handle)
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit("input index has no non-empty weight_map")

    source_shards = sorted(set(weight_map.values()))
    selected_shards = _source_shard_selection(weight_map, args.layers)
    if args.layers is not None and not selected_shards:
        raise SystemExit(f"no source shards intersect layers {args.layers[0]}-{args.layers[1]}")

    _copy_model_sidecars(in_dir, out_dir)
    _write_config(in_dir, out_dir)

    started = time.monotonic()
    shard_maps: dict[str, dict[str, int]] = {}
    shard_stats: dict[str, dict[str, Any]] = {}
    skipped_existing = 0

    # Inspect every existing source-named output shard.  This reconstructs the
    # final index after a resumed or layer-range run without trusting stale JSON.
    for shard in source_shards:
        output_path = out_dir / shard
        if output_path.is_file():
            weight_bytes, stats = _inspect_output_shard(output_path)
            shard_maps[shard] = weight_bytes
            shard_stats[shard] = stats
            if shard in selected_shards:
                skipped_existing += 1
                log(f"{shard}: skip existing output shard")
    _write_config(in_dir, out_dir, _bf16_residue_ignores(shard_stats))

    missing_selected = [
        shard for shard in source_shards if shard in selected_shards and shard not in shard_maps
    ]
    log(
        f"start: {len(source_shards)} source shards, {len(selected_shards)} selected, "
        f"{len(missing_selected)} to write, workers={args.workers}"
    )

    try:
        context = mp.get_context("fork")
    except ValueError:
        context = mp.get_context("spawn")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_init,
    ) as executor:
        for ordinal, shard in enumerate(missing_selected, start=1):
            shard_started = time.monotonic()
            log(f"{shard}: begin ({ordinal}/{len(missing_selected)})")
            weight_bytes, stats = _process_shard(
                in_dir, out_dir, shard, weight_map, executor
            )
            shard_maps[shard] = weight_bytes
            shard_stats[shard] = stats
            shard_size = sum(weight_bytes.values())
            counts = stats["tensor_counts"]
            log(
                f"{shard}: done in {time.monotonic() - shard_started:.1f}s; "
                f"{shard_size / (1024**3):.2f} GiB; "
                f"INT4={counts['expert_int4']} INT8={counts['residue_int8']} "
                f"copied={counts['copied']} dropped={counts['dropped']}"
            )

            output_index = _build_index(source_index, shard_maps)
            _atomic_write_json(out_dir / "model.safetensors.index.json", output_index)
            _write_config(in_dir, out_dir, _bf16_residue_ignores(shard_stats))
            report = _report(
                started=started,
                workers=args.workers,
                source_shards=source_shards,
                selected_shards=selected_shards,
                shard_maps=shard_maps,
                shard_stats=shard_stats,
                skipped_existing=skipped_existing,
                layer_range=args.layers,
            )
            report["total_file_bytes"] = sum(
                (out_dir / name).stat().st_size for name in shard_maps
            )
            _atomic_write_json(out_dir / "REQUANT-REPORT.json", report)

    output_index = _build_index(source_index, shard_maps)
    _atomic_write_json(out_dir / "model.safetensors.index.json", output_index)
    _write_config(in_dir, out_dir, _bf16_residue_ignores(shard_stats))
    report = _report(
        started=started,
        workers=args.workers,
        source_shards=source_shards,
        selected_shards=selected_shards,
        shard_maps=shard_maps,
        shard_stats=shard_stats,
        skipped_existing=skipped_existing,
        layer_range=args.layers,
    )
    report["total_file_bytes"] = sum(
        (out_dir / name).stat().st_size for name in shard_maps
    )
    _atomic_write_json(out_dir / "REQUANT-REPORT.json", report)

    log(
        f"finished: status={report['status']}, shards={len(shard_maps)}/"
        f"{len(source_shards)}, tensors={len(output_index['weight_map'])}, "
        f"{report['total_tensor_bytes'] / (1024**3):.2f} GiB, "
        f"elapsed={report['elapsed_seconds']:.1f}s"
    )
    if report["status"] == "partial":
        log(
            "partial layer-range run: repeat with other --layers ranges or run "
            "without --layers; existing complete shards will be skipped"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert GLM-5.3 ModelOpt NVFP4 experts to compressed-tensors "
            "INT4 and its BF16 residue linears to INT8."
        )
    )
    parser.add_argument("in_dir", nargs="?", help="source Hugging Face model directory")
    parser.add_argument("out_dir", nargs="?", help="output model directory")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(48, os.cpu_count() or 1),
        help="tensor worker processes (default: min(48, CPU count))",
    )
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        metavar="A-B",
        help=(
            "process complete shards that intersect this inclusive layer range; "
            "useful for staged/resumed conversion"
        ),
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run synthetic E2M1/packing tests and do not read a checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.selftest:
        return run_selftest()
    if args.in_dir is None or args.out_dir is None:
        parser.error("in_dir and out_dir are required unless --selftest is used")
    return convert(args)


if __name__ == "__main__":
    raise SystemExit(main())
