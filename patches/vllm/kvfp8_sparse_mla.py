# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""recipe patch (2026-10-10): FP8 KV cache read path for TRITON_MLA_SPARSE on SM80.

New file, mounted at vllm/v1/attention/ops/kvfp8_sparse_mla.py and called from
XPUMLASparseImpl.forward_mqa (patches/ported/xpu_mla_sparse.py) when the KV cache
dtype is quantized. The write side already exists in models/deepseek_v32/common/
kernels.py (fused_norm_rope, patched for SM80 in patches/ported/dsv32_kernels.py).

Cache formats (one row per token, rows contiguous inside a block):
  FMT_FP8   (--kv-cache-dtype fp8 | fp8_e4m3): 576 B = e4m3 latent[512] + e4m3 rope[64],
            one per-tensor scale (layer._k_scale, 1.0 unless the checkpoint has one).
  FMT_DSMLA (--kv-cache-dtype fp8_ds_mla): 656 B = e4m3 latent[512] + fp32 scale[4]
            (one per 128-dim tile, amax/448, computed per token at write time)
            + bf16 rope[64]. Same layout as FlashMLA sparse on Hopper.

SM80 has no fp8 convert, so bytes are expanded to exact bf16 in-kernel: either integer
bit moves (prmt/shr/lop3) plus an exact x2^120 bf16 FMA in inline PTX (4 values per asm),
or a 256-entry LUT. Scales are applied outside the tensor-core dots, on the [H, N]
score / probability tiles (per-token, per-128-tile), so the only per-element work is the
byte expansion. The latent is loaded once per tile and used for both QK and PV
(the bf16 kernel loads it twice). Everything is graph-safe: no host sync, the LUT and
scale tensors are persistent device tensors.
"""

import functools
import os

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut
from vllm.v1.attention.ops.triton_mla_sparse_kernel import _sparse_mla_merge_kernel

FMT_FP8 = 1
FMT_DSMLA = 2
FMT_DSMLA_I8 = 3  # fp8_ds_mla layout with int8 latent (amax/127), CMP_KV_DSMLA_INT8=1
ROW_BYTES = {FMT_FP8: 576, FMT_DSMLA: 656, FMT_DSMLA_I8: 656}
KV_DSMLA_INT8 = os.environ.get("CMP_KV_DSMLA_INT8", "0") == "1"
# FMT_DSMLA_I8 runs the latent dots in fp16 (int8 -> fp16 is 1.25 instr/value); p * scale
# is pre-multiplied by 256 to stay in fp16's normal range (undone via out_scale).

_DV = 512
_MERGE_DV_TILE = 128
_DV_C = tl.constexpr(_DV)

# Tunables (env overrides are for benchmarking only).
# e4m3 decode: 1 = PTX bit expand to exact bf16 (+ x2^120 FMA), 0 = LUT,
# 2 = PTX bit expand to fp16 x 2^-8 (no FMA; the 2^8 goes into the scales, dots in fp16)
_DECODE = int(os.environ.get("CMP_KVFP8_DECODE", "1"))
_BLOCK_H = int(os.environ.get("CMP_KVFP8_BLOCK_H", "16"))
_SPLIT_BLOCK_N = int(os.environ.get("CMP_KVFP8_SPLIT_BN", "32"))
_SPLIT_WARPS = int(os.environ.get("CMP_KVFP8_SPLIT_WARPS", "4"))
_SPLIT_STAGES = int(os.environ.get("CMP_KVFP8_SPLIT_STAGES", "2"))
_FINAL_BLOCK_N = int(os.environ.get("CMP_KVFP8_FINAL_BN", "32"))
_FINAL_WARPS = int(os.environ.get("CMP_KVFP8_FINAL_WARPS", "4"))
_FINAL_STAGES = int(os.environ.get("CMP_KVFP8_FINAL_STAGES", "2"))
# 64 keys per split min: T=1 decode 33 -> 27 us (bf16 kernel 25-29 us), others unchanged.
_MIN_TOPK_PER_SPLIT = int(os.environ.get("CMP_KVFP8_MIN_PER_SPLIT", "64"))
_SPLIT_MAX_OCCUPANCY = 4
# 0 = the bf16 kernel's split policy (no split once tokens x head groups >= SMs/4, e.g. MTP
# verify at 4 streams = 12 tokens); >0 = split until ~WAVES x SMs programs at any batch.
# 2.0 measured best on CMP 170HX (74 SMs): T=12 decode 2.5x faster, T=1..4 equal or better.
_SPLIT_WAVES = float(os.environ.get("CMP_KVFP8_SPLIT_WAVES", "2"))

# e4m3fn byte -> bf16, exact for every finite code (NaN codes 0x7F/0xFF never occur:
# the writer saturates). Byte b sits in bits [8,16) of a 16-bit lane after prmt;
# sign stays at bit 15, exponent+mantissa (7 bits) move to bits [4,11) = the bf16
# exponent's low 4 bits + top 3 mantissa bits, i.e. the value times 2^-120 (exponent
# bias 127 vs 7; e4m3 subnormals land on bf16 subnormals with the same 2^-120 factor).
# One fma.rn.bf16x2 by 2^120 (0x7B80) with a -0 addend restores it exactly.
_E4M3X4_TO_BF16_ASM = tl.constexpr(
    """
{
.reg .b32 t0, t1, s0, s1, km, kc, kz, k0;
mov.b32 k0, 0;
mov.b32 km, 0x07F007F0;
mov.b32 kc, 0x7B807B80;
mov.b32 kz, 0x80008000;
prmt.b32 t0, $2, k0, 0x1404;
prmt.b32 t1, $2, k0, 0x3424;
shr.b32 s0, t0, 4;
shr.b32 s1, t1, 4;
and.b32 s0, s0, km;
and.b32 s1, s1, km;
lop3.b32 t0, t0, kz, s0, 0xEA;
lop3.b32 t1, t1, kz, s1, 0xEA;
fma.rn.bf16x2 $0, t0, kc, kz;
fma.rn.bf16x2 $1, t1, kc, kz;
}
"""
)


# e4m3fn byte -> fp16 equal to value * 2^-8 (exact; e4m3 subnormals -> fp16 subnormals).
# Same placement as above, shifted so the 4 exponent bits land in fp16's exponent field.
_E4M3X4_TO_F16RAW_ASM = tl.constexpr(
    """
{
.reg .b32 t0, t1, s0, s1, km, kz, k0;
mov.b32 k0, 0;
mov.b32 km, 0x3F803F80;
mov.b32 kz, 0x80008000;
prmt.b32 t0, $2, k0, 0x1404;
prmt.b32 t1, $2, k0, 0x3424;
shr.b32 s0, t0, 1;
shr.b32 s1, t1, 1;
and.b32 s0, s0, km;
and.b32 s1, s1, km;
lop3.b32 $0, t0, kz, s0, 0xEA;
lop3.b32 $1, t1, kz, s1, 0xEA;
}
"""
)

# int8 byte -> exact fp16: bias to unsigned (xor 0x80), place under 0x64 (fp16 1024.0,
# 10 mantissa bits hold u = b + 128 exactly), subtract 1152 = 1024 + 128.
_I8X4_TO_F16_ASM = tl.constexpr(
    """
{
.reg .b32 x, a0, a1, km, kb, ks;
mov.b32 km, 0x80808080;
mov.b32 kb, 0x64646464;
mov.b32 ks, 0x64806480;
xor.b32 x, $2, km;
prmt.b32 a0, x, kb, 0x5140;
prmt.b32 a1, x, kb, 0x7362;
sub.f16x2 $0, a0, ks;
sub.f16x2 $1, a1, ks;
}
"""
)


@triton.jit
def _e4m3_to_bf16(u, lut_ptr, DECODE: tl.constexpr):
    if DECODE == 0:
        return tl.load(lut_ptr + u.to(tl.int32))
    else:
        return tl.inline_asm_elementwise(
            asm=_E4M3X4_TO_BF16_ASM,
            constraints="=r,=r,r",
            args=[u],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=4,
        )


@triton.jit
def _dec(u, lut_ptr, FMT: tl.constexpr, DECODE: tl.constexpr):
    if FMT == 3:
        return tl.inline_asm_elementwise(
            asm=_I8X4_TO_F16_ASM,
            constraints="=r,=r,r",
            args=[u],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    elif DECODE == 2:
        return tl.inline_asm_elementwise(
            asm=_E4M3X4_TO_F16RAW_ASM,
            constraints="=r,=r,r",
            args=[u],
            dtype=tl.float16,
            is_pure=True,
            pack=4,
        )
    else:
        return _e4m3_to_bf16(u, lut_ptr, DECODE)


@triton.jit
def _kvfp8_loop(
    q_ptr,
    kv_ptr,
    idx_ptr,
    lut_ptr,
    cur_q,
    cur_head,
    mask_h,
    split_start,
    split_end,
    num_slots,
    block_stride,
    stride_q_token,
    stride_q_head,
    stride_idx_token,
    qk_scale,
    FMT: tl.constexpr,
    DECODE: tl.constexpr,
    ROW: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_t = tl.arange(0, 128)
    offs_pe = tl.arange(0, 64)
    q_row = q_ptr + cur_q * stride_q_token + cur_head[:, None] * stride_q_head
    mq = mask_h[:, None]
    q0 = tl.load(q_row + offs_t[None, :], mask=mq, other=0.0)
    q1 = tl.load(q_row + 128 + offs_t[None, :], mask=mq, other=0.0)
    q2 = tl.load(q_row + 256 + offs_t[None, :], mask=mq, other=0.0)
    q3 = tl.load(q_row + 384 + offs_t[None, :], mask=mq, other=0.0)
    qpe = tl.load(q_row + 512 + offs_pe[None, :], mask=mq, other=0.0)
    if FMT == 3 or DECODE == 2:  # latent dots in fp16 (decoded values are exact in fp16)
        q0 = tl.clamp(q0.to(tl.float32), -65504.0, 65504.0).to(tl.float16)
        q1 = tl.clamp(q1.to(tl.float32), -65504.0, 65504.0).to(tl.float16)
        q2 = tl.clamp(q2.to(tl.float32), -65504.0, 65504.0).to(tl.float16)
        q3 = tl.clamp(q3.to(tl.float32), -65504.0, 65504.0).to(tl.float16)
        if FMT == 1:  # e4m3 rope too
            qpe = tl.clamp(qpe.to(tl.float32), -65504.0, 65504.0).to(tl.float16)

    # Finite sentinel, same NaN guard as the bf16 kernel (masked V rows are 0).
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc0 = tl.zeros([BLOCK_H, 128], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, 128], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, 128], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, 128], dtype=tl.float32)

    for start in range(split_start, split_end, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        idx = tl.load(
            idx_ptr + cur_q * stride_idx_token + offs_n,
            mask=offs_n < split_end,
            other=-1,
        )
        valid = (idx >= 0) & (idx < num_slots)
        sidx = tl.where(valid, idx, 0)
        row = (
            kv_ptr
            + (sidx // PAGE).to(tl.int64) * block_stride
            + (sidx % PAGE).to(tl.int64) * ROW
        )
        m2 = valid[:, None]
        k0 = _dec(
            tl.load(row[:, None] + offs_t[None, :], mask=m2, other=0), lut_ptr, FMT, DECODE
        )
        k1 = _dec(
            tl.load(row[:, None] + (128 + offs_t)[None, :], mask=m2, other=0),
            lut_ptr,
            FMT,
            DECODE,
        )
        k2 = _dec(
            tl.load(row[:, None] + (256 + offs_t)[None, :], mask=m2, other=0),
            lut_ptr,
            FMT,
            DECODE,
        )
        k3 = _dec(
            tl.load(row[:, None] + (384 + offs_t)[None, :], mask=m2, other=0),
            lut_ptr,
            FMT,
            DECODE,
        )
        if FMT >= 2:
            srow = row.to(tl.pointer_type(tl.float32))
            s0 = tl.load(srow + 128, mask=valid, other=0.0)
            s1 = tl.load(srow + 129, mask=valid, other=0.0)
            s2 = tl.load(srow + 130, mask=valid, other=0.0)
            s3 = tl.load(srow + 131, mask=valid, other=0.0)
            if FMT == 2 and DECODE == 2:  # decoded latent = value * 2^-8
                s0 = s0 * 256.0
                s1 = s1 * 256.0
                s2 = s2 * 256.0
                s3 = s3 * 256.0
            prow = (row + 528).to(tl.pointer_type(tl.bfloat16))
            kpe = tl.load(prow[:, None] + offs_pe[None, :], mask=m2, other=0.0)
            qk = tl.dot(q0, tl.trans(k0)) * s0[None, :]
            qk += tl.dot(q1, tl.trans(k1)) * s1[None, :]
            qk += tl.dot(q2, tl.trans(k2)) * s2[None, :]
            qk += tl.dot(q3, tl.trans(k3)) * s3[None, :]
            qk = tl.dot(qpe, tl.trans(kpe), qk)
        else:
            kpe = _dec(
                tl.load(row[:, None] + (512 + offs_pe)[None, :], mask=m2, other=0),
                lut_ptr,
                FMT,
                DECODE,
            )
            qk = tl.dot(q0, tl.trans(k0))
            qk = tl.dot(q1, tl.trans(k1), qk)
            qk = tl.dot(q2, tl.trans(k2), qk)
            qk = tl.dot(q3, tl.trans(k3), qk)
            qk = tl.dot(qpe, tl.trans(kpe), qk)

        qk *= qk_scale
        qk = tl.where(mq & valid[None, :], qk, NEG_LARGE)
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
        re2 = re_scale[:, None]
        if FMT == 3:
            acc0 = tl.dot((p * (s0 * 256.0)[None, :]).to(tl.float16), k0, acc0 * re2)
            acc1 = tl.dot((p * (s1 * 256.0)[None, :]).to(tl.float16), k1, acc1 * re2)
            acc2 = tl.dot((p * (s2 * 256.0)[None, :]).to(tl.float16), k2, acc2 * re2)
            acc3 = tl.dot((p * (s3 * 256.0)[None, :]).to(tl.float16), k3, acc3 * re2)
        elif FMT == 2 and DECODE == 2:  # s already carries the 2^8
            acc0 = tl.dot((p * s0[None, :]).to(tl.float16), k0, acc0 * re2)
            acc1 = tl.dot((p * s1[None, :]).to(tl.float16), k1, acc1 * re2)
            acc2 = tl.dot((p * s2[None, :]).to(tl.float16), k2, acc2 * re2)
            acc3 = tl.dot((p * s3[None, :]).to(tl.float16), k3, acc3 * re2)
        elif FMT == 1 and DECODE == 2:
            pb = (p * 256.0).to(tl.float16)
            acc0 = tl.dot(pb, k0, acc0 * re2)
            acc1 = tl.dot(pb, k1, acc1 * re2)
            acc2 = tl.dot(pb, k2, acc2 * re2)
            acc3 = tl.dot(pb, k3, acc3 * re2)
        elif FMT == 2:
            acc0 = tl.dot((p * s0[None, :]).to(tl.bfloat16), k0, acc0 * re2)
            acc1 = tl.dot((p * s1[None, :]).to(tl.bfloat16), k1, acc1 * re2)
            acc2 = tl.dot((p * s2[None, :]).to(tl.bfloat16), k2, acc2 * re2)
            acc3 = tl.dot((p * s3[None, :]).to(tl.bfloat16), k3, acc3 * re2)
        else:
            pb = p.to(tl.bfloat16)
            acc0 = tl.dot(pb, k0, acc0 * re2)
            acc1 = tl.dot(pb, k1, acc1 * re2)
            acc2 = tl.dot(pb, k2, acc2 * re2)
            acc3 = tl.dot(pb, k3, acc3 * re2)

    return acc0, acc1, acc2, acc3, e_max, e_sum


@triton.jit
def _kvfp8_final_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    lut_ptr,
    kscale_ptr,
    out_ptr,
    num_slots,
    block_stride,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_idx_token,
    stride_out_token,
    stride_out_head,
    sm_scale_log2,
    TOPK: tl.constexpr,
    FMT: tl.constexpr,
    DECODE: tl.constexpr,
    ROW: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < h_q
    if FMT == 1:
        ks = tl.load(kscale_ptr)
        if DECODE == 2:  # decoded e4m3 = value * 2^-8 (scores); PV uses p * 2^8
            qk_scale = sm_scale_log2 * ks * 256.0
        else:
            qk_scale = sm_scale_log2 * ks
        out_scale = ks
    elif FMT == 3:
        qk_scale = sm_scale_log2
        out_scale = 1.0 / 256.0
    else:
        qk_scale = sm_scale_log2
        out_scale = 1.0
    acc0, acc1, acc2, acc3, e_max, e_sum = _kvfp8_loop(
        q_ptr,
        kv_ptr,
        idx_ptr,
        lut_ptr,
        cur_q,
        cur_head,
        mask_h,
        0,
        TOPK,
        num_slots,
        block_stride,
        stride_q_token,
        stride_q_head,
        stride_idx_token,
        qk_scale,
        FMT,
        DECODE,
        ROW,
        PAGE,
        BLOCK_H,
        BLOCK_N,
    )
    inv = out_scale / tl.where(e_sum > 0, e_sum, 1.0)
    offs_t = tl.arange(0, 128)
    o = out_ptr + cur_q * stride_out_token + cur_head[:, None] * stride_out_head
    mo = mask_h[:, None]
    tl.store(o + offs_t[None, :], (acc0 * inv[:, None]).to(tl.bfloat16), mask=mo)
    tl.store(o + (128 + offs_t)[None, :], (acc1 * inv[:, None]).to(tl.bfloat16), mask=mo)
    tl.store(o + (256 + offs_t)[None, :], (acc2 * inv[:, None]).to(tl.bfloat16), mask=mo)
    tl.store(o + (384 + offs_t)[None, :], (acc3 * inv[:, None]).to(tl.bfloat16), mask=mo)


@triton.jit
def _kvfp8_split_kernel(
    q_ptr,
    kv_ptr,
    idx_ptr,
    lut_ptr,
    kscale_ptr,
    mid_ptr,
    num_slots,
    block_stride,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_idx_token,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    sm_scale_log2,
    TOPK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    FMT: tl.constexpr,
    DECODE: tl.constexpr,
    ROW: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOGE2_C: tl.constexpr,
):
    cur_q = tl.program_id(0)
    cur_head = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split_id = tl.program_id(2)
    mask_h = cur_head < h_q
    if FMT == 1:
        ks = tl.load(kscale_ptr)
        if DECODE == 2:  # decoded e4m3 = value * 2^-8 (scores); PV uses p * 2^8
            qk_scale = sm_scale_log2 * ks * 256.0
        else:
            qk_scale = sm_scale_log2 * ks
        out_scale = ks
    elif FMT == 3:
        qk_scale = sm_scale_log2
        out_scale = 1.0 / 256.0
    else:
        qk_scale = sm_scale_log2
        out_scale = 1.0
    SPLIT: tl.constexpr = TOPK // NUM_SPLITS
    split_start = split_id * SPLIT
    acc0, acc1, acc2, acc3, e_max, e_sum = _kvfp8_loop(
        q_ptr,
        kv_ptr,
        idx_ptr,
        lut_ptr,
        cur_q,
        cur_head,
        mask_h,
        split_start,
        split_start + SPLIT,
        num_slots,
        block_stride,
        stride_q_token,
        stride_q_head,
        stride_idx_token,
        qk_scale,
        FMT,
        DECODE,
        ROW,
        PAGE,
        BLOCK_H,
        BLOCK_N,
    )
    # Empty split: store 0 (not NaN) so the merge's 0-weight term stays finite.
    inv = out_scale / tl.where(e_sum > 0, e_sum, 1.0)
    offs_t = tl.arange(0, 128)
    m = (
        mid_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_id * stride_mid_split
    )
    mo = mask_h[:, None]
    tl.store(m + offs_t[None, :], acc0 * inv[:, None], mask=mo)
    tl.store(m + (128 + offs_t)[None, :], acc1 * inv[:, None], mask=mo)
    tl.store(m + (256 + offs_t)[None, :], acc2 * inv[:, None], mask=mo)
    tl.store(m + (384 + offs_t)[None, :], acc3 * inv[:, None], mask=mo)
    lse_ptr = (
        mid_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_id * stride_mid_split
        + _DV_C
    )
    tl.store(lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2_C, mask=mask_h)



@functools.lru_cache(maxsize=256)
def choose_num_splits(
    num_tokens: int, num_head_groups: int, topk: int, sm_count: int
) -> int:
    """Power-of-2 split count filling the GPU without dropping a split below
    _MIN_TOPK_PER_SPLIT keys (see _SPLIT_WAVES)."""
    baseline = num_tokens * num_head_groups
    if baseline == 0:
        return 1
    if _SPLIT_WAVES > 0:
        max_splits = max(1, int(_SPLIT_WAVES * sm_count) // baseline)
    else:
        if baseline * _SPLIT_MAX_OCCUPANCY >= sm_count:
            return 1
        max_splits = max(1, sm_count // baseline)
    ideal = triton.next_power_of_2(max(1, topk // _MIN_TOPK_PER_SPLIT))
    max_splits = 1 << (max_splits.bit_length() - 1)
    n = min(ideal, max_splits)
    while n > 1 and topk % (n * _SPLIT_BLOCK_N) != 0:
        n //= 2
    return max(1, n)


_ONES: dict[torch.device, torch.Tensor] = {}
_WRITER_CHECKED = False


def dsmla_fmt() -> int:
    """Reader format for --kv-cache-dtype fp8_ds_mla; verifies the mounted writer
    (models/deepseek_v32/common/kernels.py) uses the same payload type."""
    global _WRITER_CHECKED
    if not _WRITER_CHECKED:
        from vllm.models.deepseek_v32.common import kernels as _w

        writer_i8 = bool(getattr(_w, "CMP_KV_DSMLA_INT8", False))
        if writer_i8 != KV_DSMLA_INT8:
            raise RuntimeError(
                f"kvfp8: CMP_KV_DSMLA_INT8 reader={KV_DSMLA_INT8} writer={writer_i8}; "
                "mount patches/ported/kvfp8_dsv32_kernels.py for the int8 payload"
            )
        _WRITER_CHECKED = True
    return FMT_DSMLA_I8 if KV_DSMLA_INT8 else FMT_DSMLA


def _ones(device: torch.device) -> torch.Tensor:
    t = _ONES.get(device)
    if t is None:
        t = torch.ones(1, dtype=torch.float32, device=device)
        _ONES[device] = t
    return t


def kvfp8_sparse_mla_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    slots: torch.Tensor,
    sm_scale: float,
    fmt: int,
    k_scale: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
    decode: int | None = None,
    block_h: int | None = None,
) -> torch.Tensor:
    """Sparse MLA over an fp8 paged cache.

    q:        [T, H, 576] bf16 (absorbed latent query + rope query)
    kv_cache: [num_blocks, block_size, ROW] uint8 (any block stride; rows contiguous)
    slots:    [T, topk] int32, slot = block_id * block_size + offset, -1 = skip
    k_scale:  fp32 [1] device tensor (FMT_FP8 only; None -> 1.0)
    returns   [T, H, 512] bf16
    """
    T, H, D = q.shape
    assert D == 576, f"kvfp8 sparse MLA expects dim_qk 576, got {D}"
    assert q.dtype == torch.bfloat16 and q.stride(2) == 1
    assert kv_cache.dtype == torch.uint8 and kv_cache.dim() == 3
    row = ROW_BYTES[fmt]
    assert kv_cache.shape[2] == row and kv_cache.stride(2) == 1
    assert kv_cache.stride(1) == row, "rows inside a block must be contiguous"
    num_blocks, page, _ = kv_cache.shape
    assert page & (page - 1) == 0, "block size must be a power of 2"
    assert slots.dtype == torch.int32 and slots.dim() == 2 and slots.stride(1) == 1
    topk = slots.shape[1]
    block_h = block_h or _BLOCK_H
    decode = _DECODE if decode is None else decode
    num_head_groups = triton.cdiv(H, block_h)
    device = q.device
    lut = get_e4m3fn_bf16_lut(device)
    ks = k_scale if (fmt == FMT_FP8 and k_scale is not None) else _ones(device)
    num_slots = num_blocks * page

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = num_compute_units(device.index)
        num_kv_splits = choose_num_splits(T, num_head_groups, topk, sm_count)

    out = torch.empty((T, H, _DV), dtype=torch.bfloat16, device=device)
    if T == 0:
        return out
    if num_kv_splits == 1:
        assert topk % _FINAL_BLOCK_N == 0
        _kvfp8_final_kernel[(T, num_head_groups)](
            q,
            kv_cache,
            slots,
            lut,
            ks,
            out,
            num_slots,
            kv_cache.stride(0),
            H,
            q.stride(0),
            q.stride(1),
            slots.stride(0),
            out.stride(0),
            out.stride(1),
            sm_scale * LOG2E,
            TOPK=topk,
            FMT=fmt,
            DECODE=decode,
            ROW=row,
            PAGE=page,
            BLOCK_H=block_h,
            BLOCK_N=_FINAL_BLOCK_N,
            num_warps=_FINAL_WARPS,
            num_stages=_FINAL_STAGES,
        )
        return out

    assert topk % (num_kv_splits * _SPLIT_BLOCK_N) == 0
    mid = torch.empty(
        (T, H, num_kv_splits, _DV + 1), dtype=torch.float32, device=device
    )
    _kvfp8_split_kernel[(T, num_head_groups, num_kv_splits)](
        q,
        kv_cache,
        slots,
        lut,
        ks,
        mid,
        num_slots,
        kv_cache.stride(0),
        H,
        q.stride(0),
        q.stride(1),
        slots.stride(0),
        mid.stride(0),
        mid.stride(1),
        mid.stride(2),
        sm_scale * LOG2E,
        TOPK=topk,
        NUM_SPLITS=num_kv_splits,
        FMT=fmt,
        DECODE=decode,
        ROW=row,
        PAGE=page,
        BLOCK_H=block_h,
        BLOCK_N=_SPLIT_BLOCK_N,
        LOGE2_C=LOGE2,
        num_warps=_SPLIT_WARPS,
        num_stages=_SPLIT_STAGES,
    )
    _sparse_mla_merge_kernel[(T, H, _DV // _MERGE_DV_TILE)](
        mid_out_ptr=mid,
        out_ptr=out,
        h_q=H,
        stride_mid_token=mid.stride(0),
        stride_mid_head=mid.stride(1),
        stride_mid_split=mid.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=H,
        BLOCK_H=1,
        BLOCK_DV=_DV,
        BLOCK_DV_TILE=_MERGE_DV_TILE,
        num_warps=2,
    )
    return out
