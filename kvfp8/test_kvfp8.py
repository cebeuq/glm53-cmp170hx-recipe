"""FP8 KV for TRITON_MLA_SPARSE: correctness + timing on 1 GPU (serving image).

Mounts: /p = patches/vllm (ro), /model = model dir (ro, optional, for real kv_a_layernorm weights).
Run:  python3 -I /kv/test_kvfp8.py [--quick] [--bench] [--model /model]
"""

import argparse
import importlib.util
import math
import sys
import time

import torch

sys.path.insert(0, "/p")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kf = load("kvfp8_sparse_mla", "/p/kvfp8_sparse_mla.py")
dsk = load("dsv32_kernels_patched", "/p/dsv32_kernels.py")
import os  # noqa: E402

os.environ["CMP_KV_DSMLA_INT8"] = "1"
dsk8 = load("kvfp8_dsv32_kernels_i8", "/p/kvfp8_dsv32_kernels.py")  # int8-payload writer
os.environ.pop("CMP_KV_DSMLA_INT8")
assert dsk8.CMP_KV_DSMLA_INT8 and not kf.KV_DSMLA_INT8
from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (  # noqa: E402
    triton_mla_sparse_attention,
)

DEV = torch.device("cuda:0")
PAGE = 64
SM_SCALE = 256**-0.5  # GLM-5.3 qk_head_dim 256


# ---------------------------------------------------------------- 1. decoder
@triton.jit
def _dec_kernel(u_ptr, lut_ptr, out_ptr, DECODE: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)
    u = tl.load(u_ptr + offs)
    tl.store(out_ptr + offs, kf._e4m3_to_bf16(u, lut_ptr, DECODE))


@triton.jit
def _dec16_kernel(u_ptr, lut_ptr, out_ptr, FMT: tl.constexpr, DECODE: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)
    u = tl.load(u_ptr + offs)
    tl.store(out_ptr + offs, kf._dec(u, lut_ptr, FMT, DECODE))


def test_decoder():
    u = torch.arange(256, dtype=torch.uint8, device=DEV).repeat(4)
    ref = u.view(torch.float8_e4m3fn).to(torch.bfloat16)
    lut = kf.get_e4m3fn_bf16_lut(DEV)
    ok = True
    for mode in (0, 1):
        out = torch.empty(1024, dtype=torch.bfloat16, device=DEV)
        _dec_kernel[(1,)](u, lut, out, DECODE=mode, N=1024)
        finite = ~torch.isnan(ref)
        same = (out[finite].view(torch.int16) == ref[finite].view(torch.int16)).all().item()
        print(f"decoder mode={mode}: bit-exact on all 254x4 finite codes: {same}")
        ok &= same
    # e4m3 -> fp16 * 2^-8 (DECODE 2) and int8 -> fp16 (FMT 3)
    out = torch.empty(1024, dtype=torch.float16, device=DEV)
    _dec16_kernel[(1,)](u, lut, out, FMT=2, DECODE=2, N=1024)
    finite = ~torch.isnan(ref)
    same = torch.equal(out.float()[finite] * 256.0, ref.float()[finite])
    print(f"decoder e4m3->fp16 raw (x256): exact on all finite codes: {same}")
    ok &= same
    _dec16_kernel[(1,)](u, lut, out, FMT=3, DECODE=1, N=1024)
    same = torch.equal(out.float(), u.view(torch.int8).float())
    print(f"decoder int8->fp16: exact on all 256x4 codes: {same}")
    ok &= same
    return ok


# ---------------------------------------------------------------- 2. writer
def rope_cache(max_pos, dim=64, theta=8e6):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    f = torch.outer(t, inv)
    return torch.cat([f.cos(), f.sin()], dim=-1).to(DEV)


def write_cache(kv_c, k_pe, pos, slots, kv_w, cos_sin, dtype_str, num_blocks, k_scale):
    """Write tokens through the real (patched) fused_norm_rope kernel.
    dtype_str 'int8_ds_mla' = fp8_ds_mla layout via the int8-payload writer."""
    n = kv_c.shape[0]
    writer = dsk
    if dtype_str == "int8_ds_mla":
        writer, dtype_str = dsk8, "fp8_ds_mla"
    if dtype_str in ("auto", "bfloat16"):
        cache = torch.zeros(num_blocks, PAGE, 576, dtype=torch.bfloat16, device=DEV)
    else:
        row = 656 if dtype_str == "fp8_ds_mla" else 576
        cache = torch.zeros(num_blocks, PAGE, row, dtype=torch.uint8, device=DEV)
    q_c = torch.randn(n, 2048, dtype=torch.bfloat16, device=DEV)
    topk_buf = torch.empty(n, 2048, dtype=torch.int32, device=DEV)
    writer.fused_norm_rope(
        pos, q_c, torch.ones(2048, dtype=torch.bfloat16, device=DEV), 1e-5,
        kv_c, kv_w, 1e-5, k_pe, cos_sin,
        None, None, None, 1e-6, None, topk_buf,
        slot_mapping=slots, indexer_k_cache=None, mla_kv_cache=cache,
        mla_kv_cache_dtype=dtype_str, mla_k_scale=k_scale, has_indexer=False,
    )
    return cache


def dequant(cache, fmt, k_scale):
    """Torch dequant of an fp8 cache -> fp32 [blocks*PAGE, 576] (exact values)."""
    nb = cache.shape[0]
    c = cache.reshape(nb * PAGE, -1)
    if fmt == kf.FMT_FP8:
        return c.view(torch.float8_e4m3fn).float() * k_scale.float()
    if fmt == kf.FMT_DSMLA_I8:
        lat = c[:, :512].contiguous().view(torch.int8).float()
    else:
        lat = c[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    sc = c[:, 512:528].contiguous().view(torch.float32)  # [R, 4]
    lat = (lat.view(-1, 4, 128) * sc[:, :, None]).view(-1, 512)
    rope = c[:, 528:656].contiguous().view(torch.bfloat16).float()
    return torch.cat([lat, rope], dim=-1)


def ref_attn(q, kv_rows, slots):
    """fp32 torch sparse MLA: q [T,H,576], kv_rows [R,576], slots [T,K] -> [T,H,512]."""
    out = []
    for t in range(q.shape[0]):
        s = slots[t]
        s = s[s >= 0].long()
        k = kv_rows[s].float()
        logits = (q[t].float() @ k.T) * SM_SCALE
        p = torch.softmax(logits, dim=-1)
        out.append(p @ k[:, :512])
    return torch.stack(out)


def rel_err(a, b):
    d = (a.float() - b.float()).flatten(1)
    n = b.float().flatten(1).norm(dim=1).clamp_min(1e-12)
    r = d.norm(dim=1) / n
    return r.mean().item(), r.max().item()


def make_inputs(n_ctx, T, topk, H, kv_w, kpe_std, q_logit_std, outlier=False, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    kv_c = torch.randn(n_ctx, 512, generator=g, device=DEV)
    if outlier:  # a few heavy channels, like real residual streams
        kv_c[:, :8] *= 12.0
    kv_c = kv_c.to(torch.bfloat16)
    k_pe = (torch.randn(n_ctx, 64, generator=g, device=DEV) * kpe_std).to(torch.bfloat16)
    pos = torch.arange(n_ctx, device=DEV, dtype=torch.int64)
    perm = torch.randperm(n_ctx, generator=g, device=DEV)  # scattered physical slots
    slots_phys = perm.to(torch.int64)
    num_blocks = triton.cdiv(n_ctx, PAGE)
    # topk per query: random distinct positions -> physical slots
    sel = torch.stack(
        [torch.randperm(n_ctx, generator=g, device=DEV)[:topk] for _ in range(T)]
    )
    slots_q = slots_phys[sel].to(torch.int32)
    if topk > n_ctx:
        raise ValueError
    q = torch.randn(T, H, 576, generator=g, device=DEV)
    return kv_c, k_pe, pos, slots_phys, num_blocks, slots_q, q


def calibrate_q(q, kv_rows, slots, target_std):
    lg = []
    for t in range(min(4, q.shape[0])):
        k = kv_rows[slots[t].long()].float()
        lg.append((q[t].float() @ k.T * SM_SCALE).std())
    s = target_std / torch.stack(lg).mean()
    return (q * s).to(torch.bfloat16)


def test_attention(kv_w, label, kpe_std=1.0, outlier=False, n_ctx=8192, T=4, H=32):
    cos_sin = rope_cache(n_ctx)
    kv_c, k_pe, pos, slots_phys, nb, slots_q, q = make_inputs(
        n_ctx, T, 2048, H, kv_w, kpe_std, 3.0, outlier
    )
    one = torch.ones(1, dtype=torch.float32, device=DEV)
    c_bf = write_cache(kv_c, k_pe, pos, slots_phys, kv_w, cos_sin, "bfloat16", nb, one)
    rows_bf = c_bf.view(nb * PAGE, 576)
    q = calibrate_q(q, rows_bf, slots_q, 3.0)
    ref32 = ref_attn(q, rows_bf, slots_q)
    out_bf = triton_mla_sparse_attention(
        q, rows_bf.view(-1, 1, 576), slots_q.view(T, 1, -1), SM_SCALE
    )
    e_bf = rel_err(out_bf, ref32)
    amax_lat = rows_bf[:, :512].float().abs().max().item()
    amax_pe = rows_bf[:, 512:].float().abs().max().item()
    print(f"[{label}] |latent|max {amax_lat:.3g}  |rope|max {amax_pe:.3g}")
    print(f"  bf16 kernel vs fp32 ref: mean {e_bf[0]:.2e} max {e_bf[1]:.2e}")
    ok = True
    for dt, fmt in (("fp8", kf.FMT_FP8), ("fp8_ds_mla", kf.FMT_DSMLA),
                    ("int8_ds_mla", kf.FMT_DSMLA_I8)):
        c8 = write_cache(kv_c, k_pe, pos, slots_phys, kv_w, cos_sin, dt, nb, one)
        rows_dq = dequant(c8, fmt, one)
        # storage error of the cache itself
        d = (rows_dq.float() - rows_bf.float())
        lat_e = (d[:, :512].norm(dim=1) / rows_bf[:, :512].float().norm(dim=1).clamp_min(1e-9))
        pe_e = (d[:, 512:].norm(dim=1) / rows_bf[:, 512:].float().norm(dim=1).clamp_min(1e-9))
        for split in (1, None):  # final (prefill) kernel and split (decode) kernel
            out8 = kf.kvfp8_sparse_mla_attention(
                q, c8, slots_q, SM_SCALE, fmt, one, num_kv_splits=split
            )
            # kernel exactness: same dequantized cache through the bf16 kernel
            out_dq = triton_mla_sparse_attention(
                q, rows_dq.to(torch.bfloat16).view(-1, 1, 576), slots_q.view(T, 1, -1), SM_SCALE,
                num_kv_splits=split or 0,
            )
            ref_dq = ref_attn(q, rows_dq, slots_q)
            ek = rel_err(out8, ref_dq)
            ek_bf = rel_err(out_dq, ref_dq)
            eq = rel_err(out8, ref32)
            tag = "final" if split == 1 else "split"
            print(
                f"  {dt:11s} {tag}: cache rel err latent {lat_e.mean():.2e} rope {pe_e.mean():.2e} | "
                f"kernel vs fp32-ref(dequant) {ek[0]:.2e}/{ek[1]:.2e} (bf16 kernel same cache "
                f"{ek_bf[0]:.2e}) | end-to-end vs fp32-ref(bf16 cache) {eq[0]:.2e}/{eq[1]:.2e}"
            )
            ok &= ek[1] < 3 * max(ek_bf[1], 1e-2)
            ok &= not torch.isnan(out8).any().item()
            if fmt != kf.FMT_DSMLA_I8:  # other e4m3 decoders give the same result
                for dm in (0, 2):
                    o2 = kf.kvfp8_sparse_mla_attention(
                        q, c8, slots_q, SM_SCALE, fmt, one, num_kv_splits=split, decode=dm
                    )
                    e2 = rel_err(o2, ref_dq)
                    print(f"      decode={dm}: kernel vs fp32-ref(dequant) {e2[0]:.2e}/{e2[1]:.2e}")
                    ok &= e2[1] < 3 * max(ek_bf[1], 1e-2)
    return ok


def test_padding_and_layout():
    """-1 indices, all-masked rows, and a block stride larger than the page."""
    H, T = 32, 3
    nb = 40
    row = 656
    big = torch.randint(0, 255, (nb, PAGE * 2, row), dtype=torch.uint8, device=DEV)
    big[..., 512:528] = 0
    sc = torch.full((nb, PAGE * 2, 4), 0.01, dtype=torch.float32, device=DEV)
    big[..., 512:528] = sc.view(torch.uint8).view(nb, PAGE * 2, 16)
    big[..., :512] &= 0x7E  # avoid NaN codes / keep finite
    rope = torch.randn(nb, PAGE * 2, 64, device=DEV).to(torch.bfloat16)
    big[..., 528:656] = rope.view(torch.uint8).view(nb, PAGE * 2, 128)
    cache = big[:, :PAGE]  # block stride = 2 pages: other layers' pages in between
    q = torch.randn(T, H, 576, device=DEV).to(torch.bfloat16)
    slots = torch.randint(0, nb * PAGE, (T, 2048), dtype=torch.int32, device=DEV)
    slots[0, 1000:] = -1
    slots[1, :] = -1  # fully masked token -> 0 output, no NaN
    ok = True
    for fmt in (kf.FMT_DSMLA, kf.FMT_DSMLA_I8):
        for split in (1, None):
            out = kf.kvfp8_sparse_mla_attention(q, cache, slots, SM_SCALE, fmt, num_kv_splits=split)
            dense = dequant(cache.contiguous(), fmt, None)
            ref = ref_attn(q, dense, slots)
            e = rel_err(out[[0, 2]], ref[[0, 2]])
            zero = out[1].abs().max().item()
            nan = torch.isnan(out).any().item()
            print(f"padding/stride fmt {fmt} split {split}: rel err {e[0]:.2e}/{e[1]:.2e}, "
                  f"masked-row |out| {zero}, nan {nan}")
            ok &= e[1] < 2e-2 and zero == 0 and not nan
    return ok


def test_backend_glue():
    """Patched XPUMLASparseImpl.forward_mqa fp8 branch (slot conversion, views, k_scale)."""
    from types import SimpleNamespace

    sys.modules["vllm.v1.attention.ops.kvfp8_sparse_mla"] = kf
    xpu = load("xpu_mla_sparse_patched", "/p/xpu_mla_sparse.py")
    assert "fp8_ds_mla" in xpu.XPUMLASparseBackend.supported_kv_cache_dtypes
    H, T, nreq, nb = 32, 5, 2, 64
    ok = True
    for dt, fmt in (("fp8", kf.FMT_FP8), ("fp8_ds_mla", kf.FMT_DSMLA),
                    ("fp8_ds_mla", kf.FMT_DSMLA_I8)):
        i8 = fmt == kf.FMT_DSMLA_I8
        kf.KV_DSMLA_INT8 = i8
        kf._WRITER_CHECKED = False
        sys.modules["vllm.models.deepseek_v32.common.kernels"] = dsk8 if i8 else dsk
        row = kf.ROW_BYTES[fmt]
        cache = torch.randint(0, 0x7E, (nb, PAGE, row), dtype=torch.uint8, device=DEV)
        if fmt != kf.FMT_FP8:
            cache[..., 512:528] = torch.full((nb, PAGE, 4), 0.02, device=DEV).view(torch.uint8).view(nb, PAGE, 16)
            cache[..., 528:656] = torch.randn(nb, PAGE, 64, device=DEV).to(torch.bfloat16).view(torch.uint8).view(nb, PAGE, 128)
        kscale = torch.tensor([0.05], device=DEV)
        impl = object.__new__(xpu.XPUMLASparseImpl)
        impl.kv_cache_dtype = dt
        impl.softmax_scale = SM_SCALE
        impl.num_heads = H
        impl._indexer = None
        impl.topk_indices_buffer = torch.randint(-1, 1500, (16, 2048), dtype=torch.int32, device=DEV)
        bt = torch.stack([torch.randperm(nb, device=DEV)[:32] for _ in range(nreq)]).to(torch.int32)
        md = SimpleNamespace(
            req_id_per_token=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=DEV),
            block_table=bt, block_size=PAGE,
        )
        layer = SimpleNamespace(_k_scale=kscale)
        ql = torch.randn(T, H, 512, device=DEV).to(torch.bfloat16)
        qp = torch.randn(T, H, 64, device=DEV).to(torch.bfloat16)
        kv_in = cache.view(torch.float8_e4m3fn) if fmt == kf.FMT_FP8 else cache
        out, lse = impl.forward_mqa((ql, qp), kv_in, md, layer)
        li = impl.topk_indices_buffer[:T].long()
        req = md.req_id_per_token.long()
        blk = bt[req[:, None], (li.clamp_min(0) // PAGE)]
        slots = torch.where(li >= 0, blk.long() * PAGE + li % PAGE, -1).to(torch.int32)
        dense = dequant(cache, fmt, kscale)
        ref = ref_attn(torch.cat([ql, qp], -1), dense, slots)
        e = rel_err(out, ref)
        print(f"backend glue {dt} fmt {fmt}: out {tuple(out.shape)} lse {lse} rel err {e[0]:.2e}/{e[1]:.2e}")
        ok &= e[1] < 2e-2 and lse is None
    kf.KV_DSMLA_INT8 = False
    kf._WRITER_CHECKED = False
    sys.modules.pop("vllm.models.deepseek_v32.common.kernels", None)
    # mismatch must raise
    sys.modules["vllm.models.deepseek_v32.common.kernels"] = dsk8
    try:
        kf.dsmla_fmt()
        print("writer/reader mismatch NOT detected")
        ok = False
    except RuntimeError as e:
        print("writer/reader mismatch detected:", str(e)[:80])
    sys.modules.pop("vllm.models.deepseek_v32.common.kernels", None)
    kf._WRITER_CHECKED = False
    return ok


def test_cuda_graph():
    """Capture the decode (split) and final paths in a CUDA graph, change cache/indices/
    scale contents in place, replay, compare with eager."""
    H, nb = 32, 128
    ok = True
    for fmt in (kf.FMT_FP8, kf.FMT_DSMLA, kf.FMT_DSMLA_I8):
        row = kf.ROW_BYTES[fmt]
        cache = torch.randint(0, 0x7E, (nb, PAGE, row), dtype=torch.uint8, device=DEV)
        if fmt != kf.FMT_FP8:
            cache[..., 512:528] = torch.full((nb, PAGE, 4), 0.02, device=DEV).view(torch.uint8).view(nb, PAGE, 16)
            cache[..., 528:656] = torch.randn(nb, PAGE, 64, device=DEV).to(torch.bfloat16).view(torch.uint8).view(nb, PAGE, 128)
        ks = torch.tensor([0.03], device=DEV)
        for T in (3, 12):  # decode (split) and a bigger batch
            q = (torch.randn(T, H, 576, device=DEV) * 0.2).to(torch.bfloat16)
            slots = torch.randint(0, nb * PAGE, (T, 2048), dtype=torch.int32, device=DEV)
            fn = lambda: kf.kvfp8_sparse_mla_attention(q, cache, slots, SM_SCALE, fmt, ks)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                fn()
                torch.cuda.synchronize()
                with torch.cuda.graph(g, stream=s):
                    out_g = fn()
            torch.cuda.synchronize()
            cache[:, :, :512] = torch.randint(0, 0x7E, (nb, PAGE, 512), dtype=torch.uint8, device=DEV)
            slots.copy_(torch.randint(-1, nb * PAGE, (T, 2048), dtype=torch.int32, device=DEV))
            q.copy_((torch.randn(T, H, 576, device=DEV) * 0.2).to(torch.bfloat16))
            ks.fill_(0.05)
            g.replay()
            torch.cuda.synchronize()
            ref = fn()
            same = torch.equal(out_g, ref)
            print(f"cuda graph fmt {fmt} T={T}: replay == eager after in-place updates: {same}")
            ok &= same
    return ok


# ---------------------------------------------------------------- 3. timing
def l2_flush(buf):
    buf.zero_()


def bench_one(fn, flush, iters=20):
    """GPU time per call (us) from CUDA-graph replay (no Python launch overhead, as in the
    engine's FULL decode graphs): graph of iters x (flush, fn) minus graph of iters x flush."""
    fn()
    flush()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    g, gf = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.stream(st):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=st):
            for _ in range(iters):
                flush()
                fn()
        with torch.cuda.graph(gf, stream=st):
            for _ in range(iters):
                flush()
    torch.cuda.synchronize()

    def t(graph):
        ts = []
        for _ in range(5):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            graph.replay()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[2]

    tg, tf = t(g), t(gf)
    del g, gf
    return max(tg - tf, 0.0) * 1000 / iters


def bench(H_list=(32, 64), ctxs=(4096, 32768, 131072, 524288), Ts=(1, 2, 3, 4, 12)):
    flushbuf = torch.empty(96 * 2**20, dtype=torch.uint8, device=DEV)
    flush = lambda: l2_flush(flushbuf)  # noqa: E731
    print("decode timing (us, median, L2 flushed): H, ctx, T | bf16 | fp8 | fp8_ds_mla | int8_ds_mla")
    for H in H_list:
        for ctx in ctxs:
            nb = ctx // PAGE
            g = torch.Generator(device=DEV).manual_seed(1)
            bf = torch.randn(nb * PAGE, 576, device=DEV, dtype=torch.bfloat16)
            c8 = torch.randint(0, 0x7E, (nb, PAGE, 576), dtype=torch.uint8, device=DEV)
            cds = torch.randint(0, 0x7E, (nb, PAGE, 656), dtype=torch.uint8, device=DEV)
            cds[..., 512:528] = (
                torch.full((nb, PAGE, 4), 0.01, device=DEV).view(torch.uint8).view(nb, PAGE, 16)
            )
            cds[..., 528:656] = 0
            for T in Ts:
                q = torch.randn(T, H, 576, device=DEV).to(torch.bfloat16) * 0.1
                slots = torch.stack(
                    [torch.randperm(ctx, generator=g, device=DEV)[:2048] for _ in range(T)]
                ).to(torch.int32)
                t_bf = bench_one(
                    lambda: triton_mla_sparse_attention(
                        q, bf.view(-1, 1, 576), slots.view(T, 1, -1), SM_SCALE
                    ),
                    flush,
                )
                t8 = bench_one(
                    lambda: kf.kvfp8_sparse_mla_attention(q, c8, slots, SM_SCALE, kf.FMT_FP8),
                    flush,
                )
                tds = bench_one(
                    lambda: kf.kvfp8_sparse_mla_attention(q, cds, slots, SM_SCALE, kf.FMT_DSMLA),
                    flush,
                )
                ti8 = bench_one(
                    lambda: kf.kvfp8_sparse_mla_attention(q, cds, slots, SM_SCALE, kf.FMT_DSMLA_I8),
                    flush,
                )
                print(f"  H={H:2d} ctx={ctx:7d} T={T:2d} | {t_bf:7.1f} | {t8:7.1f} | {tds:7.1f} | {ti8:7.1f}")
            del bf, c8, cds
            torch.cuda.empty_cache()


def bench_prefill(H=32, T=2048, ctx=131072):
    nb = ctx // PAGE
    bf = torch.randn(nb * PAGE, 576, device=DEV, dtype=torch.bfloat16)
    c8 = torch.randint(0, 0x7E, (nb, PAGE, 576), dtype=torch.uint8, device=DEV)
    cds = torch.randint(0, 0x7E, (nb, PAGE, 656), dtype=torch.uint8, device=DEV)
    cds[..., 512:528] = (
        torch.full((nb, PAGE, 4), 0.01, device=DEV).view(torch.uint8).view(nb, PAGE, 16)
    )
    cds[..., 528:656] = 0
    q = torch.randn(T, H, 576, device=DEV).to(torch.bfloat16) * 0.1
    # prefill: causal-ish windows of 2048 consecutive keys, chunk at the end of ctx
    base = torch.arange(2048, device=DEV, dtype=torch.int32)
    slots = (ctx - 4096 + torch.arange(T, device=DEV, dtype=torch.int32)[:, None] // 2 + base)
    slots = slots.contiguous()
    tiny = torch.empty(1, device=DEV)
    nop = lambda: tiny.zero_()  # noqa: E731
    t_bf = bench_one(
        lambda: triton_mla_sparse_attention(q, bf.view(-1, 1, 576), slots.view(T, 1, -1), SM_SCALE),
        nop, iters=5,
    )
    t8 = bench_one(lambda: kf.kvfp8_sparse_mla_attention(q, c8, slots, SM_SCALE, kf.FMT_FP8), nop, iters=5)
    tds = bench_one(lambda: kf.kvfp8_sparse_mla_attention(q, cds, slots, SM_SCALE, kf.FMT_DSMLA), nop, iters=5)
    ti8 = bench_one(lambda: kf.kvfp8_sparse_mla_attention(q, cds, slots, SM_SCALE, kf.FMT_DSMLA_I8), nop, iters=5)
    print(f"prefill T={T} H={H} (us): bf16 {t_bf:.0f} | fp8 {t8:.0f} | fp8_ds_mla {tds:.0f} | int8_ds_mla {ti8:.0f}")


SWEEP = [
    {},
    {"_SPLIT_WAVES": 0.0},
    {"_DECODE": 0},
    {"_DECODE": 2},
    {"_BLOCK_H": 32, "_SPLIT_WARPS": 8, "_FINAL_WARPS": 8},
    {"_BLOCK_H": 32, "_FINAL_WARPS": 8, "_FINAL_BLOCK_N": 16, "_SPLIT_WARPS": 8},
    {"_SPLIT_WAVES": 4.0},
    {"_MIN_TOPK_PER_SPLIT": 64},
    {"_SPLIT_STAGES": 3},
    {"_FINAL_STAGES": 1},
    {"_FINAL_STAGES": 3},
    {"_FINAL_WARPS": 8},
    {"_FINAL_WARPS": 2, "_FINAL_BLOCK_N": 16},
]


def sweep(ctx=131072, H=32, Ts=(1, 3, 4, 12)):
    flushbuf = torch.empty(96 * 2**20, dtype=torch.uint8, device=DEV)
    flush = lambda: l2_flush(flushbuf)  # noqa: E731
    nb = ctx // PAGE
    g = torch.Generator(device=DEV).manual_seed(1)
    bf = torch.randn(nb * PAGE, 576, device=DEV, dtype=torch.bfloat16)
    cds = torch.randint(0, 0x7E, (nb, PAGE, 656), dtype=torch.uint8, device=DEV)
    cds[..., 512:528] = torch.full((nb, PAGE, 4), 0.01, device=DEV).view(torch.uint8).view(nb, PAGE, 16)
    cds[..., 528:656] = 0
    qs = {T: torch.randn(T, H, 576, device=DEV).to(torch.bfloat16) * 0.1 for T in Ts}
    sl = {
        T: torch.stack([torch.randperm(ctx, generator=g, device=DEV)[:2048] for _ in range(T)]).to(torch.int32)
        for T in Ts
    }
    Tp = 2048
    qp = torch.randn(Tp, H, 576, device=DEV).to(torch.bfloat16) * 0.1
    base = torch.arange(2048, device=DEV, dtype=torch.int32)
    slp = (ctx - 4096 + torch.arange(Tp, device=DEV, dtype=torch.int32)[:, None] // 2 + base).contiguous()
    line = " ".join(
        f"T{T}:{bench_one(lambda: triton_mla_sparse_attention(qs[T], bf.view(-1, 1, 576), sl[T].view(T, 1, -1), SM_SCALE), flush):.1f}"
        for T in Ts
    )
    tiny = torch.empty(1, device=DEV)
    nop = lambda: tiny.zero_()  # noqa: E731
    tp = bench_one(lambda: triton_mla_sparse_attention(qp, bf.view(-1, 1, 576), slp.view(Tp, 1, -1), SM_SCALE), nop, iters=5)
    print(f"sweep ctx={ctx} H={H} (us)  bf16: {line} prefill2048:{tp:.0f}")
    defaults = {k: getattr(kf, k) for c in SWEEP for k in c}
    for cfg in SWEEP:
        for k, v in defaults.items():
            setattr(kf, k, v)
        for k, v in cfg.items():
            setattr(kf, k, v)
        kf.choose_num_splits.cache_clear()
        try:
            line = " ".join(
                f"T{T}:{bench_one(lambda: kf.kvfp8_sparse_mla_attention(qs[T], cds, sl[T], SM_SCALE, kf.FMT_DSMLA), flush):.1f}"
                for T in Ts
            )
            tp = bench_one(lambda: kf.kvfp8_sparse_mla_attention(qp, cds, slp, SM_SCALE, kf.FMT_DSMLA), nop, iters=5)
            print(f"  ds_mla {cfg or 'default'}: {line} prefill2048:{tp:.0f}")
            line = " ".join(
                f"T{T}:{bench_one(lambda: kf.kvfp8_sparse_mla_attention(qs[T], cds, sl[T], SM_SCALE, kf.FMT_DSMLA_I8), flush):.1f}"
                for T in Ts
            )
            tp = bench_one(lambda: kf.kvfp8_sparse_mla_attention(qp, cds, slp, SM_SCALE, kf.FMT_DSMLA_I8), nop, iters=5)
            print(f"  int8   {cfg or 'default'}: {line} prefill2048:{tp:.0f}")
        except Exception as e:  # noqa: BLE001
            print(f"  ds_mla {cfg}: FAILED {type(e).__name__}: {str(e)[:200]}")
    for k, v in defaults.items():
        setattr(kf, k, v)
    kf.choose_num_splits.cache_clear()


def real_kv_weights(model_dir, layers):
    import json
    import os

    from safetensors import safe_open

    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for L in layers:
        name = f"model.layers.{L}.self_attn.kv_a_layernorm.weight"
        with safe_open(os.path.join(model_dir, idx[name]), "pt") as f:
            out[L] = f.get_tensor(name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--only-bench", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    a = ap.parse_args()
    print(torch.cuda.get_device_name(0), "SMs", torch.cuda.get_device_properties(0).multi_processor_count)
    ok = True
    if not (a.only_bench or a.no_tests):
        ok &= test_decoder()
        ok &= test_padding_and_layout()
        ok &= test_backend_glue()
        ok &= test_cuda_graph()
        w1 = torch.ones(512, dtype=torch.bfloat16, device=DEV)
        ok &= test_attention(w1, "unit norm w, rope std 1")
        ok &= test_attention(w1, "unit norm w, rope std 30, outlier ch", kpe_std=30.0, outlier=True)
        if a.model:
            ws = real_kv_weights(a.model, [0, 3, 20, 40, 60, 77, 78])
            for L, w in ws.items():
                wf = w.float()
                print(f"layer {L} kv_a_layernorm.w: |w| mean {wf.abs().mean():.3g} max {wf.abs().max():.3g} min {wf.abs().min():.3g}")
            for L in (3, 40, 77):
                ok &= test_attention(ws[L].to(DEV), f"real kv_a_layernorm w L{L}, rope std 1")
        print("CORRECTNESS", "PASS" if ok else "FAIL")
    if a.sweep:
        sweep()
        sweep(H=64)
    if a.bench or a.only_bench:
        if a.quick:
            bench(H_list=(32,), ctxs=(4096, 131072, 524288), Ts=(1, 3, 4, 12))
        else:
            bench()
        bench_prefill()


if __name__ == "__main__":
    main()
