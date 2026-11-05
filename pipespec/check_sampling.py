#!/usr/bin/env python3
"""check_sampling.py (CPU, stdlib): exactness of the committed-token distribution for greedy and sampled drafts,
standard and block verification, under the PipeSpec chunk A/B protocol with chain drafts.

  docker compose -f pipespec/compose.yaml run --rm check-sampling

Mirrors the fork's kernels (v1/worker/gpu/spec_decode/rejection_sampler_utils.py @2627d2a9):
  standard  (l.634-671): accept iff log p(d) > log u + log q(d); resample from max(p - q, 0) (one-hot q: p with d zeroed)
  block     (l.600-633, 311-390, 48-92, 790-823): p_i = min(1, p_{i-1} p(d_i)/q(d_i)); h_i = R/(R + 1 - p_i) with
            R = sum_x max(p_i M_b(x) - M_s(x), 0) at the next position (one-hot: p_i (1 - M_b(d_{i+1}))), h_last = p_n;
            accepted length = last i with u_i <= h_i; resample from max(p_tau M_b - M_s, 0) (one-hot: M_b with d zeroed)
  drafts    greedy = argmax q (q treated as one-hot), probabilistic = sample q_T (q used in the test)
PipeSpec epoch: drafts d1,d2 (chunk A) and chain d3,d4,d5 (chunk B, issued before A is verified); B is verified iff A
accepted d1,d2 and A's bonus == d3; otherwise the next epoch starts from A's committed tokens.
Check: Monte-Carlo frequency of every committed H-token sequence vs the exact target probability (chi-square and
max |freq - prob| against the MC bound). Also reports expected accepted draft tokens per chunk and P(B valid).
"""
import math, random, sys, itertools, collections

V, H, K = 5, 4, 2
T = 0.9
TOP_P = 0.9


def softmax(ls):
    m = max(ls); e = [math.exp(x - m) for x in ls]; s = sum(e); return [x / s for x in e]


def make_model(seed, mismatch):
    """target p(ctx) and drafter q(ctx) over contexts (tuples) of length < H + 6."""
    rng = random.Random(seed)
    cache = {}

    def dists(ctx):
        if ctx not in cache:
            r = random.Random(hash((seed, ctx)) & 0xffffffff)
            base = [r.gauss(0, 1.5) for _ in range(V)]
            noise = [r.gauss(0, mismatch) for _ in range(V)]
            p_logits = base
            q_logits = [b + n for b, n in zip(base, noise)]
            # processed target: temperature then top_p (as apply_sampling_params does, no penalties here)
            p = softmax([x / T for x in p_logits])
            order = sorted(range(V), key=lambda i: -p[i]); keep = []; acc = 0.0
            for i in order:
                keep.append(i); acc += p[i]
                if acc >= TOP_P: break
            pt = [p[i] if i in keep else 0.0 for i in range(V)]; z = sum(pt); pt = [x / z for x in pt]
            q = softmax([x / T for x in q_logits])  # drafter: temperature only (speculator.py l.402-406)
            cache[ctx] = (pt, q, q_logits)
        return cache[ctx]
    return dists


def sample(dist, rng):
    u = rng.random(); acc = 0.0
    for i, p in enumerate(dist):
        acc += p
        if u < acc: return i
    return len(dist) - 1


def draft_one(q, q_logits, greedy, rng):
    return max(range(V), key=lambda i: q_logits[i]) if greedy else sample(q, rng)


def verify(ps, qs, drafts, onehot, block, rng):
    """ps[i], qs[i]: target / draft dists at positions 0..n (ps has n+1 entries, qs n). Returns committed tokens."""
    n = len(drafts)
    if not block:
        out = []
        for i, d in enumerate(drafts):
            p, q = ps[i], qs[i]
            qd = 1.0 if onehot else q[d]
            u = rng.random()
            if p[d] > u * qd:  # log p > log u + log q
                out.append(d); continue
            if onehot:
                res = [x if j != d else 0.0 for j, x in enumerate(p)]
            else:
                res = [max(x - y, 0.0) for x, y in zip(p, q)]
            z = sum(res)
            if z <= 0:  # cannot happen for exact distributions; guard
                res = p; z = 1.0
            out.append(sample([x / z for x in res], rng)); return out
        out.append(sample(ps[n], rng)); return out
    # block verification
    cum = []; lp = 0.0
    for i, d in enumerate(drafts):
        p, q = ps[i], qs[i]
        qd = 1.0 if onehot else q[d]
        lp = min(lp + (math.log(p[d]) if p[d] > 0 else -math.inf) - math.log(qd), 0.0)
        cum.append(lp)
    acc_len = 0
    for i, d in enumerate(drafts):
        pj = math.exp(cum[i])
        if i < n - 1:
            nd = drafts[i + 1]; pn, qn = ps[i + 1], qs[i + 1]
            if onehot:
                R = pj * (1.0 - pn[nd])
            else:
                R = sum(max(pj * a - b, 0.0) for a, b in zip(pn, qn))
            den = R + 1.0 - pj
            h = R / den if den > 0 else 1.0
        else:
            h = pj
        u = rng.random()
        if u <= h: acc_len = i + 1
    out = list(drafts[:acc_len])
    if acc_len == n:
        out.append(sample(ps[n], rng)); return out
    p, q = ps[acc_len], qs[acc_len]; d = drafts[acc_len]
    if onehot:
        res = [x if j != d else 0.0 for j, x in enumerate(p)]
    else:
        ptau = math.exp(cum[acc_len - 1]) if acc_len > 0 else 1.0
        res = [max(ptau * a - b, 0.0) for a, b in zip(p, q)]
    z = sum(res)
    if z <= 0: res = p; z = 1.0
    out.append(sample([x / z for x in res], rng)); return out


def run_epochs(dists, greedy, onehot, block, rng, stats, greedy_d3=False):
    """One sequence of H committed tokens through PipeSpec epochs. greedy_d3: the CMP_SQ_GREEDY_D3 variant, the chain's
    bonus guess d3 (chain step k) is the argmax while d1,d2,d4,d5 are Gumbel draws (d3 is never verified)."""
    ctx = ()
    while len(ctx) < H:
        # chain of 2K+1 drafts from the drafter alone (stale-state approximation irrelevant here)
        chain = []; c = ctx
        for j in range(2 * K + 1):
            pt, q, ql = dists(c); d = draft_one(q, ql, greedy or (greedy_d3 and j == K), rng); chain.append(d); c = c + (d,)
        d1, d2, d3, d4, d5 = chain
        # chunk A: positions ctx, ctx+d1, ctx+d1+d2
        ps = [dists(ctx)[0], dists(ctx + (d1,))[0], dists(ctx + (d1, d2))[0]]
        qs = [dists(ctx)[1], dists(ctx + (d1,))[1]]
        toks = verify(ps, qs, [d1, d2], onehot, block, rng)
        stats["A"] += 1; stats["A_acc"] += len(toks) - 1
        ctx = ctx + tuple(toks)
        if len(toks) == K + 1 and toks[K] == d3:
            stats["B"] += 1
            base = ctx  # = old ctx + d1 d2 d3
            ps = [dists(base)[0], dists(base + (d4,))[0], dists(base + (d4, d5))[0]]
            qs = [dists(base)[1], dists(base + (d4,))[1]]
            toks = verify(ps, qs, [d4, d5], onehot, block, rng)
            stats["B_acc"] += len(toks) - 1
            ctx = ctx + tuple(toks)
    return ctx[:H]


def exact_probs(dists):
    out = {}
    for seq in itertools.product(range(V), repeat=H):
        p = 1.0; c = ()
        for t in seq:
            p *= dists(c)[0][t]; c = c + (t,)
        out[seq] = p
    return out


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    print(f"V={V} H={H} k={K} T={T} top_p={TOP_P} trials={N} per variant")
    hdr = f"{'mismatch':>8s} {'variant':>22s} {'chi2':>8s} {'dof':>4s} {'maxerr':>8s} {'bound':>8s} {'acc/A':>6s} {'P(B)':>6s} {'acc/B':>6s} {'tok/epoch':>9s} verdict"
    print(hdr)
    bad = 0
    for mismatch in (0.0, 0.5, 1.0):
        dists = make_model(7, mismatch)
        exact = exact_probs(dists)
        for greedy, onehot, block, name in ((True, True, False, "greedy, one-hot q"), (True, True, True, "greedy, one-hot, block"),
                                             (False, False, False, "sampled, real q"), (False, False, True, "sampled, real q, block"),
                                             (False, False, False, "sampled, greedy d3")):
            g3 = name.endswith("greedy d3")
            rng = random.Random(1234 + int(mismatch * 10))
            cnt = collections.Counter(); stats = collections.Counter()
            for _ in range(N):
                cnt[run_epochs(dists, greedy, onehot, block, rng, stats, greedy_d3=g3)] += 1
            chi2 = 0.0; maxerr = 0.0; dof = 0
            for seq, p in exact.items():
                e = N * p
                if e >= 5:
                    chi2 += (cnt[seq] - e) ** 2 / e; dof += 1
                maxerr = max(maxerr, abs(cnt[seq] / N - p))
            dof -= 1
            # chi2 mean dof, sd sqrt(2 dof): flag above mean + 4 sd
            bound = 4 * math.sqrt(0.25 / N)  # 4 sigma of a frequency estimate at p=0.5 (loose)
            ok = chi2 < dof + 4 * math.sqrt(2 * dof) and maxerr < bound
            bad += not ok
            A = max(1, stats["A"]); B = max(1, stats["B"])
            print(f"{mismatch:8.2f} {name:>22s} {chi2:8.1f} {dof:4d} {maxerr:8.4f} {bound:8.4f} {stats['A_acc']/A:6.3f} {stats['B']/A:6.3f} {stats['B_acc']/B:6.3f} {(stats['A_acc']+stats['A']+stats['B_acc']+stats['B'])/A:9.3f} {'OK' if ok else 'FAIL'}")
    print("RESULT", "ALL EXACT" if bad == 0 else f"{bad} variants FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
