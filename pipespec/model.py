#!/usr/bin/env python3
"""PipeSpec throughput model for GLM-5.3 on 8x CMP 170HX (TP2 x PP4, MTP k=2).

Stdlib only. Discrete-event simulation of the 4 PP stages processing the
scheduler's step sequence in order (vLLM: every rank runs steps in step order),
plus a closed form for the rolling stage-1-drafter variant.

  docker compose -f pipespec/compose.yaml run --rm model            # tables
  docker compose -f pipespec/compose.yaml run --rm model --epochs 200000

All times in ms. Parameters are derived in docs/pipespec.md.
"""

import argparse
import random

# ---- timing (ms) -------------------------------------------------------------
P = dict(
    s3=8.2,  # one PP stage forward, 3-token chunk (1 + k=2 drafts), stages 1-3
    s1=6.3,  # one PP stage forward, 1-token chunk
    o4=1.5,  # last stage extra: lm_head + (rejection) sampling
    hop=0.6,  # stage -> stage hidden-state hand-off (host-staged NCCL)
    bc=0.6,  # last stage -> earlier stages sampled/draft broadcast
    d_k2=5.6,  # MTP propose, 2 drafts (draft prefill + 1 chained step)
    d_ext=1.35,  # each further chained MTP draft step
    # Host sync after verifying chunk A on the last stage. The CPU work that
    # follows it is exposed: B's prepare/launch (success) or propose(A)'s first
    # launch (fail). Today all of it is hidden behind the GPU (async).
    sync_ok=2.0,
    sync_fail=0.5,
    # Last-stage per-step occupancy that does not delay that step's outputs
    # (host work of the worker thread). 0 reproduces c1; 5.5 reproduces the
    # measured c4 aggregate (115 tok/s, TP2xPP4 + MTP k=2, before P2P) - matters only when stage 4 is busy.
    occ4=0.0,
)

# Conditional acceptance per draft position of one chained MTP propose:
# (c1, c2) measured in-engine (accept_eval, chat T=0, before P2P); c3 = the chain's
# guess for chunk A's bonus token; c4, c5 = chunk B's drafts.
# "*-offline": c3..c5 from the stock head's offline chained eval (s3 ~0.72-0.74,
# offline chained-head sweep) with a slow decay; "*-pess": a steep decay.
ACC = {
    "coding-offline": (0.81, 0.73, 0.72, 0.70, 0.68),
    "coding": (0.81, 0.73, 0.66, 0.62, 0.58),
    "coding-pess": (0.81, 0.73, 0.55, 0.50, 0.45),
    "browse": (0.84, 0.81, 0.70, 0.65, 0.60),
    "browse-pess": (0.84, 0.81, 0.58, 0.52, 0.47),
    "prose(harness)": (0.806, 0.68, 0.60, 0.55, 0.50),
}


def draw_chunk(rng, cs):
    """Accepted drafts of a 3-token chunk with conditional acceptances cs[0:2]."""
    if rng.random() >= cs[0]:
        return 0
    if rng.random() >= cs[1]:
        return 1
    return 2


def simulate(variant, acc, nstreams=1, epochs=50000, seed=1, p=None, mtp_penalty=0.0):
    """Return (per-stream tok/s averaged over streams, stats).

    variant: base | ps1 (PipeSpec, all 5 drafts before the broadcast) |
             ps2 (two-phase P1/P2 broadcast, skip propose(A) when B is valid) |
             ps2r (ps2, but always propose(A): exact MTP KV)
    nstreams: base 1..4; PipeSpec 1..2 (each stream takes 2 of the 4 ring steps).
    mtp_penalty: relative drop of c1, c2 in the epoch after a valid B (chunk A's
                 MTP KV left approximate when propose(A) is skipped; ps2 only).
    """
    p = dict(P, **(p or {}))
    rng = random.Random(seed)
    s3, o4, hop, bc, occ4 = p["s3"], p["o4"], p["hop"], p["bc"], p["occ4"]
    d2 = p["d_k2"]
    dext3 = 3 * p["d_ext"]
    ps = variant != "base"
    assert nstreams <= (2 if ps else 4)
    if not ps:
        layout = [(i, "N") for i in range(nstreams)]
    else:
        layout = [(i, kind) for i in range(nstreams) for kind in ("A", "B")]

    free = [0.0] * 5  # time each stage 1..4 becomes free
    p1_ready = [0.0] * nstreams  # stage 1 has the inputs of the next A/N chunk
    p2_ready = [0.0] * nstreams  # stage 1 has the inputs of the next B chunk
    p2_valid = [False] * nstreams
    penal = [False] * nstreams
    pending = [False] * nstreams  # chunk B of the current epoch is valid
    tokens = [0] * nstreams
    t_first = None
    st = dict(epochs=0, a_full=0, b_valid=0, b_tok=0)

    def finish_epoch(i, t4):
        """Propose from the epoch's last valid chunk; returns stage-4 busy end."""
        if variant == "ps1":
            t4 += d2 + dext3
            p1_ready[i] = p2_ready[i] = t4 + bc
        else:
            t4 += d2
            p1_ready[i] = t4 + bc
            t4 += dext3
            p2_ready[i] = t4 + bc
        p2_valid[i] = True
        return t4

    for _ in range(epochs):
        # Inputs of this ring cycle come from the previous cycle; stage-4 work
        # below only updates them for the next cycle.
        p1_dep, p2_dep = list(p1_ready), list(p2_ready)
        for i, kind in layout:
            # ---- stages 1..3 run every scheduled chunk (they cannot know)
            t = max(free[1], p1_dep[i] if kind in ("N", "A") else p2_dep[i])
            if t_first is None:
                t_first = t
            t += s3
            free[1] = t
            for s in (2, 3):
                t = max(free[s], t + hop) + s3
                free[s] = t
            arr4 = t + hop
            # ---- stage 4
            c = acc
            if penal[i]:
                f = 1 - mtp_penalty
                c = (acc[0] * f, acc[1] * f) + tuple(acc[2:])
            if kind == "N":
                t4 = max(free[4], arr4) + s3 + o4
                tokens[i] += 1 + draw_chunk(rng, c)
                t4 += d2
                free[4] = t4 + occ4
                p1_ready[i] = t4 + bc
                st["epochs"] += 1
            elif kind == "A":
                t4 = max(free[4], arr4) + s3 + o4
                a = draw_chunk(rng, c)
                st["epochs"] += 1
                st["a_full"] += a == 2
                tokens[i] += 1 + a
                pending[i] = a == 2 and p2_valid[i] and rng.random() < acc[2]
                penal[i] = False
                if pending[i]:
                    t4 += p["sync_ok"]
                    if variant == "ps2r":
                        t4 += d2  # propose(A) anyway (exact MTP KV)
                    free[4] = t4
                else:
                    t4 = finish_epoch(i, t4 + p["sync_fail"])
                    free[4] = t4 + occ4
            else:  # B
                if pending[i]:
                    t4 = max(free[4], arr4) + s3 + o4
                    b = draw_chunk(rng, acc[3:5])
                    tokens[i] += 1 + b
                    st["b_valid"] += 1
                    st["b_tok"] += 1 + b
                    free[4] = finish_epoch(i, t4) + occ4
                    penal[i] = variant == "ps2"
                # invalid B: stage 4 only receives its hidden states
    t_end = max(p1_ready)
    per_stream = [tk / (t_end - t_first) * 1000.0 for tk in tokens]
    return sum(per_stream) / len(per_stream), st


def rolling_stage1_drafter(pi, p=None):
    """Rolling pipeline, 1-token chunks, drafter on stage 1 (no MTP).

    Each chunk commits exactly one token; a wrong guess flushes the in-flight
    chunks and costs one pipeline round trip."""
    p = dict(P, **(p or {}))
    s1, o4, hop, bc = p["s1"], p["o4"], p["hop"], p["bc"]
    sync = p["sync_fail"]
    delta = s1 + o4 + sync  # the last stage is the slowest stage per chunk
    t_rt = 4 * s1 + o4 + 3 * hop + bc + sync
    return 1000.0 / (delta + (1 - pi) * (t_rt - delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60000)
    ep = ap.parse_args().epochs

    print("== single stream (c1), tok/s; sync_ok 2.0 / sync_fail 0.5 ms ==")
    print(f"{'acceptance set':16s} {'base':>5s} {'ps1':>5s} {'ps2':>5s} {'ps2r':>5s}"
          f" {'ps2 gain':>8s} {'ps2r gain':>9s}  A-full B-valid B-tok")
    for name, acc in ACC.items():
        r = {v: simulate(v, acc, 1, ep)[0] for v in ("base", "ps1", "ps2", "ps2r")}
        st = simulate("ps2", acc, 1, ep)[1]
        print(
            f"{name:16s} {r['base']:5.1f} {r['ps1']:5.1f} {r['ps2']:5.1f} {r['ps2r']:5.1f}"
            f" {r['ps2']/r['base']-1:+8.1%} {r['ps2r']/r['base']-1:+9.1%}"
            f"  {st['a_full']/st['epochs']:.3f}  {st['b_valid']/st['epochs']:.3f}"
            f"  {st['b_tok']/max(1, st['b_valid']):.2f}"
        )

    base = simulate("base", ACC["coding"], 1, ep)[0]
    print("\n== ps2 (coding) vs exposed host latency after the A-verify sync ==")
    for ok, fail in ((0.3, 0.3), (1.0, 0.5), (2.0, 0.5), (3.0, 1.0), (5.0, 1.5)):
        v = simulate("ps2", ACC["coding"], 1, ep, p=dict(sync_ok=ok, sync_fail=fail))[0]
        print(f"sync_ok {ok:3.1f} / sync_fail {fail:3.1f} ms: {v:5.1f} ({v/base-1:+.1%})")

    print("\n== ps2 (coding) vs MTP-KV approximation (c1, c2 drop after a valid B) ==")
    for pen in (0.0, 0.02, 0.05, 0.10):
        v = simulate("ps2", ACC["coding"], 1, ep, mtp_penalty=pen)[0]
        print(f"penalty {pen:4.0%}: {v:5.1f} ({v/base-1:+.1%})")

    print("\n== ps2 vs chained-draft acceptance (c1 .81, c2 .73, c3 = x, c4 = x-.04, c5 = x-.08) ==")
    for c3 in (0.40, 0.50, 0.60, 0.70, 0.80):
        acc = (0.81, 0.73, c3, c3 - 0.04, c3 - 0.08)
        v = simulate("ps2", acc, 1, ep)[0]
        print(f"c3 {c3:.2f}: {v:5.1f} ({v/base-1:+.1%})")

    print("\n== multi-stream, per-stream tok/s (occ4 = 5.5 ms calibrates base c4 to ~115 agg) ==")
    cal = dict(occ4=5.5)
    for n in (1, 2, 3, 4):
        b = simulate("base", ACC["prose(harness)"], n, ep // n, p=cal)[0]
        line = f"c{n} prose: base {b:5.1f}/stream ({n*b:5.1f} agg)"
        if n <= 2:
            v = simulate("ps2", ACC["prose(harness)"], n, ep // n, p=cal)[0]
            line += f", ps2 {v:5.1f}/stream ({v/b-1:+.1%})"
        print(line)

    print("\n== rolling pipeline, 1-token chunks, drafter on stage 1 (no MTP), tok/s ==")
    for pi in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"next-token hit rate {pi:.1f}: {rolling_stage1_drafter(pi):5.1f}")


if __name__ == "__main__":
    main()
