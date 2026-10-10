#!/usr/bin/env python3
"""Offline (CPU, stdlib) checks for the PipeSpec patches.

  docker compose -f pipespec/compose.yaml run --rm check

1. py_compile every patches/vllm/pipespec_*.py.
2. Scheduler policy: the real pipespec_core.SchedulerPipeSpec driven by a stub
   scheduler that mimics the async V2+PP cadence (marks A,B,-,- ; gating off
   with a second request ; A without B on pause ; the B-step guard).
3. Protocol simulation: a token-level mirror of the worker logic (PP ring with
   N/A/B messages, early chunk-B consumption, P2 stash, last-rank decision,
   dropped-B rollback, A-without-B restore, scheduler accounting with async
   lag) on a toy deterministic "target model". Checks: committed tokens equal
   greedy decoding; first and last stage agree on every chunk's positions and
   input tokens; no stage reads KV that is not the committed context; the
   scheduler's 1+k tokens-per-step invariant.
The simulation checks the design; only the engine run checks the CUDA code.
"""

import collections
import importlib.util
import os
import py_compile
import random
import sys
import types
import zlib

PORTED = os.environ.get("PORTED", "/ported")


# ---------------------------------------------------------------- 1. compile
def check_compile():
    for f in sorted(os.listdir(PORTED)):
        if f.startswith("pipespec_") and f.endswith(".py"):
            py_compile.compile(os.path.join(PORTED, f), cfile="/tmp/x.pyc", doraise=True)
            print("compile ok", f)


# ------------------------------------------------------ 2. scheduler policy
def load_core():
    vllm = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")

    class _L:
        def info(self, *a, **k):
            pass

        warning = info

    logger_mod.init_logger = lambda name: _L()
    sys.modules["vllm"] = vllm
    sys.modules["vllm.logger"] = logger_mod
    os.environ["CMP_PIPESPEC"] = "1"
    spec = importlib.util.spec_from_file_location(
        "cmp_pipespec", os.path.join(PORTED, "pipespec_core.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NS(types.SimpleNamespace):
    pass


class StubSched:
    """Just enough of Scheduler/AsyncScheduler for SchedulerPipeSpec."""

    def __init__(self, core, pp=4, k=2):
        self.num_spec_tokens = k
        self.parallel_config = NS(pipeline_parallel_size=pp)
        self.vllm_config = NS(speculative_config=NS(method="mtp", draft_sample_method="greedy"))
        self.use_v2_model_runner = True
        self.scheduler_config = NS(async_scheduling=True)
        self.running, self.waiting, self.skipped_waiting = [], [], []
        self.requests = {}
        self.current_step = 0
        self.max_model_len = 131072
        self.kv_cache_manager = NS(block_pool=NS(get_num_free_blocks=lambda: 1000))
        self.ps = core.SchedulerPipeSpec(self)
        self.pp, self.k = pp, k
        self.paused = False

    def add(self, rid):
        r = NS(request_id=rid, next_decode_eligible_step=0, is_prefill_chunk=False,
               use_structured_output=False, lora_request=None,
               sampling_params=NS(logprobs=None, prompt_logprobs=None),
               num_prompt_tokens=100, max_tokens=4000, num_computed_tokens=100)
        self.requests[rid] = r
        self.running.append(r)
        return r

    def schedule(self):
        self.current_step += 1
        so = NS(num_scheduled_tokens={}, scheduled_spec_decode_tokens={})
        if not self.paused:
            for r in self.running:
                if self.current_step < r.next_decode_eligible_step:
                    continue
                n = self.k + 1
                if self.ps.is_b_step(r, self.current_step) and not self.ps.b_token_ok(n):
                    continue
                so.num_scheduled_tokens[r.request_id] = n
                so.scheduled_spec_decode_tokens[r.request_id] = [-1] * self.k
        for rid, n in so.num_scheduled_tokens.items():  # AsyncScheduler cadence
            r = self.requests[rid]
            r.num_computed_tokens += n
            r.next_decode_eligible_step = self.current_step + self.pp
        self.ps.after_schedule(self, so)
        return so


def check_scheduler(core):
    s = StubSched(core)
    s.add("r1")
    marks = []
    for _ in range(16):
        so = s.schedule()
        m = getattr(so, core.MARK_ATTR, {})
        marks.append(m.get("r1", "N") if "r1" in so.num_scheduled_tokens else "-")
    assert "".join(marks) == "AB--" * 4, marks
    # A second request turns PipeSpec off (both decode as N in the ring).
    s.add("r2")
    seen = []
    for _ in range(8):
        so = s.schedule()
        seen.append(dict(getattr(so, core.MARK_ATTR, {})))
    assert all(not m for m in seen), seen
    s.running.pop()  # r2 finished
    s.requests.pop("r2")
    # Pause between A and B: A without B, then normal cadence resumes.
    while True:
        so = s.schedule()
        if getattr(so, core.MARK_ATTR, {}).get("r1") == "A":
            a_step = s.current_step
            break
    s.paused = True
    so = s.schedule()
    assert not so.num_scheduled_tokens and s.ps.stats["b_missed"] == 1
    s.paused = False
    r1 = s.requests["r1"]
    assert r1.next_decode_eligible_step == a_step + s.pp, (r1.next_decode_eligible_step, a_step)
    out = []
    for _ in range(8):
        so = s.schedule()
        out.append(getattr(so, core.MARK_ATTR, {}).get("r1", "N") if so.num_scheduled_tokens else "-")
    assert "".join(out) == "--AB--AB", out
    # Accounting hooks.
    s.ps.on_output("r1", "A", 3)
    s.ps.on_output("r1", "B", 0)
    assert s.ps.stats["a_full"] >= 1 and s.ps.stats["b_invalid"] == 1
    print("scheduler policy ok:", "".join(marks), s.ps.stats)


# ------------------------------------------------------ 3. protocol model
V = 997  # toy vocabulary
GARBAGE = -7


def target(ctx):
    return zlib.crc32(repr(tuple(ctx)).encode()) % V


def drafter(ctx, alpha):
    t = target(ctx)
    u = (zlib.crc32(b"d" + repr(tuple(ctx)).encode()) % 10000) / 10000.0
    return t if u < alpha else (t + 1 + int(u * 7)) % V


Msg = collections.namedtuple("Msg", "fmt step sampled num_sampled num_rejected p1 p2")


def simulate(seed, alpha, pp=4, k=2, epochs=1500, pause_p=0.0, second_req_p=0.0):
    rng = random.Random(seed)
    prompt = [rng.randrange(V) for _ in range(6)]
    oracle = list(prompt)
    while len(oracle) < len(prompt) + epochs * 2 * (k + 1) + 64:
        oracle.append(target(oracle))

    def chain(ctx, n):
        out, c = [], list(ctx)
        for _ in range(n):
            d = drafter(c, alpha)
            out.append(d)
            c.append(d)
        return out

    # committed tokens as the engine reports them
    committed = list(prompt)
    kv1, kv4 = {}, {}  # first-stage (stands for stages 1..P-1) and last-stage KV
    for j, t in enumerate(prompt):
        kv1[j] = kv4[j] = t
    # prefill step: sample first token, propose k drafts (N message, no P2)
    first = target(prompt)
    committed.append(first)
    p1 = chain(committed, k)
    s1 = dict(nc=len(prompt), last=first, drafts=list(p1))
    s4 = dict(nc=len(prompt), last=first, drafts=list(p1), p2=None, expect_b=False,
              b_valid=False, a_drafts=None, pending=None)
    ring = collections.deque([None] * pp)
    ring[-1] = Msg("N", 0, [first], 1, 0, p1, None)
    stash = {}
    # scheduler accounting (async: outputs applied with a lag)
    sch = dict(num_tokens=len(committed), nc=len(prompt), ph=0)
    out_q = collections.deque()
    stats = collections.Counter()
    CONSUMED = object()

    step, a_step = 0, None
    next_ok = pp  # next step R may decode
    while len(committed) < len(oracle) - 2 * (k + 1) - 8 and stats["epochs"] < epochs:
        step += 1
        # ---------------- scheduler: decide marker for R at this step
        mark = None
        if a_step is not None and step == a_step + 1:
            if rng.random() < pause_p:
                stats["paused_b"] += 1  # A without B
            else:
                mark = "B"
            a_step = None
            next_ok = step - 1 + pp
        elif step >= next_ok:
            other = rng.random() < second_req_p  # PipeSpec gated off this epoch
            mark = "N" if other else "A"
            if mark == "A":
                a_step = step
                stats["epochs"] += 1
            next_ok = step + 1 if mark == "A" else step + pp
        if mark is not None:
            n_new = sch["num_tokens"] + k + sch["ph"] - sch["nc"]
            assert n_new == k + 1, ("scheduler invariant", n_new)
            sch["nc"] += k + 1
            sch["ph"] += k + 1

        # ---------------- first stage: execute_model(step)
        slot = ring.popleft()
        ring.append(None)
        stash_prev, stash = stash, {}
        chunk_b = None
        if isinstance(slot, Msg):
            if slot.fmt == "A" and isinstance(ring[0], Msg) and ring[0].fmt == "B":
                chunk_b, ring[0] = ring[0], CONSUMED
            for m in (slot, chunk_b):
                if m is None:
                    continue
                if m.num_sampled > 0:
                    s1["last"] = m.sampled[m.num_sampled - 1]
                s1["nc"] -= m.num_rejected
                if m.p1 is not None:
                    s1["drafts"] = list(m.p1)
                if m.fmt == "B":
                    stash["R"] = list(m.p2)
        rec = None
        if mark is not None:
            q = s1["nc"]
            inputs = [s1["last"]] + s1["drafts"]
            if mark == "B" and "R" in stash_prev:
                inputs = list(stash_prev["R"])
            ctx_ok1 = all(kv1.get(j) == oracle[j] for j in range(q))
            for i, t in enumerate(inputs):
                kv1[q + i] = t
            rec = (q, inputs, ctx_ok1)
            s1["nc"] += k + 1  # optimistic (sample_tokens)

        # ---------------- last stage: execute_model + sample_tokens(step)
        msg = None
        if mark is not None:
            skip = False
            if s4["expect_b"] and mark != "B":  # A without B
                s4["drafts"] = list(s4["a_drafts"])
                s4.update(expect_b=False, b_valid=False, pending=None, p2=None)
                stats["restored"] += 1
            if mark == "A":
                s4["a_drafts"] = list(s4["drafts"])
            elif mark == "B":
                assert s4["expect_b"]
                skip = not s4["b_valid"]
            elif mark == "N":
                s4["p2"] = None
            if skip:
                p1, p2 = s4["pending"]
                s4.update(pending=None, expect_b=False)
                msg = Msg("B", step, [0] * (k + 1), 0, k + 1, p1, p2)
                gen = []
                stats["b_dropped"] += 1
            else:
                q4 = s4["nc"]
                inputs4 = [s4["last"]] + s4["drafts"]
                q, inputs1, ctx_ok1 = rec
                assert q4 == q, ("position mismatch", step, mark, q4, q)
                assert inputs4 == inputs1, ("input mismatch", step, mark, inputs4, inputs1)
                ctx_ok4 = all(kv4.get(j) == oracle[j] for j in range(q))
                for i, t in enumerate(inputs4):
                    kv4[q + i] = t
                outs = []
                for i in range(k + 1):
                    ctx = oracle[:q] + inputs4[: i + 1]
                    outs.append(target(ctx) if (ctx_ok1 and ctx_ok4) else GARBAGE)
                acc = 0
                while acc < k and outs[acc] == inputs4[acc + 1]:
                    acc += 1
                sampled = inputs4[1 : 1 + acc] + [outs[acc]]
                n_s = acc + 1
                n_r = k + 1 - n_s
                s4["last"] = sampled[-1]
                s4["nc"] += k + 1 - n_r
                gen = list(sampled)
                committed.extend(sampled)
                ctx_after = list(committed)
                if mark == "N":
                    p1 = chain(ctx_after, k)
                    s4["drafts"] = p1
                    msg = Msg("N", step, sampled, n_s, n_r, p1, None)
                elif mark == "A":
                    p2 = s4["p2"]
                    s4["p2"] = None
                    s4["expect_b"] = True
                    valid = p2 is not None and n_r == 0 and sampled[k] == p2[0]
                    s4["b_valid"] = valid
                    msg = Msg("A", step, sampled, n_s, n_r, None, None)
                    if valid:
                        s4["drafts"] = list(p2[1:])
                        stats["b_armed"] += 1
                    else:
                        d = chain(ctx_after, 2 * k + 1)
                        s4["drafts"] = d[:k]
                        s4["pending"] = (d[:k], d[k:])
                        s4["p2"] = d[k:]
                else:  # valid B
                    d = chain(ctx_after, 2 * k + 1)
                    s4["drafts"] = d[:k]
                    s4["p2"] = d[k:]
                    s4["expect_b"] = False
                    msg = Msg("B", step, sampled, n_s, n_r, d[:k], d[k:])
                    stats["b_valid"] += 1
                    stats["b_tokens"] += n_s
            # first stage receives this step's message (fmt fixed by marker)
            assert msg.fmt == ("N" if mark == "N" else mark)
            ring[-1] = msg
            out_q.append((mark, gen))
        # scheduler processes outputs with a lag of pp+1 steps
        while len(out_q) > pp + 1 or (out_q and stats["epochs"] >= epochs):
            m, g = out_q.popleft()
            if g:
                rej = k + 1 - len(g)
                sch["nc"] -= rej
                sch["ph"] -= rej + len(g)
                sch["num_tokens"] += len(g)
            elif m == "B":
                sch["nc"] -= k + 1
                sch["ph"] -= k + 1
            assert sch["ph"] >= 0
            if stats["epochs"] >= epochs and not out_q:
                break
    n = len(committed)
    assert committed == oracle[:n], next(
        (i, committed[i], oracle[i]) for i in range(n) if committed[i] != oracle[i])
    stats["tokens"] = n - len(prompt)
    return stats


def check_protocol():
    for seed, alpha, pause, other in ((1, 0.8, 0.0, 0.0), (2, 0.6, 0.0, 0.0),
                                      (3, 0.9, 0.05, 0.0), (4, 0.75, 0.05, 0.2),
                                      (5, 0.3, 0.1, 0.1), (6, 0.95, 0.0, 0.0)):
        st = simulate(seed, alpha, pause_p=pause, second_req_p=other)
        ep = max(1, st["epochs"])
        print(f"protocol ok: seed {seed} alpha {alpha:.2f} pause {pause:.2f} "
              f"N-epochs {other:.2f}: {st['tokens']} tokens, {st['epochs']} PipeSpec epochs, "
              f"B valid {st['b_valid'] / ep:.2f}, dropped {st['b_dropped'] / ep:.2f}, "
              f"A-without-B {st['paused_b']} (restored {st['restored']})")


if __name__ == "__main__":
    check_compile()
    check_scheduler(load_core())
    check_protocol()
    print("ALL CHECKS PASSED")
