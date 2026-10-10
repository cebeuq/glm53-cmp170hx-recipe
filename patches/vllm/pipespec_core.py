# SPDX-License-Identifier: Apache-2.0
"""PipeSpec: pipelined speculative decoding across PP stages (one request).

Mounted at vllm/v1/cmp_pipespec.py. Inert unless CMP_PIPESPEC=1.
Design and correctness argument: docs/pipespec.md in the recipe repository.

One "epoch" of a lone decoding request R = two consecutive scheduler steps:
  A (step e)   : [last token, d1, d2]        verified as usual (MTP k=2 drafts)
  B (step e+1) : [d3, d4, d5]                 issued before A is verified; d3 is
                 the MTP chain's guess of A's bonus token, d4/d5 its next drafts.
B is valid iff A accepted d1 and d2 and A's bonus == d3. The last PP stage
decides after verifying A (host sync) and either verifies B or drops it.
Next epoch starts at step e+pp (normal ring cadence).

PP broadcast messages per step (last rank -> earlier ranks), by step marker:
  N (normal) : sampled, counts, drafts(k)                      (fork original)
  A          : sampled, counts
  B          : sampled, counts, P1 = drafts(k), P2 = drafts(k+1)
P1 feeds the next epoch's A, P2 the next epoch's B. For an invalid B the last
rank sends num_sampled=0, num_rejected=k+1, which rolls back B's optimistic
num_computed_tokens on the earlier ranks.
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("CMP_PIPESPEC", "0") == "1"
# Always run propose() after a valid A, so chunk A's MTP-layer KV is written
# from target hidden states (costs ~5.6 ms on the success path).
REFILL = os.environ.get("CMP_PIPESPEC_REFILL", "0") == "1"
# Keep this many tokens of headroom to max_tokens / max_model_len.
MIN_LEFT = int(os.environ.get("CMP_PIPESPEC_MIN_LEFT", "16") or 16)
# Log engine-side counters every N epochs (0 = never).
LOG_EVERY = int(os.environ.get("CMP_PIPESPEC_LOG", "500") or 0)

MARK_ATTR = "cmp_ps"  # SchedulerOutput attribute: dict[req_id, "A" | "B"]
SUPPORTED_K = 2


def num_p2(k: int) -> int:
    """Drafts beyond the first k that one propose must produce for chunk B."""
    return k + 1


def markers(scheduler_output) -> dict:
    return getattr(scheduler_output, MARK_ATTR, None) or {}


class SchedulerPipeSpec:
    """Scheduler (engine-core process) side: decides A/B steps, keeps counters."""

    def __init__(self, sched) -> None:
        self.k = sched.num_spec_tokens
        self.pp = sched.parallel_config.pipeline_parallel_size
        spec = sched.vllm_config.speculative_config
        self.enabled = (
            ENABLED
            and self.pp > 1
            and self.k == SUPPORTED_K
            and sched.use_v2_model_runner
            and sched.scheduler_config.async_scheduling
            and spec is not None
            and spec.method == "mtp"
            and getattr(spec, "draft_sample_method", "greedy") == "greedy"
        )
        if ENABLED and not self.enabled:
            logger.warning(
                "CMP_PIPESPEC=1 ignored: needs PP>1, V2 runner, async scheduling, "
                "MTP k=%d greedy drafts (got pp=%d k=%d)",
                SUPPORTED_K,
                self.pp,
                self.k,
            )
        # req_id -> step index of its pending chunk A
        self.epoch_a_step: dict[str, int] = {}
        self.stats = dict(epochs=0, a_full=0, b_sched=0, b_missed=0, b_valid=0,
                          b_invalid=0, b_tokens=0, a_tokens=0)
        if self.enabled:
            logger.info("PipeSpec enabled (k=%d, pp=%d, refill=%s)",
                        self.k, self.pp, REFILL)

    # ---- hooks called from Scheduler.schedule() ---------------------------------
    def lookahead(self, cur: int) -> int:
        # MTP chain of 2k+1 drafts writes MTP KV up to 2k+1 slots past the
        # last scheduled token.
        return max(cur, 2 * self.k + 1) if self.enabled else cur

    def is_b_step(self, request, current_step: int) -> bool:
        a = self.epoch_a_step.get(request.request_id)
        return a is not None and a + 1 == current_step

    def b_step_pending(self, current_step: int) -> bool:
        """True while this step is some request's chunk B: no new admissions."""
        return any(a + 1 == current_step for a in self.epoch_a_step.values())

    def b_token_ok(self, num_new_tokens: int) -> bool:
        return num_new_tokens == self.k + 1

    def after_schedule(self, sched, so) -> None:
        if not self.enabled:
            return
        cur = sched.current_step
        marks: dict[str, str] = {}
        for req_id, a_step in list(self.epoch_a_step.items()):
            if a_step + 1 > cur:
                continue
            del self.epoch_a_step[req_id]
            req = sched.requests.get(req_id)
            n = so.num_scheduled_tokens.get(req_id)
            if a_step + 1 == cur and n == self.k + 1 and len(
                so.scheduled_spec_decode_tokens.get(req_id, ())
            ) == self.k:
                marks[req_id] = "B"
                self.stats["b_sched"] += 1
            else:
                # A without B (abort/preempt/pause). Workers fall back safely as
                # long as R is not decoded again before step a_step + pp.
                assert n is None, f"PipeSpec: {req_id} scheduled off-cadence ({n})"
                self.stats["b_missed"] += 1
            if req is not None:
                req.next_decode_eligible_step = a_step + self.pp
        if marks:
            setattr(so, MARK_ATTR, marks)
            return
        if len(so.num_scheduled_tokens) != 1:
            return
        ((req_id, n),) = so.num_scheduled_tokens.items()
        req = sched.requests.get(req_id)
        if req is None or not self._eligible_a(sched, req, n, so):
            return
        self.epoch_a_step[req_id] = cur
        req.next_decode_eligible_step = cur + 1
        setattr(so, MARK_ATTR, {req_id: "A"})
        self.stats["epochs"] += 1

    def _eligible_a(self, sched, req, n, so) -> bool:
        if n != self.k + 1 or len(so.scheduled_spec_decode_tokens.get(req.request_id, ())) != self.k:
            return False
        if len(sched.running) != 1 or sched.waiting or sched.skipped_waiting:
            return False
        if getattr(sched, "num_waiting_for_streaming_input", 0):
            return False
        if req.is_prefill_chunk or req.use_structured_output or req.lora_request:
            return False
        sp = req.sampling_params
        if sp is None or sp.logprobs is not None or sp.prompt_logprobs is not None:
            return False
        # num_computed_tokens already includes chunk A.
        if req.num_prompt_tokens + req.max_tokens - req.num_computed_tokens < (
            self.k + 1 + MIN_LEFT
        ):
            return False
        if req.num_computed_tokens + (self.k + 1) + (2 * self.k + 1) + MIN_LEFT >= (
            sched.max_model_len
        ):
            return False
        if sched.kv_cache_manager.block_pool.get_num_free_blocks() < 4:
            return False
        return True

    # ---- hook called from Scheduler.update_from_output() ------------------------
    def on_output(self, req_id: str, mark: str | None, num_generated: int) -> None:
        if mark == "A":
            self.stats["a_tokens"] += num_generated
            self.stats["a_full"] += num_generated == self.k + 1
        elif mark == "B":
            if num_generated:
                self.stats["b_valid"] += 1
                self.stats["b_tokens"] += num_generated
            else:
                self.stats["b_invalid"] += 1
            done = self.stats["b_valid"] + self.stats["b_invalid"]
            if LOG_EVERY and done % LOG_EVERY == 0:
                s = self.stats
                ep = max(1, s["epochs"])
                logger.info(
                    "PipeSpec: epochs %d, A full %.3f, B valid %.3f (%.2f tok), "
                    "B missed %d, tokens/epoch %.2f",
                    s["epochs"], s["a_full"] / ep, s["b_valid"] / ep,
                    s["b_tokens"] / max(1, s["b_valid"]), s["b_missed"],
                    (s["a_tokens"] + s["b_tokens"]) / ep,
                )
