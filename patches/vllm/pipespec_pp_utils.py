# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner.

PipeSpec (CMP_PIPESPEC=1, vllm/v1/cmp_pipespec.py): per-step message
formats N/A/B, early consumption of a chunk-B slot together with its chunk-A
slot, and the P2 draft stash that feeds the next chunk B on the first rank.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]
    # PipeSpec: message format of this step ("N", "A" or "B"); for "B",
    # draft_tokens holds P1 and p2_tokens/event2 the chunk-B drafts (k+1).
    fmt: str = "N"
    p2_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps + 1]
    event2: torch.cuda.Event | None = None


# PipeSpec: queue entry of a chunk-B slot already consumed with its A slot.
_CONSUMED = object()


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )
        self.aux_hidden_state_relay_keys: tuple[str, ...] = ()
        # PipeSpec: req_idx -> (p2 [k+1] tensor, event); set while
        # consuming a chunk-B slot, valid only for the step right after.
        self.p2_stash: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}
        self.p2_stash_prev: dict[int, tuple[torch.Tensor, torch.cuda.Event]] = {}

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def configure_aux_hidden_state_relay(self, model: torch.nn.Module) -> None:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            aux_hidden_state_relay_keys,
        )

        self.aux_hidden_state_relay_keys = aux_hidden_state_relay_keys(model)

    def relay_aux_hidden_states(
        self,
        intermediate_tensors: IntermediateTensors | None,
        output_intermediate_tensors: IntermediateTensors,
    ) -> IntermediateTensors:
        if not self.aux_hidden_state_relay_keys:
            return output_intermediate_tensors
        assert intermediate_tensors is not None
        return IntermediateTensors(
            output_intermediate_tensors.tensors
            | {
                key: intermediate_tensors[key]
                for key in self.aux_hidden_state_relay_keys
            }
        )

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.

        PipeSpec: if that entry is a chunk-A slot and the next entry is
        its chunk-B slot, the B slot is claimed too; the caller must then call
        `take_chunk_b_outputs` and post-process it right after this one.
        """
        # P2 drafts stashed during the previous step are usable in this step only.
        self.p2_stash_prev = self.p2_stash
        self.p2_stash = {}
        self._chunk_b_slot = None
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None or slot is _CONSUMED:
            return None
        if (
            slot.fmt == "A"
            and self.queue
            and isinstance(self.queue[0], PendingRecv)
            and self.queue[0].fmt == "B"
            and np.array_equal(self.queue[0].idx_mapping_np, slot.idx_mapping_np)
        ):
            self._chunk_b_slot = self.queue[0]
            self.queue[0] = _CONSUMED
        return self._consume_slot(slot, draft_tokens_to_update)

    def take_chunk_b_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """PipeSpec: outputs of the chunk-B slot claimed by the last
        `get_prev_sampled_outputs` (P1 drafts -> draft_tokens_to_update), and
        stash its P2 drafts for the chunk B of the current epoch."""
        slot = self._chunk_b_slot
        self._chunk_b_slot = None
        if slot is None:
            return None
        out = self._consume_slot(slot, draft_tokens_to_update)
        if out is not None and slot.p2_tokens is not None:
            freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
            for row, req_idx in enumerate(slot.idx_mapping_np):
                if not freed[row] and slot.need_sampled_mask[row]:
                    assert slot.event2 is not None
                    self.p2_stash[int(req_idx)] = (slot.p2_tokens[row], slot.event2)
        return out

    def take_p2_for_chunk_b(self, req_idx: int) -> torch.Tensor | None:
        """PipeSpec: P2 drafts for request `req_idx`'s chunk B in this
        step, after making the main stream wait for their arrival."""
        entry = self.p2_stash_prev.pop(req_idx, None)
        if entry is None:
            return None
        p2, event = entry
        self.main_stream.wait_event(event)
        return p2

    def _consume_slot(
        self, slot: PendingRecv, draft_tokens_to_update: torch.Tensor | None
    ) -> dict[str, torch.Tensor] | None:
        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        self.main_stream.wait_event(slot.event)
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(
                    slot.idx_mapping_np[keep], device=self.device
                )
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def broadcast_rows(self, rows: torch.Tensor, ready: torch.cuda.Event) -> None:
        """PipeSpec: broadcast `rows` ([num_reqs, n] int64, produced on the
        main stream before `ready`) without waiting for later main-stream work."""
        assert self.is_last_rank
        assert rows.dtype == torch.int64
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_event(ready)
            rows.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                rows, src=self.last_rank, group=self.broadcast_group
            )

    def broadcast_dropped_chunk(self, num_reqs: int) -> None:
        """PipeSpec: sampled/counts message of a dropped chunk B
        (num_sampled 0, num_rejected k+1). Depends on no GPU work."""
        assert self.is_last_rank
        with torch.cuda.stream(self.broadcast_stream):
            send_tokens = torch.zeros(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.zeros(2, num_reqs, dtype=torch.int32, device=self.device)
            combined[1].fill_(self.max_sample_len)
            torch.distributed.broadcast(
                send_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )

    def receive(self, input_batch: InputBatch, fmt: str = "N") -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch.

        PipeSpec: fmt "A" receives no drafts; fmt "B" receives P1 (k
        drafts, covered by `event`) and P2 (k+1 drafts, covered by `event2`)."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if self.num_speculative_steps > 0 and fmt != "A":
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            p2_tokens = None
            event2 = None
            if fmt == "B":
                p2_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps + 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    p2_tokens, src=self.last_rank, group=self.broadcast_group
                )
                event2 = self.broadcast_stream.record_event()
                p2_tokens.record_stream(self.main_stream)
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
            fmt,
            p2_tokens,
            event2,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
