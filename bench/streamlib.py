#!/usr/bin/env python3
"""Shared parsing and timing helpers for OpenAI-compatible SSE streams.

The functions in this module do not open network connections.  Callers open
the HTTP response themselves and pass its iterable of SSE lines to
``measure_sse``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
import warnings as warnings_module
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


MULTIPLICITY_TOLERANCE = 0.005
DEFAULT_INITIAL_TOKENS_TO_DROP = 5


class MultiplicityMismatch(RuntimeError):
    """Raised when strict validation finds inconsistent token accounting."""


@dataclass(frozen=True)
class StreamEvent:
    """One parsed JSON event from an SSE ``data:`` line."""

    received_at: float
    payload: Mapping[str, Any]
    content: str | None = None
    reasoning: str | None = None
    usage: Mapping[str, Any] | None = None

    @property
    def has_token_delta(self) -> bool:
        """Return true even when a present delta is the empty string."""

        return self.content is not None or self.reasoning is not None


@dataclass
class StreamSummary:
    """Server-accounted result and client-side diagnostics for one stream."""

    completion_tokens: int | None
    content_chunks: int
    chunk_deficit: int | None
    timestamps: list[float]
    itls: list[float]
    ttft: float | None
    total_wall_time: float
    decode_tps: float | None
    ms_per_token: float | None
    multiplicity_reconstruction: int | None
    multiplicity_relative_error: float | None
    multiplicity_pass: bool | None
    warnings: list[str]
    usage: dict[str, Any] | None
    raw_chunks: int
    text: str
    text_content: str
    text_reasoning: str


def build_request(
    body: Mapping[str, Any] | None = None, **fields: Any
) -> dict[str, Any]:
    """Return a streaming request body with server usage accounting enabled.

    ``fields`` is provided for callers that prefer ``build_request(model=...)``.
    Explicit keyword fields replace fields from ``body``.
    """

    request = dict(body or {})
    request.update(fields)
    stream_options = request.get("stream_options")
    if stream_options is None:
        stream_options = {}
    if not isinstance(stream_options, Mapping):
        raise TypeError("stream_options must be a mapping")
    request["stream"] = True
    request["stream_options"] = {
        **stream_options,
        "include_usage": True,
    }
    return request


def cache_bust_tag(prefix: str = "cache-bust") -> str:
    """Return a unique UUID-based tag that defeats prompt-prefix caching."""

    return f"{prefix}-{uuid.uuid4().hex}"


def cache_bust_prefix(prompt: str, prefix: str = "cache-bust") -> str:
    """Prefix ``prompt`` with a unique cache-busting tag."""

    return f"[{cache_bust_tag(prefix)}] {prompt}"


def _decode_sse_line(raw_line: bytes | str) -> str | None:
    if isinstance(raw_line, bytes):
        line = raw_line.decode("utf-8", errors="replace")
    else:
        line = raw_line
    line = line.rstrip("\r\n")
    if not line.startswith("data:"):
        return None
    return line[5:].lstrip()


def _event_fields(
    payload: Mapping[str, Any],
) -> tuple[str | None, str | None, Mapping[str, Any] | None]:
    content = None
    reasoning = None
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        choice = choices[0]
        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            # Do not use truthiness here.  An empty string is a received delta.
            if delta.get("content") is not None:
                content = delta.get("content")
            if delta.get("reasoning") is not None:
                reasoning = delta.get("reasoning")
            elif delta.get("reasoning_content") is not None:
                reasoning = delta.get("reasoning_content")
        # /v1/completions streams text directly on the choice.
        elif choice.get("text") is not None:
            content = choice.get("text")

    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        usage = None
    return content, reasoning, usage


def iter_sse_events(
    lines: Iterable[bytes | str],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> Iterator[StreamEvent]:
    """Yield parsed events from raw SSE lines, stopping at ``[DONE]``."""

    for raw_line in lines:
        received_at = clock()
        data = _decode_sse_line(raw_line)
        if data is None:
            continue
        if data == "[DONE]":
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            warnings_module.warn(
                "ignored malformed JSON in SSE data line",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not isinstance(payload, Mapping):
            warnings_module.warn(
                "ignored non-object JSON in SSE data line",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        content, reasoning, usage = _event_fields(payload)
        yield StreamEvent(
            received_at=received_at,
            payload=payload,
            content=content,
            reasoning=reasoning,
            usage=usage,
        )


def inter_token_intervals(
    timestamps: Sequence[float], *, drop_first_tokens: int = 0
) -> list[float]:
    """Return adjacent token/chunk intervals in seconds.

    ``drop_first_tokens=5`` implements the within-run part of the current
    warmup policy.  It drops five timestamps before constructing intervals;
    it does not implement the old, retracted 70-token discard rule.
    """

    if drop_first_tokens < 0:
        raise ValueError("drop_first_tokens must be non-negative")
    kept = timestamps[drop_first_tokens:]
    return [later - earlier for earlier, later in zip(kept, kept[1:])]


def itl_median(itls: Sequence[float]) -> float | None:
    """Return the median inter-token latency in seconds (also the p50)."""

    return statistics.median(itls) if itls else None


def itl_p50(itls: Sequence[float]) -> float | None:
    """Return the p50 inter-token latency in seconds."""

    return itl_median(itls)


def itl_p99(itls: Sequence[float]) -> float | None:
    """Return linearly interpolated p99 inter-token latency in seconds."""

    if not itls:
        return None
    ordered = sorted(itls)
    position = (len(ordered) - 1) * 0.99
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def discard_campaign_warmup(
    timestamp_reps: Sequence[Sequence[float]],
    *,
    drop_first_tokens: int = DEFAULT_INITIAL_TOKENS_TO_DROP,
) -> list[list[float]]:
    """Drop rep 0 and the first five token stamps from every kept run."""

    if drop_first_tokens < 0:
        raise ValueError("drop_first_tokens must be non-negative")
    return [list(stamps[drop_first_tokens:]) for stamps in timestamp_reps[1:]]


def _multiplicity_reconstruction(
    itls: Sequence[float],
) -> tuple[int | None, float | None]:
    median_itl = itl_median(itls)
    if median_itl is None or median_itl <= 0:
        return None, median_itl
    reconstructed = sum(
        max(1, round(itl / median_itl))
        for itl in itls
    )
    return reconstructed, median_itl


def measure_sse(
    lines: Iterable[bytes | str],
    *,
    started_at: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
    strict_multiplicity: bool = False,
    emit_warnings: bool = True,
) -> StreamSummary:
    """Parse and measure one already-open SSE response.

    The primary token count always comes from ``usage.completion_tokens``.
    Client-observed chunks are retained only as an accounting cross-check.
    Set ``strict_multiplicity`` to raise instead of only warning when the
    reconstructed count differs from server accounting by more than 0.5%.
    """

    if started_at is None:
        started_at = clock()

    timestamps: list[float] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    ordered_text_parts: list[str] = []
    usage: dict[str, Any] | None = None
    raw_chunks = 0

    for event in iter_sse_events(lines, clock=clock):
        raw_chunks += 1
        if event.usage is not None:
            usage = dict(event.usage)
        if not event.has_token_delta:
            continue
        timestamps.append(event.received_at)
        if event.reasoning is not None:
            reasoning_parts.append(event.reasoning)
            ordered_text_parts.append(event.reasoning)
        if event.content is not None:
            content_parts.append(event.content)
            ordered_text_parts.append(event.content)

    ended_at = clock()
    result_warnings: list[str] = []

    def add_warning(message: str) -> None:
        result_warnings.append(message)
        if emit_warnings:
            warnings_module.warn(message, RuntimeWarning, stacklevel=2)

    completion_tokens: int | None = None
    if usage is None:
        add_warning(
            "stream ended without a usage object; include_usage ground truth is missing"
        )
    else:
        reported_tokens = usage.get("completion_tokens")
        if isinstance(reported_tokens, int) and not isinstance(reported_tokens, bool):
            completion_tokens = reported_tokens
        else:
            add_warning("usage.completion_tokens is missing or is not an integer")

    content_chunks = len(timestamps)
    chunk_deficit = (
        completion_tokens - content_chunks
        if completion_tokens is not None
        else None
    )
    if chunk_deficit:
        add_warning(
            "server/client token accounting differs: "
            f"chunk_deficit={chunk_deficit} "
            f"(completion_tokens={completion_tokens}, content_chunks={content_chunks})"
        )

    itls = inter_token_intervals(timestamps)
    reconstruction, median_itl = _multiplicity_reconstruction(itls)
    multiplicity_relative_error = None
    multiplicity_pass = None
    if completion_tokens is not None and reconstruction is not None:
        denominator = max(1, completion_tokens)
        multiplicity_relative_error = abs(
            reconstruction - completion_tokens
        ) / denominator
        multiplicity_pass = multiplicity_relative_error <= MULTIPLICITY_TOLERANCE
        if not multiplicity_pass:
            message = (
                "multiplicity reconstruction mismatch: "
                f"reconstructed={reconstruction}, "
                f"completion_tokens={completion_tokens}, "
                f"relative_error={multiplicity_relative_error:.3%}, "
                f"median_itl={median_itl!r}"
            )
            add_warning(message)
            if strict_multiplicity:
                raise MultiplicityMismatch(message)
    elif completion_tokens is not None:
        add_warning(
            "multiplicity reconstruction unavailable: at least two positive-timed "
            "content chunks are required"
        )

    ttft = timestamps[0] - started_at if timestamps else None
    total_wall_time = ended_at - started_at
    decode_tps = None
    ms_per_token = None
    if completion_tokens is not None and completion_tokens > 1 and len(timestamps) > 1:
        decode_span = timestamps[-1] - timestamps[0]
        if decode_span > 0:
            tokens_between_first_and_last = completion_tokens - 1
            decode_tps = tokens_between_first_and_last / decode_span
            ms_per_token = 1000.0 * decode_span / tokens_between_first_and_last
        else:
            add_warning("decode span is not positive; token rate is unavailable")

    return StreamSummary(
        completion_tokens=completion_tokens,
        content_chunks=content_chunks,
        chunk_deficit=chunk_deficit,
        timestamps=timestamps,
        itls=itls,
        ttft=ttft,
        total_wall_time=total_wall_time,
        decode_tps=decode_tps,
        ms_per_token=ms_per_token,
        multiplicity_reconstruction=reconstruction,
        multiplicity_relative_error=multiplicity_relative_error,
        multiplicity_pass=multiplicity_pass,
        warnings=result_warnings,
        usage=usage,
        raw_chunks=raw_chunks,
        text="".join(ordered_text_parts),
        text_content="".join(content_parts),
        text_reasoning="".join(reasoning_parts),
    )


def _selftest() -> None:
    transcript = [
        'data: {"choices":[{"delta":{"content":"alpha"}}],"usage":null}',
        'data: {"choices":[{"delta":{"content":""}}],"usage":null}',
        'data: {"choices":[{"delta":{}}],"usage":null}',
        'data: {"choices":[{"delta":{"reasoning":"beta"}}],"usage":null}',
        'data: {"choices":[{"delta":{"content":"gamma"}}],"usage":null}',
        'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":5}}',
        "data: [DONE]",
    ]
    clock_values = iter([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    result = measure_sse(
        transcript,
        started_at=0.0,
        clock=lambda: next(clock_values),
        emit_warnings=False,
    )

    assert result.content_chunks == 4, (
        "empty-string delta was not counted, or missing delta was counted"
    )
    assert result.timestamps == [0.01, 0.02, 0.04, 0.05]
    assert result.completion_tokens == 5, "server usage was not authoritative"
    assert result.completion_tokens != result.content_chunks
    assert result.chunk_deficit == 1
    assert result.multiplicity_reconstruction is not None
    assert result.multiplicity_pass is not None
    assert result.warnings, "cross-check warnings were not recorded"
    request = build_request(model="model", prompt="prompt")
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}

    completion_transcript = [
        'data: {"choices":[{"text":""}],"usage":null}',
        'data: {"choices":[{"text":"delta"}],"usage":null}',
        'data: {"choices":[],"usage":{"completion_tokens":2}}',
        "data: [DONE]",
    ]
    completion_clock = iter([0.1, 0.2, 0.3, 0.4, 0.5])
    completion_result = measure_sse(
        completion_transcript,
        started_at=0.0,
        clock=lambda: next(completion_clock),
        emit_warnings=False,
    )
    assert completion_result.content_chunks == 2
    assert completion_result.completion_tokens == 2
    assert completion_result.text == "delta"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the embedded offline SSE parser test",
    )
    args = parser.parse_args()
    if not args.selftest:
        parser.error("no action selected; use --selftest")
    try:
        _selftest()
    except Exception as exc:  # A CLI self-test must convert any failure to status 1.
        print(f"streamlib selftest: FAIL: {exc}")
        return 1
    print("streamlib selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
