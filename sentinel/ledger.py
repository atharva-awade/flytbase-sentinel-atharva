"""Timing ledger → honest runtime_metadata (PLAN.md §6.4).

    ledger = Ledger()
    with ledger.video():                      # end-to-end clock for one video
        with ledger.call("siglip2-b16"): ...  # every model call
    md = ledger.runtime_metadata(frames_processed=600, chunks_processed=24)
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from statistics import median

from .schema import ModelRuntime, RuntimeMetadata


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class Ledger:
    def __init__(self) -> None:
        self.calls: dict[str, list[float]] = {}
        self._video_start: float | None = None
        self.video_elapsed_ms: float = 0.0

    @contextmanager
    def video(self):
        self._video_start = time.perf_counter()
        try:
            yield self
        finally:
            self.video_elapsed_ms = (time.perf_counter() - self._video_start) * 1000.0

    @contextmanager
    def call(self, model_name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.calls.setdefault(model_name, []).append((time.perf_counter() - t0) * 1000.0)

    def record(self, model_name: str, elapsed_ms: float) -> None:
        """For calls timed elsewhere (e.g. server-reported latency)."""
        self.calls.setdefault(model_name, []).append(float(elapsed_ms))

    def model_runtimes(self, include_call_times: bool = False) -> list[ModelRuntime]:
        out = []
        for name, xs in self.calls.items():
            n = len(xs)
            total_r = round(float(sum(xs)), 4)          # average is derived from the *reported* total,
            avg_r = round(total_r / n, 6) if n else 0.0  # so total/count always matches within 2%
            out.append(ModelRuntime(
                model_name=name, call_count=n, total_time_ms=total_r,
                average_time_ms=avg_r,
                p50_time_ms=round(median(xs), 3) if xs else 0.0,
                p95_time_ms=round(_pct(xs, 0.95), 3),
                max_time_ms=round(max(xs), 3) if xs else 0.0,
                call_times_ms=[round(x, 3) for x in xs] if include_call_times else None,
            ))
        return out

    def runtime_metadata(self, frames_processed: int, chunks_processed: int,
                         end_to_end_ms: float | None = None) -> RuntimeMetadata:
        return RuntimeMetadata(
            frames_processed=int(frames_processed),
            chunks_processed=int(chunks_processed),
            end_to_end_internal_time_ms=round(end_to_end_ms if end_to_end_ms is not None else self.video_elapsed_ms, 3),
            model_runtimes=self.model_runtimes(),
        )
