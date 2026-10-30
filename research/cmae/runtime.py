"""Synchronous audit/commit boundaries and small runtime measurements."""

from __future__ import annotations

import math
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from research.cmae.engine import Decision

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class GateResult(Generic[R]):
    decision: Decision
    elapsed_seconds: float
    committed: bool
    payload: R | None = None


class SynchronousBudgetGate:
    """Run an audit before commit and fail closed on timeout or unhealthy evidence."""

    def __init__(
        self, budget_seconds: float, *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        if not math.isfinite(budget_seconds) or budget_seconds < 0:
            raise ValueError("budget must be finite and nonnegative")
        self.budget_seconds = budget_seconds
        self.clock = clock

    def run(
        self,
        audit: Callable[[], T],
        commit: Callable[[T], R],
        *,
        healthy: bool = True,
    ) -> GateResult[R]:
        started = self.clock()
        audited = audit()
        elapsed = self.clock() - started
        if not healthy:
            return GateResult(Decision.ABSTAIN, elapsed, False)
        if not math.isfinite(elapsed) or elapsed > self.budget_seconds:
            return GateResult(Decision.STOP, elapsed, False)
        committed = commit(audited)
        return GateResult(Decision.TRAIN, elapsed, True, committed)


@dataclass(frozen=True)
class OperationMeasurement:
    elapsed_seconds: float
    peak_tracemalloc_bytes: int


def measure_operation(operation: Callable[[], T]) -> tuple[T, OperationMeasurement]:
    """Measure one synchronous operation; memory is Python tracemalloc only."""

    tracemalloc.start()
    started = time.perf_counter()
    try:
        result = operation()
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        return result, OperationMeasurement(elapsed, peak)
    finally:
        tracemalloc.stop()
