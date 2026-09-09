"""Long-form history events and streaming shared regression metrics."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """One row in a model-agnostic, long-form training history."""

    epoch: int
    split: str
    metric: str
    value: float

    def __post_init__(self) -> None:
        if self.epoch < 1:
            raise ValueError("metric event epoch must be positive.")
        if not self.split or not self.metric:
            raise ValueError("metric event split and metric must be non-empty.")
        if not math.isfinite(float(self.value)):
            raise ValueError("metric event value must be finite.")

    def to_dict(self) -> dict[str, int | str | float]:
        return asdict(self)


class History:
    """Append-only collector for named metric events."""

    def __init__(self) -> None:
        self._events: list[MetricEvent] = []

    def record(self, epoch: int, split: str, metric: str, value: float) -> None:
        self._events.append(MetricEvent(epoch, split, metric, float(value)))

    def extend(self, events: list[MetricEvent]) -> None:
        self._events.extend(events)

    @property
    def events(self) -> tuple[MetricEvent, ...]:
        return tuple(self._events)

    def values(self, *, split: str, metric: str) -> tuple[float, ...]:
        return tuple(
            event.value
            for event in self._events
            if event.split == split and event.metric == metric
        )


@dataclass(slots=True)
class StreamingRegressionMetrics:
    """Accumulate shared physical-space metrics without retaining predictions."""

    count: int = 0
    squared_error: float = 0.0
    reference_squared: float = 0.0
    max_absolute_error: float = 0.0

    def update(
        self,
        prediction: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> None:
        predicted = _as_float64_numpy(prediction)
        reference = _as_float64_numpy(target)
        if predicted.shape != reference.shape:
            raise ValueError(
                "prediction and target must have identical shapes; "
                f"received {predicted.shape} and {reference.shape}."
            )
        if predicted.size == 0:
            return
        if not np.isfinite(predicted).all() or not np.isfinite(reference).all():
            raise ValueError("prediction and target must contain only finite values.")
        error = predicted - reference
        self.count += error.size
        self.squared_error += float(np.sum(error * error))
        self.reference_squared += float(np.sum(reference * reference))
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(np.max(np.abs(error))),
        )

    def compute(self) -> dict[str, float]:
        if self.count == 0:
            raise RuntimeError("No observations have been accumulated.")
        return {
            "rmse": math.sqrt(self.squared_error / self.count),
            "relative_l2": math.sqrt(
                self.squared_error / max(self.reference_squared, 1.0e-30)
            ),
            "max_absolute_error": self.max_absolute_error,
        }


def _as_float64_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(slots=True)
class TimingRecorder:
    """Collect elapsed times under explicit stage names."""

    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        if not name:
            raise ValueError("timing name must be non-empty.")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def add(self, name: str, seconds: float) -> None:
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("elapsed time must be finite and non-negative.")
        self.timings[name] = self.timings.get(name, 0.0) + seconds
