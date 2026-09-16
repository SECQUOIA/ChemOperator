"""Small, problem-agnostic contracts for model experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from .metrics import MetricEvent


@dataclass(frozen=True, slots=True)
class RunPaths:
    """Canonical paths for one versioned experiment run."""

    run_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        problem: str,
        model: str,
        run_id: str,
    ) -> "RunPaths":
        for name, value in (
            ("problem", problem),
            ("model", model),
            ("run_id", run_id),
        ):
            if not value or Path(value).name != value:
                raise ValueError(f"{name} must be one non-empty path component.")
        return cls(Path(runs_root) / problem / model / run_id)

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def best_config(self) -> Path:
        return self.run_dir / "best_config.json"

    @property
    def history(self) -> Path:
        return self.run_dir / "history.csv"

    @property
    def metrics(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def reconstructions(self) -> Path:
        return self.run_dir / "reconstructions.npz"

    @property
    def tuning_trials(self) -> Path:
        return self.run_dir / "tuning_trials.csv"

    def checkpoint(self, suffix: str = ".pt") -> Path:
        if not suffix.startswith(".") or Path(suffix).name != suffix:
            raise ValueError("checkpoint suffix must start with '.' and be a suffix.")
        return self.run_dir / f"checkpoint{suffix}"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Explicit paths, execution settings, and provenance for one run."""

    paths: RunPaths
    seed: int
    dtype: torch.dtype
    device: torch.device
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.paths, RunPaths):
            object.__setattr__(self, "paths", RunPaths(Path(self.paths)))
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        object.__setattr__(self, "device", torch.device(self.device))
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype.")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        *,
        problem: str,
        model: str,
        run_id: str,
        seed: int,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        provenance: Mapping[str, Any] | None = None,
    ) -> "RunContext":
        return cls(
            paths=RunPaths.create(runs_root, problem, model, run_id),
            seed=seed,
            dtype=dtype,
            device=torch.device(device),
            provenance={} if provenance is None else provenance,
        )

    def trial(self, trial_id: str) -> "RunContext":
        """Return an isolated context for one tuning worker."""
        if not trial_id or Path(trial_id).name != trial_id:
            raise ValueError("trial_id must be one non-empty path component.")
        return RunContext(
            paths=RunPaths(self.paths.run_dir / "trials" / trial_id),
            seed=self.seed,
            dtype=self.dtype,
            device=self.device,
            provenance={**self.provenance, "trial_id": trial_id},
        )


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Serializable facts produced by a trainer's final fit."""

    history: Sequence["MetricEvent"]
    best_epoch: int
    timings: Mapping[str, float]
    checkpoint: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.best_epoch < 1:
            raise ValueError("best_epoch must be positive.")
        object.__setattr__(self, "history", tuple(self.history))
        object.__setattr__(self, "timings", dict(self.timings))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "metrics", dict(self.metrics))


@runtime_checkable
class Trainer(Protocol):
    """Minimal interface implemented by every model-family trainer."""

    def fit(
        self,
        train: Any,
        validation: Any,
        context: RunContext,
    ) -> TrainingOutcome:
        """Fit a model and restore its best validation checkpoint."""

    def predict(self, data: Any, context: RunContext) -> Any:
        """Evaluate the currently loaded checkpoint in batches."""

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist the currently loaded model."""

    def load_checkpoint(self, path: str | Path, context: RunContext) -> None:
        """Load a checkpoint onto ``context.device``."""
