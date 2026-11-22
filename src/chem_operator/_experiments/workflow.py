"""Shared CLI stage values; argument parsing remains in model scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .device import resolve_device
from .types import RunContext


@dataclass(frozen=True, slots=True)
class WorkflowStages:
    """Explicit switches for tuning and training."""

    tune: bool = True
    train: bool = True
    train_config: str = "best"
    plot_cases: int = 2
    seeds: tuple[int, ...] | None = None
    tune_seed: int | None = None

    def __post_init__(self) -> None:
        if not self.train_config:
            raise ValueError("train_config must be non-empty.")
        if self.plot_cases < 1:
            raise ValueError("plot_cases must be positive.")
        if self.seeds is not None:
            seeds = tuple(int(seed) for seed in self.seeds)
            if not seeds:
                raise ValueError("--seeds requires at least one seed.")
            if any(seed < 0 for seed in seeds):
                raise ValueError("Seeds cannot be negative.")
            if len(set(seeds)) != len(seeds):
                raise ValueError("Seeds must be unique.")
            if not self.train:
                raise ValueError("--seeds requires --train.")
            object.__setattr__(self, "seeds", seeds)
        if self.tune_seed is not None and int(self.tune_seed) < 0:
            raise ValueError("--tune-seed cannot be negative.")

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "WorkflowStages":
        seeds = getattr(namespace, "seeds", None)
        tune_seed = getattr(namespace, "tune_seed", None)
        return cls(
            tune=bool(namespace.tune),
            train=bool(namespace.train),
            train_config=str(namespace.train_config),
            plot_cases=int(namespace.plot_cases),
            seeds=None if seeds is None else tuple(int(seed) for seed in seeds),
            tune_seed=None if tune_seed is None else int(tune_seed),
        )


def add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard flags to a parser owned by a model script."""
    parser.add_argument(
        "--tune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train-config",
        default="best",
        help=(
            "Hyperparameters for final training: 'best' (this run's "
            "best_config.json), 'recent' (the newest tuned run of this model), "
            "a completed run directory, or a JSON file."
        ),
    )
    parser.add_argument("--plot-cases", type=int, default=2)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        metavar="SEED",
        help=(
            "Train the final model once per seed under <run>/seeds/<seed>/; "
            "requires --train. Omit to write one flat run with the default seed."
        ),
    )
    parser.add_argument(
        "--tune-seed",
        type=int,
        default=None,
        help="Seed for the Optuna study (default: the problem's SEED constant).",
    )


def add_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_runs_root: str | Path = "artifacts/runs",
) -> None:
    """Add canonical run identity options to a model-owned parser."""
    parser.add_argument("--runs-root", type=Path, default=Path(default_runs_root))
    parser.add_argument(
        "--run-id",
        help="One run-directory component (default: a UTC timestamp).",
    )
    parser.add_argument("--device", default="auto")


def run_context_from_namespace(
    namespace: argparse.Namespace,
    *,
    problem: str,
    model: str,
    seed: int,
    dtype: torch.dtype = torch.float32,
    provenance: dict[str, object] | None = None,
) -> RunContext:
    """Create the same versioned run context in every model script.

    ``seed`` is the problem's default; ``--tune-seed`` overrides it. The
    resulting context seeds tuning and, for flat single-seed runs, training.
    Multi-seed children are derived with :meth:`RunContext.seed_child`.
    """
    run_id = namespace.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    tune_seed = getattr(namespace, "tune_seed", None)
    return RunContext.create(
        namespace.runs_root,
        problem=problem,
        model=model,
        run_id=run_id,
        seed=seed if tune_seed is None else int(tune_seed),
        dtype=dtype,
        device=resolve_device(namespace.device),
        provenance=provenance,
    )
