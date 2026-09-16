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
    """Explicit switches for generation, tuning, training, and plotting."""

    generate: bool = False
    tune: bool = True
    train: bool = True
    plot: bool = False
    train_config: str = "best"
    plot_cases: int = 2

    def __post_init__(self) -> None:
        if not self.train_config:
            raise ValueError("train_config must be non-empty.")
        if self.plot_cases < 1:
            raise ValueError("plot_cases must be positive.")

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> "WorkflowStages":
        return cls(
            generate=bool(namespace.generate),
            tune=bool(namespace.tune),
            train=bool(namespace.train),
            plot=bool(namespace.plot),
            train_config=str(namespace.train_config),
            plot_cases=int(namespace.plot_cases),
        )


def add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the standard flags to a parser owned by a model script."""
    parser.add_argument(
        "--generate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compatibility hook; prefer the separate artifact plotter.",
    )
    parser.add_argument("--train-config", default="best")
    parser.add_argument("--plot-cases", type=int, default=2)


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
    """Create the same versioned run context in every model script."""
    run_id = namespace.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    return RunContext.create(
        namespace.runs_root,
        problem=problem,
        model=model,
        run_id=run_id,
        seed=seed,
        dtype=dtype,
        device=resolve_device(namespace.device),
        provenance=provenance,
    )
