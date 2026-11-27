"""Shared workflow validation and CLI helpers for model experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

import torch

from .data_loading import DATA_CACHE_MODES, data_loading_provenance
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
        if not (self.tune or self.train):
            raise ValueError("At least one of tuning or training must be enabled.")
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


def parse_workflow_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse arguments and report workflow validation failures as CLI errors.

    The parser must include :func:`add_workflow_arguments`. Validate before
    resolving caches, opening datasets, or creating run artifacts.
    """
    args = parser.parse_args(argv)
    try:
        WorkflowStages.from_namespace(args)
    except ValueError as error:
        parser.error(str(error))
    return args


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
    parser.add_argument(
        "--num-workers",
        type=_non_negative_int,
        default=0,
        help=(
            "DataLoader worker processes (default: 0). Workers only help when "
            "data is not cached in memory."
        ),
    )
    parser.add_argument(
        "--data-cache",
        choices=DATA_CACHE_MODES,
        default="auto",
        help=(
            "Keep dataset arrays in memory after the first read: 'auto' "
            "(default) caches when the estimated size fits in 25%% of "
            "available RAM."
        ),
    )
    # Tuning resources describe the machine, not the experiment: every script
    # shares these defaults (1 CPU, GPUs from --device, 1 trial, no time limit).
    parser.add_argument(
        "--cpus-per-trial",
        type=_positive_float,
        default=1,
        help=(
            "CPUs each tuning trial requests from Ray (default: 1). Ray also sets "
            "the trial's OMP_NUM_THREADS to this value."
        ),
    )
    # None means 1 GPU per trial when --device resolves to CUDA, else 0. A whole
    # GPU per trial makes Ray queue extra concurrent trials on a single-GPU
    # machine; a fraction such as 0.5 lets two trials share it.
    parser.add_argument(
        "--gpus-per-trial",
        type=_non_negative_float,
        default=None,
        help=(
            "GPUs each tuning trial requests (default: 1 when --device resolves "
            "to CUDA, else 0). With --concurrent-trials above 1 on one GPU, pass "
            "a fraction such as 0.5; with whole GPUs Ray queues the extra trials "
            "instead of running them in parallel."
        ),
    )
    parser.add_argument(
        "--concurrent-trials",
        type=_positive_int,
        default=1,
        help=(
            "Tuning trials run at the same time (default: 1). Values above 1 are "
            "not reproducible: Optuna proposes new trials before earlier ones "
            "finish, and the finishing order varies, so a fixed seed no longer "
            "fixes the tuning result."
        ),
    )
    parser.add_argument(
        "--time-budget",
        type=_positive_float,
        default=None,
        metavar="SECONDS",
        help=(
            "Stop tuning after this many seconds (default: no limit). How many "
            "trials finish then depends on machine speed and load."
        ),
    )


def tuning_resources(args: argparse.Namespace) -> dict[str, Any]:
    """Return :func:`build_tuning_config` keyword arguments from the resource flags.

    A namespace without the flags gets the library defaults: 1 CPU per trial,
    GPUs from the device, 1 concurrent trial and no time limit.
    """
    return {
        "cpus_per_trial": getattr(args, "cpus_per_trial", 1),
        "gpus_per_trial": getattr(args, "gpus_per_trial", None),
        "max_concurrent_trials": int(getattr(args, "concurrent_trials", 1)),
        "time_budget_s": getattr(args, "time_budget", None),
    }


def _non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if not number >= 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return number


def final_epochs(args: argparse.Namespace) -> int:
    """Return final-training epochs: ``--epochs`` when given, otherwise ``2 * --tune-epochs``."""
    epochs = getattr(args, "epochs", None)
    if epochs is None:
        tune_epochs = int(args.tune_epochs)
        if tune_epochs < 1:
            raise ValueError("--tune-epochs must be positive.")
        return 2 * tune_epochs
    if int(epochs) < 1:
        raise ValueError("--epochs must be positive.")
    return int(epochs)


def final_epochs_provenance(args: argparse.Namespace) -> dict[str, object]:
    """Return the resolved final epochs and their source (``2x_tune`` or ``override``)."""
    return {
        "final_epochs": final_epochs(args),
        "final_epochs_source": "2x_tune" if getattr(args, "epochs", None) is None else "override",
    }


def tuning_resources_provenance(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Return the per-trial resources and concurrency, with GPUs resolved for ``device``."""
    resources = tuning_resources(args)
    gpus = resources["gpus_per_trial"]
    return {
        "cpus_per_trial": resources["cpus_per_trial"],
        "gpus_per_trial": (1 if device.type == "cuda" else 0) if gpus is None else gpus,
        "concurrent_trials": resources["max_concurrent_trials"],
    }


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
    Namespaces from :func:`parse_operator_args` also record the resolved final
    epochs and their source, and the tuning resources, in the provenance.
    """
    run_id = namespace.run_id or datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    tune_seed = getattr(namespace, "tune_seed", None)
    provenance = dict(provenance or {})
    if hasattr(namespace, "num_workers"):
        provenance.setdefault("data_loading", data_loading_provenance(namespace))
    if hasattr(namespace, "tune_epochs"):
        for key, value in final_epochs_provenance(namespace).items():
            provenance.setdefault(key, value)
    device = resolve_device(namespace.device)
    if hasattr(namespace, "cpus_per_trial"):
        provenance.setdefault("tuning_resources", tuning_resources_provenance(namespace, device))
    return RunContext.create(
        namespace.runs_root,
        problem=problem,
        model=model,
        run_id=run_id,
        seed=seed if tune_seed is None else int(tune_seed),
        dtype=dtype,
        device=device,
        num_workers=int(getattr(namespace, "num_workers", 0)),
        provenance=provenance,
    )
