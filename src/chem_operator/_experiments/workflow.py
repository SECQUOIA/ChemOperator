"""Shared CLI stage values; argument parsing remains in model scripts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkflowStages:
    """Explicit switches for generation, tuning, training, and plotting."""

    generate: bool = False
    tune: bool = True
    train: bool = True
    plot: bool = True
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
        default=True,
    )
    parser.add_argument("--train-config", default="best")
    parser.add_argument("--plot-cases", type=int, default=2)
