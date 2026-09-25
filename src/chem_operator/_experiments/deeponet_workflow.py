"""Shared orchestration for direct and POD-DeepONet experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chem_operator._models.pod import PODTransform

from .evaluation import deeponet_training_config
from .trainers import DeepONetTrainer
from .tuning import (
    DatasetPairFactory,
    RayRuntimeConfig,
    Tuner,
    TuningConfig,
)
from .types import RunContext


@dataclass(frozen=True, slots=True)
class DeepONetTuningSettings:
    max_epochs: int
    num_samples: int
    max_concurrent_trials: int = 1
    time_budget_s: float | None = None
    cpus_per_trial: float = 1
    gpus_per_trial: float = 0
    startup_trials: int = 2
    grace_period: int | None = None
    reduction_factor: int = 2
    ray_runtime: RayRuntimeConfig = RayRuntimeConfig()

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.num_samples < 1:
            raise ValueError("max_epochs and num_samples must be positive.")


def deeponet_tuner(
    *,
    search_space: Mapping[str, Any],
    pod: PODTransform | None,
    dataset_factory: DatasetPairFactory,
    context: RunContext,
    settings: DeepONetTuningSettings,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuner:
    def trainer_factory(config: Mapping[str, Any]) -> DeepONetTrainer:
        return DeepONetTrainer(
            deeponet_training_config(config, display_every=1),
            pod=pod,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    metric = "best_valid_loss"
    return Tuner(
        trainer_factory,
        search_space,
        config=TuningConfig(
            metric=metric,
            mode="min",
            num_samples=settings.num_samples,
            max_epochs=settings.max_epochs,
            resources_per_trial={
                "cpu": settings.cpus_per_trial,
                "gpu": settings.gpus_per_trial,
            },
            max_concurrent_trials=settings.max_concurrent_trials,
            time_budget_s=settings.time_budget_s,
            grace_period=settings.grace_period or max(1, settings.max_epochs // 3),
            reduction_factor=settings.reduction_factor,
            optuna_seed=context.seed,
            optuna_startup_trials=settings.startup_trials,
            optuna_multivariate=True,
            best_scope="last",
            ray_runtime=settings.ray_runtime,
        ),
        dataset_factory=dataset_factory,
    )


__all__ = ["DeepONetTuningSettings", "deeponet_tuner"]
