"""Shared orchestration for direct and POD-DeepONet experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chem_operator._models.pod import PODTransform
from chem_operator._normalization.base import Normalizer

from .artifacts import ArtifactStore
from .evaluation import deeponet_training_config, evaluate_deeponet
from .runner import ExperimentResult, ExperimentRunner, ExperimentSpec
from .trainers import DeepONetTrainer
from .tuning import (
    DatasetPairFactory,
    RayRuntimeConfig,
    Tuner,
    TuningConfig,
    TuningOutcome,
)
from .types import RunContext

FinalDatasetFactory = Callable[[], Any]


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


def run_deeponet_variant(
    *,
    context: RunContext,
    spec: ExperimentSpec,
    normalizer: Normalizer,
    pod: PODTransform | None,
    search_space: Mapping[str, Any],
    tuning_data: DatasetPairFactory,
    final_data: FinalDatasetFactory,
    tuning_settings: DeepONetTuningSettings,
    tune: bool = True,
    train: bool = True,
    storage_path: str | Path | None = None,
    epoch_multiplier: float = 2.0,
    reconstruction_cases: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> ExperimentResult | TuningOutcome | None:
    """Tune and/or train one direct or POD-DeepONet experiment run."""

    expected_pod = {"deeponet": False, "pod_deeponet": True}
    if spec.model_id not in expected_pod:
        raise ValueError("model_id must be 'deeponet' or 'pod_deeponet'.")
    if (pod is not None) != expected_pod[spec.model_id]:
        requirement = "requires" if expected_pod[spec.model_id] else "does not accept"
        raise ValueError(f"{spec.model_id} {requirement} a POD transform.")
    runner = ExperimentRunner(context, spec)
    tuning: TuningOutcome | None = None
    result: ExperimentResult | TuningOutcome | None = None
    if tune:
        tuning = runner.tune(
            deeponet_tuner(
                search_space=search_space,
                pod=pod,
                dataset_factory=tuning_data,
                context=context,
                settings=tuning_settings,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            None,
            None,
            storage_path=storage_path,
            experiment_name=(
                f"{spec.problem_id}_{spec.model_id}_{context.paths.run_dir.name}"
            ),
        )
        result = tuning
    if not train:
        return result
    config = (
        dict(tuning.best_config)
        if tuning is not None
        else ArtifactStore(context).read_best_config()
    )
    trainer = DeepONetTrainer(
        deeponet_training_config(config, epoch_multiplier=epoch_multiplier),
        pod=pod,
        num_workers=num_workers,
        pin_memory=pin_memory,
        checkpoint_metadata={"normalizer": normalizer.state_dict()},
    )
    with final_data() as (training, validation, test):
        return runner.run(
            trainer,
            training,
            validation,
            test,
            config=config,
            tuning=tuning,
            evaluator=lambda fitted, data, run_context: evaluate_deeponet(
                fitted,
                data,
                run_context,
                normalizer=normalizer,
                reconstruction_cases=reconstruction_cases,
            ),
        )


__all__ = ["DeepONetTuningSettings", "deeponet_tuner", "run_deeponet_variant"]
