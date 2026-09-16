"""Model-neutral Ray/Optuna orchestration for script-defined search spaces."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# These experiments launch local workers in the driver's installed environment.
# Ray's automatic uv hook otherwise copies the project and installs another
# environment under its temporary directory. Set this before importing Ray;
# callers can explicitly opt back in for a distributed deployment.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from .artifacts import ArtifactStore
from .trainers import accepts_context
from .types import RunContext, Trainer, TrainingOutcome

TrainerFactory = Callable[..., Trainer]
DatasetPairFactory = Callable[
    [RunContext],
    tuple[Any, Any] | AbstractContextManager[tuple[Any, Any]],
]
TrialReporter = Callable[[Mapping[str, float | int]], None]
TrialObjective = Callable[
    [Mapping[str, Any], Any, Any, RunContext, TrialReporter],
    TrainingOutcome | Mapping[str, float | int] | None,
]


@dataclass(frozen=True, slots=True)
class RayRuntimeConfig:
    """Ray process settings owned by the shared tuning layer."""

    num_cpus: int | None = None
    num_gpus: int | float | None = None
    temp_dir: str | Path | None = None
    object_store_memory: int | None = None
    runtime_env: Mapping[str, Any] | None = None
    include_dashboard: bool = False
    ignore_reinit_error: bool = True

    def __post_init__(self) -> None:
        if self.num_cpus is not None and self.num_cpus < 1:
            raise ValueError("num_cpus must be positive.")
        if self.num_gpus is not None and self.num_gpus < 0:
            raise ValueError("num_gpus cannot be negative.")
        if self.object_store_memory is not None and self.object_store_memory < 1:
            raise ValueError("object_store_memory must be positive.")
        if self.temp_dir is not None:
            object.__setattr__(self, "temp_dir", Path(self.temp_dir))
        if self.runtime_env is not None:
            object.__setattr__(self, "runtime_env", dict(self.runtime_env))

    def init_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by :func:`ray.init`."""
        values: dict[str, Any] = {
            "include_dashboard": self.include_dashboard,
            "ignore_reinit_error": self.ignore_reinit_error,
        }
        for name in ("num_cpus", "num_gpus", "object_store_memory", "runtime_env"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        if self.temp_dir is not None:
            values["_temp_dir"] = str(self.temp_dir)
        return values


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Reusable tuning policy; experiment scripts still define the search space."""

    metric: str = "valid_relative_l2"
    mode: str = "min"
    num_samples: int = 1
    max_epochs: int = 1
    resources_per_trial: Mapping[str, float] = field(
        default_factory=lambda: {"cpu": 1.0}
    )
    max_concurrent_trials: int | None = None
    time_budget_s: float | None = None
    grace_period: int | None = None
    reduction_factor: int = 2
    optuna_seed: int = 0
    optuna_startup_trials: int = 2
    optuna_multivariate: bool = True
    best_scope: str = "last"
    resume: bool = True
    resume_unfinished: bool = True
    resume_errored: bool = True
    reuse_actors: bool = False
    verbose: int = 1
    ray_runtime: RayRuntimeConfig = field(default_factory=RayRuntimeConfig)

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric must be non-empty.")
        if self.mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if self.num_samples < 1 or self.max_epochs < 1:
            raise ValueError("num_samples and max_epochs must be positive.")
        if self.max_concurrent_trials is not None and self.max_concurrent_trials < 1:
            raise ValueError("max_concurrent_trials must be positive.")
        if self.time_budget_s is not None and self.time_budget_s <= 0:
            raise ValueError("time_budget_s must be positive.")
        if self.grace_period is not None and self.grace_period < 1:
            raise ValueError("grace_period must be positive.")
        if self.reduction_factor < 2:
            raise ValueError("reduction_factor must be at least 2.")
        if self.optuna_seed < 0 or self.optuna_startup_trials < 0:
            raise ValueError("Optuna seed/startup trial counts cannot be negative.")
        if self.verbose not in {0, 1, 2, 3}:
            raise ValueError("verbose must be between 0 and 3.")
        if not self.resources_per_trial:
            raise ValueError("resources_per_trial cannot be empty.")
        if any(value < 0 for value in self.resources_per_trial.values()):
            raise ValueError("resources_per_trial values cannot be negative.")
        object.__setattr__(self, "resources_per_trial", dict(self.resources_per_trial))


@dataclass(frozen=True, slots=True)
class TuningOutcome:
    """Best configuration and portable, flattened trial records."""

    best_config: Mapping[str, Any]
    trials: Sequence[Mapping[str, Any]]
    tuning_seconds: float
    best_metrics: Mapping[str, Any] = field(default_factory=dict)


class Tuner:
    """Run or resume Ray Tune with one standardized ASHA/Optuna policy."""

    def __init__(
        self,
        trainer_factory: TrainerFactory | None,
        search_space: Mapping[str, Any],
        *,
        config: TuningConfig | None = None,
        dataset_factory: DatasetPairFactory | None = None,
        objective: TrialObjective | None = None,
    ) -> None:
        if not search_space:
            raise ValueError("search_space cannot be empty.")
        if (trainer_factory is None) == (objective is None):
            raise ValueError("Provide exactly one of trainer_factory or objective.")
        self.trainer_factory = trainer_factory
        self.objective = objective
        self.search_space = dict(search_space)
        self.config = TuningConfig() if config is None else config
        self.dataset_factory = dataset_factory

    @classmethod
    def from_objective(
        cls,
        objective: TrialObjective,
        search_space: Mapping[str, Any],
        *,
        config: TuningConfig | None = None,
        dataset_factory: DatasetPairFactory | None = None,
    ) -> "Tuner":
        """Adapt an existing trial function while its trainer is being migrated."""
        return cls(
            None,
            search_space,
            config=config,
            dataset_factory=dataset_factory,
            objective=objective,
        )

    def fit(  # pylint: disable=too-many-locals
        self,
        train: Any,
        validation: Any,
        context: RunContext,
        *,
        storage_path: str | Path | None = None,
        experiment_name: str = "tuning",
        artifact_store: ArtifactStore | None = None,
    ) -> TuningOutcome:
        """Own the Ray lifecycle, run/resume tuning, and persist portable results."""
        import optuna
        import ray
        from ray import tune
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch

        if not experiment_name or Path(experiment_name).name != experiment_name:
            raise ValueError("experiment_name must be one path component.")
        owned_ray = not ray.is_initialized()
        if owned_ray:
            ray.init(**self.config.ray_runtime.init_kwargs())

        factory_takes_context = (
            accepts_context(self.trainer_factory)
            if self.trainer_factory is not None
            else False
        )

        def trainable(trial_config: Mapping[str, Any]) -> None:
            trial_id = tune.get_context().get_trial_id() or "trial"
            trial_context = context.trial(trial_id)
            reported = False

            def report(values: Mapping[str, float | int]) -> None:
                nonlocal reported
                reported = True
                tune.report(_metric_payload(values, self.config.metric))

            datasets = (
                nullcontext((train, validation))
                if self.dataset_factory is None
                else _dataset_context(self.dataset_factory(trial_context))
            )
            with datasets as (trial_train, trial_validation):
                if self.objective is not None:
                    result = self.objective(
                        trial_config,
                        trial_train,
                        trial_validation,
                        trial_context,
                        report,
                    )
                else:
                    assert self.trainer_factory is not None
                    trainer = (
                        self.trainer_factory(trial_config, context=trial_context)
                        if factory_takes_context
                        else self.trainer_factory(trial_config)
                    )
                    if hasattr(trainer, "reporter"):
                        trainer.reporter = report
                    result = trainer.fit(trial_train, trial_validation, trial_context)
            if not reported:
                report(_result_metrics(result, self.config.metric))

        configured_trainable = tune.with_resources(
            trainable,
            resources=dict(self.config.resources_per_trial),
        )
        root = (
            context.paths.run_dir / "ray_results"
            if storage_path is None
            else Path(storage_path)
        )
        experiment_path = root / experiment_name
        started = time.perf_counter()
        try:
            if self.config.resume and tune.Tuner.can_restore(str(experiment_path)):
                tuner = tune.Tuner.restore(
                    str(experiment_path),
                    trainable=configured_trainable,
                    resume_unfinished=self.config.resume_unfinished,
                    resume_errored=self.config.resume_errored,
                )
            else:
                scheduler = ASHAScheduler(
                    metric=self.config.metric,
                    mode=self.config.mode,
                    time_attr="training_iteration",
                    max_t=self.config.max_epochs,
                    grace_period=self.config.grace_period
                    or max(1, self.config.max_epochs // 3),
                    reduction_factor=self.config.reduction_factor,
                )
                search_algorithm = OptunaSearch(
                    metric=self.config.metric,
                    mode=self.config.mode,
                    sampler=optuna.samplers.TPESampler(
                        seed=self.config.optuna_seed,
                        n_startup_trials=self.config.optuna_startup_trials,
                        multivariate=self.config.optuna_multivariate,
                    ),
                )
                tuner = tune.Tuner(
                    configured_trainable,
                    param_space=self.search_space,
                    tune_config=tune.TuneConfig(
                        num_samples=self.config.num_samples,
                        scheduler=scheduler,
                        search_alg=search_algorithm,
                        max_concurrent_trials=self.config.max_concurrent_trials,
                        time_budget_s=self.config.time_budget_s,
                        reuse_actors=self.config.reuse_actors,
                    ),
                    run_config=tune.RunConfig(
                        name=experiment_name,
                        storage_path=str(root.resolve()),
                        verbose=self.config.verbose,
                    ),
                )
            results = tuner.fit()
            tuning_seconds = time.perf_counter() - started
            best = results.get_best_result(
                metric=self.config.metric,
                mode=self.config.mode,
                scope=self.config.best_scope,
            )
            best_config = dict(best.config)
            trials = tuple(
                _trial_record(result, metric=self.config.metric) for result in results
            )
            outcome = TuningOutcome(
                best_config,
                trials,
                tuning_seconds,
                {
                    key: value
                    for key, value in dict(best.metrics or {}).items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                },
            )
            if artifact_store is not None:
                artifact_store.write_best_config(best_config)
                artifact_store.write_tuning_trials(trials)
                if artifact_store.paths.manifest.is_file():
                    artifact_store.update_manifest(
                        timings={"tuning_seconds": tuning_seconds}
                    )
            return outcome
        finally:
            if owned_ray:
                ray.shutdown()


def _metric_payload(
    values: Mapping[str, float | int], metric: str
) -> dict[str, float | int]:
    payload = dict(values)
    if metric not in payload:
        aliases = {
            "valid_relative_l2": "validation_relative_l2",
            "best_valid_loss": "validation_loss",
        }
        alias = aliases.get(metric)
        if alias is not None and alias in payload:
            payload[metric] = payload[alias]
    if metric not in payload:
        raise KeyError(f"Trial report did not contain tuning metric {metric!r}.")
    return payload


def _result_metrics(
    result: TrainingOutcome | Mapping[str, float | int] | None,
    metric: str,
) -> dict[str, float | int]:
    if result is None:
        raise RuntimeError("The trial neither reported metrics nor returned an outcome.")
    if isinstance(result, TrainingOutcome):
        payload: dict[str, float | int] = dict(result.metrics)
        payload["best_epoch"] = result.best_epoch
        payload.update(
            {
                key: value
                for key, value in result.metadata.items()
                if isinstance(value, (int, float))
            }
        )
        return _metric_payload(payload, metric)
    return _metric_payload(result, metric)


def _dataset_context(
    value: tuple[Any, Any] | AbstractContextManager[tuple[Any, Any]],
) -> AbstractContextManager[tuple[Any, Any]]:
    if hasattr(value, "__enter__") and hasattr(value, "__exit__"):
        return value  # type: ignore[return-value]
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("dataset_factory must return a pair or a context manager.")
    return nullcontext(value)


def _trial_record(result: Any, *, metric: str) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    record: dict[str, Any] = {
        "trial_id": metrics.get("trial_id", Path(str(result.path)).name),
        "status": "error" if result.error is not None else "completed",
        "metric": metric,
        "value": metrics.get(metric),
        "training_iteration": metrics.get("training_iteration"),
    }
    record.update(
        {
            f"config.{name}": value
            for name, value in sorted(dict(result.config).items())
        }
    )
    if result.error is not None:
        record["error"] = str(result.error)
    return record


__all__ = [
    "DatasetPairFactory",
    "RayRuntimeConfig",
    "TrialObjective",
    "TrialReporter",
    "Tuner",
    "TuningConfig",
    "TuningOutcome",
]
