"""Model-neutral Ray/Optuna orchestration for caller-defined search spaces."""

from __future__ import annotations

import math
import os
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

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
    """Reusable tuning policy; callers still define the search space."""

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
    reduction_factor: int = 3
    # False skips LossTerm(physics=True) residuals in trial validation unless
    # the tuning or selection metric depends on them.
    validation_physics: bool = False
    optuna_seed: int = 0
    # None resolves to num_samples // 5 so random exploration scales with budget.
    optuna_startup_trials: int | None = None
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
        if self.optuna_startup_trials is None:
            object.__setattr__(self, "optuna_startup_trials", self.num_samples // 5)
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
    """Run or resume Ray Tune with one standardized ASHA/Optuna policy.

    ``trainer_factory(config[, context=...])`` receives each sampled configuration
    with ``seed`` set to the trial context's seed and ``epochs`` set to
    ``config.max_epochs``, so the search space must not define either.
    """

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
        reserved = sorted(_INJECTED_KEYS & set(search_space))
        if trainer_factory is not None and reserved:
            raise ValueError(
                f"search_space must not define {reserved}: the tuner sets 'seed' from the "
                "trial context and 'epochs' from max_epochs."
            )
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
            _local_ray_defaults()
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
                    trainer_config = {
                        **trial_config,
                        "seed": trial_context.seed,
                        "epochs": self.config.max_epochs,
                    }
                    trainer = (
                        self.trainer_factory(trainer_config, context=trial_context)
                        if factory_takes_context
                        else self.trainer_factory(trainer_config)
                    )
                    if hasattr(trainer, "reporter"):
                        trainer.reporter = report
                    _configure_validation_physics(trainer, self.config)
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
                    or min(
                        self.config.max_epochs,
                        max(3, self.config.max_epochs // 5),
                    ),
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
                artifact_store.write_tuning_progress(
                    row for result in results for row in _trial_progress(result)
                )
                _record_tuning_summary(artifact_store, self.config, trials, tuning_seconds)
            return outcome
        finally:
            if owned_ray:
                ray.shutdown()


def _record_tuning_summary(
    store: ArtifactStore,
    config: TuningConfig,
    trials: Sequence[Mapping[str, Any]],
    tuning_seconds: float,
) -> None:
    """Add the tuning time, time budget and completed-trial count to the manifest.

    ``tuning_budget.completed_trials`` counts trials that ran without error,
    including ones ASHA stopped early. It is below the sample budget when
    ``time_budget_s`` stopped tuning before every trial started.
    """
    if not store.paths.manifest.is_file():
        return
    manifest = store.read_manifest()
    store.update_manifest(
        timings={"tuning_seconds": tuning_seconds},
        tuning_budget={
            **dict(manifest.get("tuning_budget") or {}),
            "time_budget_s": config.time_budget_s,
            "completed_trials": sum(trial.get("status") == "completed" for trial in trials),
        },
    )


# Trainer-config keys the tuner supplies for every trial.
_INJECTED_KEYS = frozenset({"seed", "epochs"})


def build_tuning_config(  # pylint: disable=too-many-arguments
    *,
    metric: str,
    samples: int,
    epochs: int,
    seed: int,
    device: torch.device | str,
    cpus_per_trial: float = 1,
    gpus_per_trial: float | None = None,
    max_concurrent_trials: int = 1,
    time_budget_s: float | None = None,
    ray_cpus: int | None = None,
    ray_gpus: float | None = None,
    temp_dir: str | Path | None = None,
    object_store_memory: int | None = 100 * 1024**2,
    runtime_env: Mapping[str, Any] | None = None,
) -> TuningConfig:
    """Build the shared ASHA/Optuna :class:`TuningConfig` from budgets and resources.

    ``samples`` trials of up to ``epochs`` epochs are minimized on ``metric``, with
    Optuna seeded by ``seed``. Each trial requests ``cpus_per_trial`` CPUs and
    ``gpus_per_trial`` GPUs (default: 1 when ``device`` is CUDA, else 0). The local
    Ray cluster's totals are separate: ``ray_cpus`` and ``ray_gpus`` default to the
    per-trial request times ``max_concurrent_trials``. ``runtime_env`` is passed to
    Ray unchanged; nothing is added to it.
    """
    if gpus_per_trial is None:
        gpus_per_trial = 1 if torch.device(device).type == "cuda" else 0
    if ray_cpus is None:
        ray_cpus = math.ceil(cpus_per_trial * max_concurrent_trials)
    if ray_gpus is None:
        ray_gpus = gpus_per_trial * max_concurrent_trials
    return TuningConfig(
        metric=metric,
        mode="min",
        num_samples=samples,
        max_epochs=epochs,
        resources_per_trial={"cpu": cpus_per_trial, "gpu": gpus_per_trial},
        max_concurrent_trials=max_concurrent_trials,
        time_budget_s=time_budget_s,
        optuna_seed=seed,
        ray_runtime=RayRuntimeConfig(
            num_cpus=ray_cpus,
            num_gpus=ray_gpus,
            temp_dir=None if temp_dir is None else Path(temp_dir).resolve(),
            object_store_memory=object_store_memory,
            runtime_env=runtime_env,
        ),
    )


def _local_ray_defaults() -> None:
    """Set environment defaults for the local Ray cluster that ``Tuner.fit`` starts.

    Variables the caller has already set are kept. Workers run in the driver's
    installed environment, so Ray's ``uv run`` hook (which copies the project and
    builds another environment under the Ray temp dir) is off. Ray parses that
    flag once, when it is first imported, possibly before this module was, so the
    parsed value is refreshed here. Ray's memory monitor is off as well.
    """
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")
    from ray._private import ray_constants

    ray_constants.RAY_ENABLE_UV_RUN_RUNTIME_ENV = ray_constants.env_bool(
        "RAY_ENABLE_UV_RUN_RUNTIME_ENV", True
    )


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


def _configure_validation_physics(trainer: Any, config: TuningConfig) -> None:
    """Skip trial-validation physics unless the tuning metric needs it."""
    if not hasattr(trainer, "validation_physics"):
        return
    keep = config.validation_physics
    needs = getattr(trainer, "metric_needs_validation_physics", None)
    if not keep and needs is not None and needs(config.metric):
        warnings.warn(
            f"Ray metric {config.metric!r} depends on physics terms; keeping "
            "validation physics in tuning.",
            stacklevel=2,
        )
        keep = True
    trainer.validation_physics = keep


# Ray bookkeeping columns that say nothing about the trial's learning curve.
_PROGRESS_DROPPED_COLUMNS = frozenset(
    {"pid", "hostname", "node_ip", "date", "checkpoint_dir_name", "done"}
)


def _trial_progress(result: Any) -> list[dict[str, Any]]:
    """Return one row per reported iteration of a trial, from Ray's progress log.

    ``config/*`` columns are dropped because ``tuning_trials.csv`` already
    holds each trial's configuration; only numeric values are kept.
    """
    frame = getattr(result, "metrics_dataframe", None)
    if frame is None or frame.empty:
        return []
    fallback_id = (result.metrics or {}).get("trial_id", Path(str(result.path)).name)
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict("records"):
        row: dict[str, Any] = {"trial_id": record.get("trial_id") or fallback_id}
        for name, value in record.items():
            if (
                name == "trial_id"
                or name in _PROGRESS_DROPPED_COLUMNS
                or name.startswith("config/")
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                continue
            row[name] = value.item() if hasattr(value, "item") else value
        rows.append(row)
    return rows


__all__ = [
    "DatasetPairFactory",
    "build_tuning_config",
    "RayRuntimeConfig",
    "TrialObjective",
    "TrialReporter",
    "Tuner",
    "TuningConfig",
    "TuningOutcome",
]
