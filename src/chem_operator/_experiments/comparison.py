"""Artifact-driven, plotting-independent comparison helpers.

A run directory may be a flat single-seed run or a multi-seed parent whose
children live under ``seeds/<seed>/``. A run's *members* are the root itself
when it holds ``metrics.csv`` plus every completed child; aggregates across
members are computed here at read time and never stored.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ARTIFACT_SCHEMA_VERSION, ArtifactStore, ArtifactValidationError
from .metrics import MetricEvent
from .types import SEEDS_DIRNAME

COMPARABLE_MANIFEST_FIELDS = (
    "problem_id",
    "benchmark_protocol_id",
    "dataset_fingerprints",
    "fields",
    "channels",
    "units",
    "coordinates",
    "selected_test_case_ids",
)
MEMBER_MANIFEST_FIELDS = ("model_id", *COMPARABLE_MANIFEST_FIELDS)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Portable tabular artifacts loaded from one completed run.

    For a multi-seed parent, ``history`` and ``metrics`` belong to the
    representative (first) member and ``members`` lists every seed run.
    """

    path: Path
    manifest: Mapping[str, Any]
    history: Sequence[MetricEvent]
    metrics: Sequence[Mapping[str, Any]]
    members: tuple["RunArtifacts", ...] = field(default=())

    @property
    def label(self) -> str:
        if self.path.parent.name == SEEDS_DIRNAME:
            return (
                f"{self.manifest['problem_id']}/{self.manifest['model_id']}/"
                f"{self.path.parent.parent.name}/{SEEDS_DIRNAME}/{self.path.name}"
            )
        return (
            f"{self.manifest['problem_id']}/{self.manifest['model_id']}/"
            f"{self.path.name}"
        )

    @property
    def seed_members(self) -> tuple["RunArtifacts", ...]:
        """Every seed run contributing to this run (itself for flat runs)."""
        return self.members or (self,)

    @property
    def representative(self) -> "RunArtifacts":
        return self.seed_members[0]

    @property
    def seed(self) -> int:
        return int(self.manifest["random_seed"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(member.seed for member in self.seed_members)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Mean and sample standard deviation of one scalar across members."""

    mean: float
    std: float
    count: int
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AggregateSeries:
    """Per-epoch mean/std of one history metric across members."""

    split: str
    metric: str
    epochs: tuple[int, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    count: int


@dataclass(frozen=True, slots=True)
class ComparisonSeries:
    """One directly comparable metric series for a run."""

    label: str
    events: Sequence[MetricEvent]
    summary: AggregateSeries | None = None


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """One tidy metric value addressable by problem, model, run, and seed."""

    problem: str
    model: str
    run_id: str
    benchmark_protocol: str
    split: str
    metric: str
    value: float
    seed: int = 0


def _summarize(values: Sequence[float]) -> MetricSummary:
    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("Cannot summarize an empty sequence.")
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return MetricSummary(statistics.fmean(values), std, len(values), values)


def _split_aliases(split: str) -> frozenset[str]:
    if split in {"val", "validation"}:
        return frozenset({"val", "validation"})
    return frozenset({split})


def _load_flat(run_dir: Path, manifest: Mapping[str, Any]) -> RunArtifacts:
    store = ArtifactStore(run_dir)
    history = tuple(_read_history(store.paths.history))
    metrics = tuple(_read_csv(store.paths.metrics))
    return RunArtifacts(run_dir, manifest, history, metrics)


def seed_member_dirs(run_dir: str | Path, *, require_complete: bool = True) -> tuple[Path, ...]:
    """Return ``seeds/<seed>`` child directories sorted by seed."""
    seeds_dir = Path(run_dir) / SEEDS_DIRNAME
    if not seeds_dir.is_dir():
        return ()
    children = []
    for child in seeds_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        if require_complete:
            with manifest_path.open(encoding="utf-8") as stream:
                if json.load(stream).get("status") != "completed":
                    continue
        children.append(child)
    return tuple(sorted(children, key=lambda path: int(path.name)))


def _validate_members(
    run_dir: Path,
    manifest: Mapping[str, Any],
    members: Sequence[RunArtifacts],
    *,
    require_complete: bool,
) -> None:
    if not members:
        raise ValueError(f"Run at {run_dir} has no completed seed members.")
    seeds = [member.seed for member in members]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Run at {run_dir} has duplicate seed members: {seeds}.")
    parent_config = None
    if (run_dir / "best_config.json").is_file():
        parent_config = ArtifactStore(run_dir).read_best_config()
    for member in members:
        if member.path.parent.name == SEEDS_DIRNAME and member.seed != int(member.path.name):
            raise ValueError(f"Member {member.label} directory does not match its random_seed.")
        mismatched = [
            name for name in MEMBER_MANIFEST_FIELDS
            if member.manifest.get(name) != manifest.get(name)
        ]
        if mismatched:
            raise ValueError(
                f"Member {member.label} is not a seed of {run_dir}; mismatched "
                + ", ".join(mismatched) + "."
            )
        if parent_config is not None and member.path != run_dir:
            member_config_path = member.path / "best_config.json"
            if member_config_path.is_file():
                with member_config_path.open(encoding="utf-8") as stream:
                    member_config = json.load(stream)
                # Children store the trainer's full config, so compare only
                # the hyperparameters both files name; seed and epochs vary.
                differing = [
                    key for key in parent_config
                    if key not in ("seed", "epochs")
                    and key in member_config
                    and member_config[key] != parent_config[key]
                ]
                if differing:
                    raise ValueError(
                        f"Member {member.label} trained a different best_config than "
                        f"{run_dir}: " + ", ".join(differing) + "."
                    )
    if require_complete:
        listed = manifest.get("seeds") or ()
        missing = [int(seed) for seed in listed if int(seed) not in seeds]
        if missing:
            raise ValueError(
                f"Run {run_dir} lists seeds {missing} but they are not completed."
            )


def load_run(path: str | Path, *, require_complete: bool = True) -> RunArtifacts:
    """Load validated CSV/JSON artifacts without importing any model code.

    A multi-seed parent returns its representative member's history and
    metrics together with ``members``; a flat run returns ``members=()``.
    """
    run_dir = Path(path)
    store = ArtifactStore(run_dir)
    manifest = store.read_manifest()
    if manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Cannot compare runs with an unsupported artifact schema.")
    if require_complete and manifest["status"] != "completed":
        raise ValueError(f"Run at {path} is not completed.")
    children = seed_member_dirs(run_dir, require_complete=True)
    if not children:
        return _load_flat(run_dir, manifest)
    members: list[RunArtifacts] = []
    if store.paths.metrics.is_file():
        members.append(_load_flat(run_dir, manifest))
    members.extend(_load_flat(child, ArtifactStore(child).read_manifest()) for child in children)
    _validate_members(run_dir, manifest, members, require_complete=require_complete)
    representative = members[0]
    return RunArtifacts(
        run_dir, manifest, representative.history, representative.metrics, members=tuple(members),
    )


def validate_complete_run(run: RunArtifacts | str | Path) -> RunArtifacts:
    """Check that a run and every seed member hold all required artifacts."""
    loaded = run if isinstance(run, RunArtifacts) else load_run(run)
    store = ArtifactStore(loaded.path)
    missing: list[str] = []
    for name in ("best_config.json", "tuning_trials.csv"):
        if not (loaded.path / name).is_file():
            missing.append(name)
    for member in loaded.seed_members:
        prefix = "" if member.path == loaded.path else (
            f"{member.path.relative_to(loaded.path)}/"
        )
        for name in ("history.csv", "metrics.csv", "reconstructions.npz"):
            if not (member.path / name).is_file():
                missing.append(prefix + name)
        if not tuple(member.path.glob("checkpoint.*")):
            missing.append(prefix + "checkpoint.*")
    if missing:
        raise ArtifactValidationError(
            "Completed run is missing artifacts: " + ", ".join(missing)
        )
    if store.read_manifest()["status"] != "completed":
        raise ArtifactValidationError("Run manifest is not completed.")
    return loaded


def aggregate_history(
    members: Iterable[RunArtifacts],
    *,
    metric: str,
    split: str,
) -> AggregateSeries:
    """Average one history metric across members on their common epochs.

    Members whose early stopping produced different epoch counts are
    truncated to the epochs they all share, so ``count`` is constant.
    """
    aliases = _split_aliases(split)
    per_member: list[dict[int, float]] = []
    loaded = tuple(members)
    for member in loaded:
        values: dict[int, float] = {}
        for event in member.history:
            if event.metric != metric or event.split not in aliases:
                continue
            if event.epoch in values:
                raise ValueError(
                    f"Run {member.label} repeats epoch {event.epoch} for {split} {metric}."
                )
            values[event.epoch] = event.value
        if not values:
            raise ValueError(f"Run {member.label} has no {split} {metric} history.")
        per_member.append(values)
    epochs = sorted(set.intersection(*(set(values) for values in per_member)))
    if not epochs:
        raise ValueError(f"Members share no epochs for {split} {metric}.")
    columns = [[values[epoch] for values in per_member] for epoch in epochs]
    summaries = [_summarize(column) for column in columns]
    return AggregateSeries(
        split=split,
        metric=metric,
        epochs=tuple(epochs),
        mean=tuple(summary.mean for summary in summaries),
        std=tuple(summary.std for summary in summaries),
        minimum=tuple(min(column) for column in columns),
        maximum=tuple(max(column) for column in columns),
        count=len(per_member),
    )


def aggregate_metrics(
    members: Iterable[RunArtifacts],
    *,
    split: str = "test",
    metric: str | None = None,
) -> dict[str, MetricSummary]:
    """Summarize ``metrics.csv`` scalars across members."""
    loaded = tuple(members)
    if not loaded:
        raise ValueError("At least one member is required.")
    per_member: list[dict[str, float]] = []
    for member in loaded:
        values: dict[str, float] = {}
        for row in member.metrics:
            if str(row.get("split")) != split:
                continue
            name = str(row["metric"])
            if name in values:
                raise ValueError(f"Run {member.label} repeats {split} {name}.")
            values[name] = float(row["value"])
        per_member.append(values)
    names = {metric} if metric is not None else set().union(*(set(v) for v in per_member))
    summaries: dict[str, MetricSummary] = {}
    for name in sorted(names):
        column = []
        for member, values in zip(loaded, per_member):
            if name not in values:
                raise ValueError(f"Run {member.label} has no {split} {name} metric.")
            column.append(values[name])
        summaries[name] = _summarize(column)
    return summaries


def _load_all(runs: Iterable[RunArtifacts | str | Path]) -> tuple[RunArtifacts, ...]:
    return tuple(run if isinstance(run, RunArtifacts) else load_run(run) for run in runs)


def shared_history(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "val",
) -> tuple[ComparisonSeries, ...]:
    """Select one shared metric; raw per-model objectives are rejected."""
    if metric == "objective":
        raise ValueError(
            "Raw objectives are model-specific and cannot be overlaid directly."
        )
    loaded = _load_all(runs)
    if len(loaded) < 2:
        raise ValueError("At least two runs are required for a comparison.")
    aliases = _split_aliases(split)
    selected: list[ComparisonSeries] = []
    for run in loaded:
        events = tuple(
            event
            for event in run.history
            if event.metric == metric and event.split in aliases
        )
        if not events:
            raise ValueError(
                f"Run {run.label} has no {split} {metric} history to compare."
            )
        summary = None
        if len(run.seed_members) > 1:
            summary = aggregate_history(run.seed_members, metric=metric, split=split)
        selected.append(ComparisonSeries(run.label, events, summary))
    return tuple(selected)


def final_metric_summaries(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "test",
) -> dict[str, MetricSummary]:
    """Return mean/std of a shared scalar metric across each run's members."""
    compared: dict[str, MetricSummary] = {}
    for run in _load_all(runs):
        compared[run.label] = aggregate_metrics(run.seed_members, split=split, metric=metric)[metric]
    return compared


def final_metrics(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "test",
) -> dict[str, float]:
    """Return a shared scalar metric (mean across seeds) from each run."""
    return {
        label: summary.mean
        for label, summary in final_metric_summaries(runs, metric=metric, split=split).items()
    }


def comparison_records(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str | None = None,
    split: str | None = "test",
) -> tuple[ComparisonRecord, ...]:
    """Return artifact metrics in one model/problem-neutral tidy schema.

    Multi-seed runs contribute one record per member, sharing the parent
    ``run_id`` and differing by ``seed``.
    """
    records: list[ComparisonRecord] = []
    for run in _load_all(runs):
        for member in run.seed_members:
            for row in member.metrics:
                row_metric = str(row["metric"])
                row_split = str(row["split"])
                if metric is not None and row_metric != metric:
                    continue
                if split is not None and row_split != split:
                    continue
                records.append(
                    ComparisonRecord(
                        problem=str(run.manifest["problem_id"]),
                        model=str(run.manifest["model_id"]),
                        run_id=run.path.name,
                        benchmark_protocol=str(run.manifest["benchmark_protocol_id"]),
                        split=row_split,
                        metric=row_metric,
                        value=float(row["value"]),
                        seed=member.seed,
                    )
                )
    return tuple(records)


def metric_summary_matrix(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "test",
) -> dict[str, dict[str, MetricSummary]]:
    """Arrange a metric summary as problem rows and model columns.

    Seeds of one run aggregate into a cell; two distinct runs for the same
    problem/model are rejected so callers cannot silently mix protocol
    versions. Select the runs explicitly first.
    """
    cells: dict[tuple[str, str], tuple[str, list[float]]] = {}
    for record in comparison_records(runs, metric=metric, split=split):
        key = (record.problem, record.model)
        run_id, values = cells.setdefault(key, (record.run_id, []))
        if run_id != record.run_id:
            raise ValueError(
                f"Multiple runs populate {record.problem}/{record.model}; "
                "select one run per matrix cell."
            )
        values.append(record.value)
    if not cells:
        raise ValueError(f"No {split} {metric} values were found.")
    matrix: dict[str, dict[str, MetricSummary]] = {}
    for (problem, model), (_, values) in cells.items():
        matrix.setdefault(problem, {})[model] = _summarize(values)
    return matrix


def metric_matrix(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "test",
) -> dict[str, dict[str, float]]:
    """Arrange a metric (mean across seeds) as problem rows and model columns."""
    return {
        problem: {model: summary.mean for model, summary in row.items()}
        for problem, row in metric_summary_matrix(runs, metric=metric, split=split).items()
    }


def validate_model_comparison(
    runs: Iterable[RunArtifacts | str | Path],
) -> tuple[RunArtifacts, ...]:
    """Validate that runs differ only by model for a fair within-problem test."""
    loaded = _load_all(runs)
    if len(loaded) < 2:
        raise ValueError("At least two runs are required for a comparison.")
    first = loaded[0].manifest
    for run in loaded[1:]:
        mismatches = [
            name
            for name in COMPARABLE_MANIFEST_FIELDS
            if run.manifest.get(name) != first.get(name)
        ]
        if mismatches:
            raise ValueError(
                f"Run {run.label} is not directly comparable; mismatched "
                + ", ".join(mismatches)
                + "."
            )
    return loaded


def _read_history(path: Path) -> list[MetricEvent]:
    return [
        MetricEvent(
            epoch=int(row["epoch"]),
            split=row["split"],
            metric=row["metric"],
            value=float(row["value"]),
        )
        for row in _read_csv(path)
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_reconstructions(path: str | Path) -> dict[str, Any]:
    """Load reconstruction arrays only when a plotting/analysis script asks."""
    import numpy as np

    run_path = Path(path)
    target = run_path / "reconstructions.npz" if run_path.is_dir() else run_path
    with np.load(target, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def read_best_config(path: str | Path) -> dict[str, Any]:
    """Read a run's best configuration with the standard clear failure mode."""
    target = Path(path)
    if target.is_dir():
        return ArtifactStore(target).read_best_config()
    if not target.is_file():
        raise FileNotFoundError(
            f"Best configuration not found at {target}. Run or resume tuning first."
        )
    with target.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict) or not config:
        raise ValueError("Best configuration must be a non-empty JSON object.")
    return config
