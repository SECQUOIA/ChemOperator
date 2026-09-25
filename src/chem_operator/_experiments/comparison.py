"""Artifact-driven, plotting-independent comparison helpers."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ARTIFACT_SCHEMA_VERSION, ArtifactStore
from .metrics import MetricEvent


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Portable tabular artifacts loaded from one completed run."""

    path: Path
    manifest: Mapping[str, Any]
    history: Sequence[MetricEvent]
    metrics: Sequence[Mapping[str, Any]]

    @property
    def label(self) -> str:
        return (
            f"{self.manifest['problem_id']}/{self.manifest['model_id']}/"
            f"{self.path.name}"
        )


@dataclass(frozen=True, slots=True)
class ComparisonSeries:
    """One directly comparable metric series for a run."""

    label: str
    events: Sequence[MetricEvent]


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    """One tidy metric value addressable by problem, model, and run."""

    problem: str
    model: str
    run_id: str
    benchmark_protocol: str
    split: str
    metric: str
    value: float


def load_run(path: str | Path, *, require_complete: bool = True) -> RunArtifacts:
    """Load validated CSV/JSON artifacts without importing any model code."""
    store = ArtifactStore(path)
    manifest = store.read_manifest()
    if manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Cannot compare runs with an unsupported artifact schema.")
    if require_complete and manifest["status"] != "completed":
        raise ValueError(f"Run at {path} is not completed.")
    history = tuple(_read_history(store.paths.history))
    metrics = tuple(_read_csv(store.paths.metrics))
    return RunArtifacts(Path(path), manifest, history, metrics)


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
    loaded = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in runs
    )
    if len(loaded) < 2:
        raise ValueError("At least two runs are required for a comparison.")
    aliases = {split}
    if split == "val":
        aliases.add("validation")
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
        selected.append(ComparisonSeries(run.label, events))
    return tuple(selected)


def final_metrics(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "test",
) -> dict[str, float]:
    """Return a shared scalar metric from each run's ``metrics.csv``."""
    loaded = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in runs
    )
    compared: dict[str, float] = {}
    for run in loaded:
        matches = [
            row
            for row in run.metrics
            if row.get("metric") == metric and row.get("split") == split
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Run {run.label} must contain exactly one {split} {metric} metric."
            )
        compared[run.label] = float(matches[0]["value"])
    return compared


def comparison_records(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str | None = None,
    split: str | None = "test",
) -> tuple[ComparisonRecord, ...]:
    """Return artifact metrics in one model/problem-neutral tidy schema."""
    loaded = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in runs
    )
    records: list[ComparisonRecord] = []
    for run in loaded:
        for row in run.metrics:
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
                    benchmark_protocol=str(
                        run.manifest["benchmark_protocol_id"]
                    ),
                    split=row_split,
                    metric=row_metric,
                    value=float(row["value"]),
                )
            )
    return tuple(records)


def metric_matrix(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    metric: str = "relative_l2",
    split: str = "test",
) -> dict[str, dict[str, float]]:
    """Arrange a metric as problem rows and model columns.

    A duplicate problem/model cell is rejected so callers cannot silently mix
    repeated seeds or protocol versions. Select the runs explicitly first.
    """
    matrix: dict[str, dict[str, float]] = {}
    for record in comparison_records(runs, metric=metric, split=split):
        row = matrix.setdefault(record.problem, {})
        if record.model in row:
            raise ValueError(
                f"Multiple runs populate {record.problem}/{record.model}; "
                "select one run per matrix cell."
            )
        row[record.model] = record.value
    if not matrix:
        raise ValueError(f"No {split} {metric} values were found.")
    return matrix


def validate_model_comparison(
    runs: Iterable[RunArtifacts | str | Path],
) -> tuple[RunArtifacts, ...]:
    """Validate that runs differ only by model for a fair within-problem test."""
    loaded = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in runs
    )
    if len(loaded) < 2:
        raise ValueError("At least two runs are required for a comparison.")
    first = loaded[0].manifest
    comparable_fields = (
        "problem_id",
        "benchmark_protocol_id",
        "dataset_fingerprints",
        "fields",
        "channels",
        "units",
        "coordinates",
        "selected_test_case_ids",
    )
    for run in loaded[1:]:
        mismatches = [
            name
            for name in comparable_fields
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
