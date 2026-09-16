"""Atomic persistence and validation for versioned experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO, Iterator

import numpy as np
import torch

from .device import hardware_metadata
from .metrics import MetricEvent
from .types import RunContext, RunPaths, TrainingOutcome

ARTIFACT_SCHEMA_VERSION = 1
MANIFEST_STATUSES = frozenset({"running", "completed", "failed"})
REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "artifact_schema_version",
        "problem_id",
        "model_id",
        "benchmark_protocol_id",
        "git_commit",
        "dirty_worktree",
        "dataset_fingerprints",
        "fields",
        "channels",
        "units",
        "coordinates",
        "selected_test_case_ids",
        "random_seed",
        "device",
        "precision",
        "hardware",
        "dependency_lock_hash",
        "parameter_count",
        "tuning_budget",
        "timings",
        "status",
    }
)
REQUIRED_TIMINGS = frozenset(
    {
        "data_generation_seconds",
        "tuning_seconds",
        "final_training_seconds",
        "inference_seconds",
    }
)


class ArtifactValidationError(ValueError):
    """Raised when an artifact would violate the run schema."""


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable.")


def _validate_json(payload: Any) -> None:
    try:
        json.dumps(payload, allow_nan=False, default=_json_default)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"Artifact is not valid JSON: {exc}") from exc


@contextmanager
def _atomic_file(
    destination: Path,
    *,
    mode: str,
    encoding: str | None = None,
    newline: str | None = None,
) -> Iterator[IO[Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(
            descriptor,
            mode,
            encoding=encoding,
            newline=newline,
        ) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fingerprint_path(path: str | Path) -> str:
    """Return a deterministic SHA-256 fingerprint for a file or directory."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    files = [source] if source.is_file() else sorted(
        item for item in source.rglob("*") if item.is_file()
    )
    for item in files:
        relative = item.name if source.is_file() else item.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def dependency_lock_hash(root: str | Path) -> str | None:
    """Fingerprint the first recognized dependency lock in a project root."""
    directory = Path(root)
    for name in ("uv.lock", "poetry.lock", "Pipfile.lock", "pixi.lock"):
        candidate = directory / name
        if candidate.is_file():
            return fingerprint_path(candidate)
    return None


def git_provenance(root: str | Path) -> tuple[str | None, bool | None]:
    """Return commit and dirty status, or ``(None, None)`` outside Git."""
    directory = Path(root)
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None
    return commit, bool(status.strip())


def build_manifest(  # pylint: disable=too-many-arguments
    context: RunContext,
    *,
    problem_id: str,
    model_id: str,
    benchmark_protocol_id: str,
    dataset_fingerprints: Mapping[str, str | None],
    fields: Sequence[str],
    channels: Sequence[str] | Mapping[str, Sequence[str]],
    units: Mapping[str, str],
    coordinates: Sequence[str] | Mapping[str, Any],
    selected_test_case_ids: Sequence[str | int],
    parameter_count: int,
    tuning_budget: Mapping[str, Any],
    project_root: str | Path | None = None,
    timings: Mapping[str, float | None] | None = None,
    status: str = "running",
) -> dict[str, Any]:
    """Build a complete manifest from explicit scientific metadata."""
    root = Path.cwd() if project_root is None else Path(project_root)
    commit, dirty = git_provenance(root)
    all_timings = dict.fromkeys(REQUIRED_TIMINGS)
    if timings is not None:
        all_timings.update(timings)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "problem_id": problem_id,
        "model_id": model_id,
        "benchmark_protocol_id": benchmark_protocol_id,
        "git_commit": commit,
        "dirty_worktree": dirty,
        "dataset_fingerprints": dict(dataset_fingerprints),
        "fields": list(fields),
        "channels": channels,
        "units": dict(units),
        "coordinates": coordinates,
        "selected_test_case_ids": list(selected_test_case_ids),
        "random_seed": context.seed,
        "device": str(context.device),
        "precision": str(context.dtype).removeprefix("torch."),
        "hardware": hardware_metadata(context.device),
        "dependency_lock_hash": dependency_lock_hash(root),
        "parameter_count": int(parameter_count),
        "tuning_budget": dict(tuning_budget),
        "timings": all_timings,
        "status": status,
        "failure": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": dict(context.provenance),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate required manifest fields and finite timing/count values."""
    missing = REQUIRED_MANIFEST_FIELDS - manifest.keys()
    if missing:
        raise ArtifactValidationError(
            "Manifest is missing required fields: " + ", ".join(sorted(missing))
        )
    if manifest["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"Unsupported artifact schema {manifest['artifact_schema_version']!r}."
        )
    if manifest["status"] not in MANIFEST_STATUSES:
        raise ArtifactValidationError(f"Unknown run status {manifest['status']!r}.")
    fingerprints = manifest["dataset_fingerprints"]
    if not isinstance(fingerprints, Mapping) or not {"train", "validation", "test"} <= set(
        fingerprints
    ):
        raise ArtifactValidationError(
            "dataset_fingerprints must contain train, validation, and test."
        )
    timings = manifest["timings"]
    if not isinstance(timings, Mapping) or not REQUIRED_TIMINGS <= set(timings):
        raise ArtifactValidationError(
            "timings must contain data-generation, tuning, final-training, and "
            "inference durations."
        )
    if int(manifest["parameter_count"]) < 0:
        raise ArtifactValidationError("parameter_count cannot be negative.")
    _validate_json(manifest)


class ArtifactStore:
    """Write and validate one canonical experiment run directory."""

    def __init__(self, paths: RunPaths | RunContext | str | Path) -> None:
        if isinstance(paths, RunContext):
            self.paths = paths.paths
        elif isinstance(paths, RunPaths):
            self.paths = paths
        else:
            self.paths = RunPaths(Path(paths))

    def write_json(self, path: str | Path, payload: Any) -> Path:
        _validate_json(payload)
        destination = self._within_run(path)
        with _atomic_file(destination, mode="w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False, default=_json_default)
            stream.write("\n")
        return destination

    def write_manifest(self, manifest: Mapping[str, Any]) -> Path:
        validate_manifest(manifest)
        return self.write_json(self.paths.manifest, dict(manifest))

    def read_manifest(self) -> dict[str, Any]:
        with self.paths.manifest.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        validate_manifest(manifest)
        return manifest

    def update_manifest(self, **updates: Any) -> Path:
        manifest = self.read_manifest()
        if "timings" in updates:
            manifest["timings"] = {
                **manifest["timings"],
                **dict(updates.pop("timings")),
            }
        manifest.update(updates)
        validate_manifest(manifest)
        return self.write_manifest(manifest)

    def write_best_config(self, config: Mapping[str, Any]) -> Path:
        if not config:
            raise ArtifactValidationError("best_config.json cannot be empty.")
        return self.write_json(self.paths.best_config, dict(config))

    def read_best_config(self) -> dict[str, Any]:
        if not self.paths.best_config.is_file():
            raise FileNotFoundError(
                f"Best configuration not found at {self.paths.best_config}. "
                "Run or resume tuning before requesting --train-config best."
            )
        with self.paths.best_config.open(encoding="utf-8") as stream:
            config = json.load(stream)
        if not isinstance(config, dict) or not config:
            raise ArtifactValidationError("best_config.json must be a non-empty object.")
        return config

    def write_history(self, events: Iterable[MetricEvent | Mapping[str, Any]]) -> Path:
        rows = [
            event.to_dict() if isinstance(event, MetricEvent) else dict(event)
            for event in events
        ]
        fieldnames = ("epoch", "split", "metric", "value")
        required = set(fieldnames)
        for row in rows:
            if set(row) != required:
                raise ArtifactValidationError(
                    "History rows must contain exactly epoch, split, metric, value."
                )
            MetricEvent(
                epoch=int(row["epoch"]),
                split=str(row["split"]),
                metric=str(row["metric"]),
                value=float(row["value"]),
            )
        return self._write_csv(self.paths.history, fieldnames, rows)

    def write_metrics(self, metrics: Mapping[str, float] | Iterable[Mapping[str, Any]]) -> Path:
        if isinstance(metrics, Mapping):
            rows = [
                {"split": "test", "metric": name, "value": float(value)}
                for name, value in metrics.items()
            ]
        else:
            rows = [dict(row) for row in metrics]
        required = {"split", "metric", "value"}
        for row in rows:
            if not required <= set(row):
                raise ArtifactValidationError(
                    "Metric rows must contain split, metric, and value."
                )
            value = float(row["value"])
            if not np.isfinite(value):
                raise ArtifactValidationError("Metric values must be finite.")
        fieldnames = _ordered_fieldnames(rows, ("model", "split", "metric", "value"))
        return self._write_csv(self.paths.metrics, fieldnames, rows)

    def write_tuning_trials(self, trials: Iterable[Mapping[str, Any]]) -> Path:
        rows = [dict(row) for row in trials]
        fieldnames = _ordered_fieldnames(
            rows,
            ("trial_id", "status", "metric", "value", "training_iteration"),
        )
        if not fieldnames:
            fieldnames = ("trial_id", "status", "metric", "value")
        return self._write_csv(self.paths.tuning_trials, fieldnames, rows)

    def write_reconstructions(self, **arrays: Any) -> Path:
        if not arrays:
            raise ArtifactValidationError("reconstructions.npz cannot be empty.")
        destination = self.paths.reconstructions
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _atomic_file(destination, mode="wb") as stream:
            np.savez_compressed(stream, **arrays)
        return destination

    def write_torch_checkpoint(
        self,
        state: Mapping[str, Any],
        *,
        suffix: str = ".pt",
    ) -> Path:
        destination = self.paths.checkpoint(suffix)
        with _atomic_file(destination, mode="wb") as stream:
            torch.save(dict(state), stream)
        return destination

    def write_checkpoint(
        self,
        writer: Callable[[Path], Any],
        *,
        suffix: str = ".pt",
    ) -> Path:
        """Atomically publish a checkpoint produced by a trainer callback."""
        destination = self.paths.checkpoint(suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=suffix,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            writer(temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ArtifactValidationError("Checkpoint writer produced no data.")
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return destination

    def copy_checkpoint(self, source: str | Path, *, suffix: str | None = None) -> Path:
        checkpoint = Path(source)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        resolved_suffix = checkpoint.suffix if suffix is None else suffix
        return self.write_checkpoint(
            lambda temporary: shutil.copyfile(checkpoint, temporary),
            suffix=resolved_suffix,
        )

    def write_outcome(self, outcome: TrainingOutcome) -> None:
        self.write_history(outcome.history)
        self.write_metrics(outcome.metrics)

    def mark_completed(self, *, timings: Mapping[str, float] | None = None) -> None:
        self.update_manifest(
            status="completed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            timings={} if timings is None else timings,
        )
        self.validate_complete()

    def mark_failed(self, error: BaseException) -> None:
        self.update_manifest(
            status="failed",
            failure={"type": type(error).__name__, "message": str(error)},
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

    def validate_complete(self) -> None:
        manifest = self.read_manifest()
        if manifest["status"] != "completed":
            raise ArtifactValidationError("Run manifest is not completed.")
        required = (
            self.paths.best_config,
            self.paths.history,
            self.paths.metrics,
            self.paths.reconstructions,
            self.paths.tuning_trials,
        )
        missing = [path.name for path in required if not path.is_file()]
        checkpoints = tuple(self.paths.run_dir.glob("checkpoint.*"))
        if not checkpoints:
            missing.append("checkpoint.*")
        if missing:
            raise ArtifactValidationError(
                "Completed run is missing artifacts: " + ", ".join(missing)
            )

    def _within_run(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.paths.run_dir / candidate
        resolved_run = self.paths.run_dir.resolve()
        resolved = candidate.resolve()
        if resolved != resolved_run and resolved_run not in resolved.parents:
            raise ValueError(f"Artifact path escapes the run directory: {path}")
        return candidate

    @staticmethod
    def _write_csv(
        destination: Path,
        fieldnames: Iterable[str],
        rows: Iterable[Mapping[str, Any]],
    ) -> Path:
        names = tuple(fieldnames)
        with _atomic_file(
            destination,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=names)
            writer.writeheader()
            writer.writerows(rows)
        return destination


def _ordered_fieldnames(
    rows: Sequence[Mapping[str, Any]],
    preferred: Sequence[str],
) -> tuple[str, ...]:
    available = {name for row in rows for name in row}
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(available - set(ordered)))
    return tuple(ordered)
