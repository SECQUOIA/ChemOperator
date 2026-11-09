"""Select and plot canonical runs for the problem-specific plot CLIs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import numpy as np

from chem_operator._experiments.comparison import (
    RunArtifacts, load_run, read_reconstructions, validate_model_comparison,
)
from chem_operator._experiments.artifacts import ArtifactStore
from chem_operator._plotting.plots import plot_experiment_histories, plot_operator_runs


def print_plot_paths(paths: Iterable[Path]) -> None:
    """Report saved plots with one path per line."""
    print("Plots written to:")
    for path in paths:
        print(path)


def add_plot_run_arguments(parser: argparse.ArgumentParser, *, root: Path) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run", type=Path, action="append",
                           help="Canonical run directory; repeat to plot multiple runs.")
    selection.add_argument("--recent", action="store_true",
                           help="Plot the newest completed run for every model.")
    parser.add_argument("--runs-root", type=Path, default=root / "artifacts" / "runs",
                        help="Root containing problem/model/run artifact directories.")


def _timestamp(manifest: dict) -> datetime:
    value = manifest.get("completed_at") or manifest.get("created_at")
    if not isinstance(value, str):
        raise ValueError("Missing run timestamp.")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Run timestamp must include a timezone.")
    return stamp


def _plot_ready(run: RunArtifacts, arrays: dict[str, np.ndarray]) -> bool:
    """Reject completed directories that cannot supply the generic run plots."""
    if not run.history or not run.metrics or "case_ids" not in arrays:
        return False
    if arrays["case_ids"].ndim != 1 or not len(arrays["case_ids"]):
        return False
    if "pfr_reference" in arrays:
        required = {"pfr_reference", "pfr_prediction", "pfr_labels",
                    "wall_reference", "wall_prediction", "wall_labels", "z", "r"}
        return required <= arrays.keys() and all(
            arrays[f"{prefix}_reference"].ndim >= 3
            and arrays[f"{prefix}_reference"].shape == arrays[f"{prefix}_prediction"].shape
            and arrays[f"{prefix}_reference"].shape[0] == len(arrays["case_ids"])
            and arrays[f"{prefix}_reference"].shape[1] == len(arrays[f"{prefix}_labels"])
            for prefix in ("pfr", "wall"))
    required = {"reference", "prediction", "labels"}
    if not required <= arrays.keys():
        return False
    reference = arrays["reference"]
    if reference.shape != arrays["prediction"].shape or reference.shape[0] != len(arrays["case_ids"]):
        return False
    if "coordinate_names" in arrays:
        names = tuple(str(name) for name in arrays["coordinate_names"])
        return (reference.ndim in (3, 4) and len(names) == reference.ndim - 2
                and reference.shape[1] == len(arrays["labels"])
                and all(name in arrays for name in names))
    return (reference.ndim == 3 and reference.shape[-1] == len(arrays["labels"])
            and "coordinates" in arrays
            and arrays["coordinates"].shape[:2] == reference.shape[:2]
            and bool(run.manifest.get("coordinates")))


def select_plot_runs(args: argparse.Namespace, parser: argparse.ArgumentParser,
                     *, problem_id: str) -> tuple[RunArtifacts, ...]:
    """Validate explicit paths or find the newest usable completed run per model."""
    if args.run:
        return tuple(load_run(path) for path in args.run)
    problem = args.runs_root / problem_id
    latest: list[tuple[datetime, RunArtifacts]] = []
    if problem.is_dir():
        for model_dir in sorted(problem.iterdir()):
            if not model_dir.is_dir():
                continue
            candidates = []
            for run_dir in model_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                try:
                    run = load_run(run_dir)
                    ArtifactStore(run_dir).validate_complete()
                    manifest = run.manifest
                    if (manifest["problem_id"] != problem_id
                            or manifest["model_id"] != model_dir.name):
                        continue
                    arrays = read_reconstructions(run_dir)
                    if not _plot_ready(run, arrays):
                        continue
                    candidates.append((_timestamp(manifest), run))
                except (OSError, ValueError, KeyError, TypeError, AttributeError,
                        IndexError, BadZipFile):
                    continue
            if candidates:
                latest.append(max(candidates, key=lambda item: (item[0], item[1].path.name)))
    if not latest:
        parser.error(f"No completed runs found under {problem}.")
    return tuple(run for _, run in latest)


def _same_array(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return bool(np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True))
    return bool(np.array_equal(a, b))


def _reference_arrays(run: RunArtifacts) -> dict[str, np.ndarray]:
    data = read_reconstructions(run.path)
    return {key: value for key, value in data.items()
            if key in {"case_ids", "reference", "coordinates", "coordinate_names",
                       "t", "r", "z", "labels"}
            or key.endswith(("_reference", "_labels"))}


def _comparable(a: RunArtifacts, b: RunArtifacts) -> bool:
    try:
        validate_model_comparison((a, b))
        left, right = _reference_arrays(a), _reference_arrays(b)
        return left.keys() == right.keys() and all(
            _same_array(left[key], right[key]) for key in left)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def comparison_groups(runs: tuple[RunArtifacts, ...]) -> tuple[tuple[RunArtifacts, ...], ...]:
    groups: list[list[RunArtifacts]] = []
    for run in runs:
        for group in groups:
            if all(_comparable(run, member) for member in group):
                group.append(run)
                break
        else:
            groups.append([run])
    return tuple(tuple(group) for group in groups if len(group) > 1)


def _overlay(group: tuple[RunArtifacts, ...], destination: Path, cases: int) -> Path | None:
    """Overlay channel-last or channel-first one-dimensional reconstructions."""
    data = [read_reconstructions(run.path) for run in group]
    baseline = data[0]
    if "pfr_reference" in baseline:
        reference_key, prediction_key, labels_key = (
            "pfr_reference", "pfr_prediction", "pfr_labels")
        if "z" not in baseline:
            return None
        x = baseline["z"]
        channel_last = False
        channel_first = True
    else:
        reference_key, prediction_key, labels_key = "reference", "prediction", "labels"
        if not {reference_key, prediction_key, labels_key} <= baseline.keys():
            return None
        if "coordinates" in baseline:
            x = baseline["coordinates"]
        elif "coordinate_names" in baseline and len(baseline["coordinate_names"]) == 1:
            name = str(baseline["coordinate_names"][0])
            if name not in baseline:
                return None
            x = baseline[name]
        else:
            return None
        channel_last = baseline[reference_key].shape[-1] == len(baseline[labels_key])
        channel_first = baseline[reference_key].shape[1] == len(baseline[labels_key])
    reference = baseline[reference_key]
    if reference.ndim != 3 or cases < 1:
        return None
    if not (channel_last or channel_first):
        return None
    count = min(cases, len(reference))
    channels = len(baseline[labels_key])
    points = reference.shape[1] if channel_last else reference.shape[2]
    if count == 0 or x.shape[0] < count or x[0].size != points:
        return None
    if any(prediction_key not in sample or sample[prediction_key].shape != reference.shape
           for sample in data):
        return None
    fig, axes = plt.subplots(channels, count, squeeze=False,
                            figsize=(5 * count, 2.8 * channels))
    for column in range(count):
        coordinate = x[column].reshape(-1)
        for row, label in enumerate(baseline[labels_key]):
            axis = axes[row, column]
            actual = reference[column, :, row] if channel_last else reference[column, row]
            axis.plot(coordinate, actual, color="black", label="Reference")
            for run, sample in zip(group, data):
                prediction = sample[prediction_key]
                values = prediction[column, :, row] if channel_last else prediction[column, row]
                axis.plot(coordinate, values, label=str(run.manifest["model_id"]))
            axis.set_ylabel(str(label))
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"Case {baseline['case_ids'][column]}")
            if row == channels - 1:
                axis.set_xlabel("Coordinate")
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    fig.tight_layout()
    path = destination / "reconstruction_overlay.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_selected_runs(runs: tuple[RunArtifacts, ...], output_dir: Path,
                       *, cases: int) -> tuple[Path, ...]:
    saved: list[Path] = []
    for run in runs:
        destination = output_dir / str(run.manifest["model_id"]) / run.path.name
        saved.extend(plot_operator_runs((run.path,), destination, cases=cases))
    for index, group in enumerate(comparison_groups(runs), start=1):
        destination = output_dir / "comparisons" / f"group_{index}"
        saved.append(plot_experiment_histories(
            group, output=destination / "validation_history.png",
            metric="relative_l2", split="val"))
        overlay = _overlay(group, destination, cases)
        if overlay is not None:
            saved.append(overlay)
    return tuple(saved)
