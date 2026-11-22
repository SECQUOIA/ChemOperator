"""Plot saved training artifacts without loading datasets or models."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from chem_operator._experiments.comparison import (
    RunArtifacts,
    aggregate_history,
    final_metric_summaries,
    load_run,
    metric_matrix,
    read_reconstructions,
    read_tuning_progress,
    tuning_source_dir,
    validate_model_comparison,
)

MODEL_LABELS = {
    "deeponet": "DeepONet",
    "pod_deeponet": "POD-DeepONet",
    "fno": "FNO",
}
MODEL_STYLES = {
    "deeponet": ("tab:blue", "--"),
    "pod_deeponet": ("tab:orange", ":"),
    "fno": ("tab:green", "-."),
}

STYLE_PATH = Path(__file__).parent / "styles" / "publication.mplstyle"

def use_publication_style(*, use_tex: bool = False) -> None:
    """Apply the bundled style, optionally using an external LaTeX installation."""
    plt.style.use(STYLE_PATH)
    plt.rcParams["text.usetex"] = use_tex

@dataclass(frozen=True, slots=True)
class HistoryPanel:
    """Select and label history metrics for one shared plot panel.

    Metric patterns may use glob wildcards, such as ``"*_loss"`` to match
    loss terms. Set ``splits`` to ``None`` to include every recorded split.
    ``metrics_by_model`` selects equivalent quantities stored under different
    metric names; it overrides ``metrics`` for the specified model IDs.
    """

    metrics: tuple[str, ...]
    title: str
    splits: tuple[str, ...] | None = None
    yscale: str | None = None
    metrics_by_model: Mapping[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        patterns = (self.metrics, *(self.metrics_by_model or {}).values())
        if any(not metrics or any(not metric for metric in metrics) for metrics in patterns):
            raise ValueError("HistoryPanel metrics cannot be empty.")
        if not self.title:
            raise ValueError("HistoryPanel title cannot be empty.")
        if self.yscale not in {None, "auto", "linear", "log"}:
            raise ValueError(
                "HistoryPanel yscale must be None, 'auto', 'linear', or 'log'."
            )


def _matches_metric(metric: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(metric, pattern) for pattern in patterns)


def _is_validation_split(split: str) -> bool:
    return split in {"val", "valid", "validation"}


def _read_history(
    path: Path,
) -> dict[tuple[str, str, str], tuple[list[int], list[float]]]:
    grouped: dict[tuple[str, str, str], tuple[list[int], list[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected = {"model", "epoch", "split", "metric", "value"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"{path} does not use the DeepONet history schema.")
        mutable: defaultdict[
            tuple[str, str, str], tuple[list[int], list[float]]
        ] = defaultdict(lambda: ([], []))
        for row in reader:
            key = (row["model"], row["split"], row["metric"])
            mutable[key][0].append(int(row["epoch"]))
            mutable[key][1].append(float(row["value"]))
        grouped.update(mutable)
    if not grouped:
        raise ValueError(f"{path} contains no history rows.")
    return grouped


def _plot_history(artifact_dir: Path, output_dir: Path) -> Path:
    history = _read_history(artifact_dir / "history.csv")
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    metrics: set[str] = set()
    for (model_id, split, metric), (epochs, values) in history.items():
        if model_id not in MODEL_STYLES:
            raise ValueError(f"Unknown DeepONet model ID {model_id!r}.")
        if split not in {"train", "validation"}:
            raise ValueError(f"Unknown history split {split!r}.")
        color, _ = MODEL_STYLES[model_id]
        metrics.add(metric)
        axis.semilogy(
            epochs,
            values,
            color=color,
            linestyle="--" if _is_validation_split(split) else "-",
            label=f"{MODEL_LABELS[model_id]} {split}",
        )
    loss_labels = {
        "relative_l2": "Relative L2 loss",
        "mse": "Mean-squared loss",
    }
    ylabel = (
        loss_labels.get(next(iter(metrics)), "Loss")
        if len(metrics) == 1
        else "Loss"
    )
    axis.set(xlabel="Epoch", ylabel=ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    path = output_dir / "training_validation_loss.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _plot_reconstructions(
    artifact_dir: Path,
    output_dir: Path,
    *,
    selected_labels: Sequence[str],
    coordinate_label: str,
    cases: int,
) -> Path:
    if not selected_labels:
        raise ValueError("selected_labels cannot be empty.")
    if cases < 1:
        raise ValueError("cases must be positive.")
    with np.load(
        artifact_dir / "reconstructions.npz",
        allow_pickle=False,
    ) as data:
        required = {
            "coordinates",
            "reference",
            "predictions",
            "model_ids",
            "labels",
        }
        missing_keys = required.difference(data.files)
        if missing_keys:
            raise ValueError(
                "reconstructions.npz is missing: "
                + ", ".join(sorted(missing_keys))
            )
        coordinates = np.array(data["coordinates"], copy=True)
        reference = np.array(data["reference"], copy=True)
        predictions = np.array(data["predictions"], copy=True)
        model_ids = tuple(
            str(value) for value in np.asarray(data["model_ids"]).tolist()
        )
        labels = tuple(
            str(value) for value in np.asarray(data["labels"]).tolist()
        )
    if predictions.shape[0] != len(model_ids):
        raise ValueError("Prediction and model ID counts do not match.")
    unknown_models = [name for name in model_ids if name not in MODEL_STYLES]
    if unknown_models:
        raise ValueError(
            "Unknown DeepONet model IDs: " + ", ".join(unknown_models)
        )
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing_labels = [
        label for label in selected_labels if label not in label_to_index
    ]
    if missing_labels:
        raise KeyError("Unknown plot labels: " + ", ".join(missing_labels))
    n_cases = min(cases, reference.shape[0])
    figure, axes = plt.subplots(
        len(selected_labels),
        n_cases,
        figsize=(5.0 * n_cases, 2.4 * len(selected_labels)),
        squeeze=False,
        sharex="col",
    )
    for column in range(n_cases):
        coordinate = coordinates[column].reshape(-1)
        for row, label in enumerate(selected_labels):
            axis = axes[row, column]
            channel = label_to_index[label]
            axis.plot(
                coordinate,
                reference[column, :, channel],
                color="black",
                linewidth=1.8,
                label="Reference",
            )
            for model_index, model_id in enumerate(model_ids):
                color, linestyle = MODEL_STYLES[model_id]
                axis.plot(
                    coordinate,
                    predictions[model_index, column, :, channel],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=MODEL_LABELS[model_id],
                )
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"Test trajectory {column + 1}")
            if row == len(selected_labels) - 1:
                axis.set_xlabel(coordinate_label)
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    path = output_dir / "test_reconstructions.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_deeponet_artifacts(
    artifact_dir: str | Path,
    *,
    selected_labels: Sequence[str],
    coordinate_label: str,
    cases: int = 2,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Create loss and reconstruction figures from saved artifacts."""

    artifacts = Path(artifact_dir)
    output = Path(output_dir) if output_dir is not None else artifacts
    output.mkdir(parents=True, exist_ok=True)
    return (
        _plot_history(artifacts, output),
        _plot_reconstructions(
            artifacts,
            output,
            selected_labels=selected_labels,
            coordinate_label=coordinate_label,
            cases=cases,
        ),
    )


def _plot_deeponet_run_reconstructions(
    runs: Sequence[RunArtifacts],
    output: Path,
    *,
    selected_labels: Sequence[str],
    coordinate_label: str,
    cases: int,
) -> Path:
    if not selected_labels:
        raise ValueError("selected_labels cannot be empty.")
    if cases < 1:
        raise ValueError("cases must be positive.")
    samples = [read_reconstructions(run.best_member.path) for run in runs]
    required = {"case_ids", "coordinates", "reference", "prediction", "labels"}
    for run, sample in zip(runs, samples):
        missing = required.difference(sample)
        if missing:
            raise ValueError(
                f"Run {run.label} reconstructions are missing: "
                + ", ".join(sorted(missing))
            )
        reference = np.asarray(sample["reference"])
        prediction = np.asarray(sample["prediction"])
        coordinates = np.asarray(sample["coordinates"])
        case_ids = np.asarray(sample["case_ids"])
        labels = np.asarray(sample["labels"])
        if reference.ndim != 3:
            raise ValueError(f"Run {run.label} reference must be three-dimensional.")
        if prediction.shape != reference.shape:
            raise ValueError(
                f"Run {run.label} prediction and reference shapes differ."
            )
        if coordinates.shape[:2] != reference.shape[:2]:
            raise ValueError(
                f"Run {run.label} coordinate and reference shapes differ."
            )
        if case_ids.shape != (reference.shape[0],):
            raise ValueError(f"Run {run.label} case IDs do not match its cases.")
        if labels.shape != (reference.shape[-1],):
            raise ValueError(f"Run {run.label} labels do not match its channels.")
    baseline = samples[0]
    for run, sample in zip(runs[1:], samples[1:]):
        for key in ("case_ids", "labels"):
            if not np.array_equal(sample[key], baseline[key]):
                raise ValueError(
                    f"Run {run.label} has mismatched reconstruction {key}."
                )
        for key in ("coordinates", "reference"):
            if sample[key].shape != baseline[key].shape or not np.allclose(
                sample[key], baseline[key], rtol=1.0e-6, atol=1.0e-8
            ):
                raise ValueError(
                    f"Run {run.label} has mismatched reconstruction {key}."
                )

    labels = tuple(str(value) for value in np.asarray(baseline["labels"]).tolist())
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing_labels = [label for label in selected_labels if label not in label_to_index]
    if missing_labels:
        raise KeyError("Unknown plot labels: " + ", ".join(missing_labels))
    reference = np.asarray(baseline["reference"])
    coordinates = np.asarray(baseline["coordinates"])
    n_cases = min(cases, reference.shape[0])
    figure, axes = plt.subplots(
        len(selected_labels),
        n_cases,
        figsize=(5.0 * n_cases, 2.4 * len(selected_labels)),
        squeeze=False,
        sharex="col",
    )
    for column in range(n_cases):
        coordinate = coordinates[column].reshape(-1)
        for row, label in enumerate(selected_labels):
            axis = axes[row, column]
            channel = label_to_index[label]
            axis.plot(
                coordinate,
                reference[column, :, channel],
                color="black",
                linewidth=1.8,
                label="Reference",
            )
            for run, sample in zip(runs, samples):
                model_id = str(run.manifest["model_id"])
                color, linestyle = MODEL_STYLES[model_id]
                prediction = np.asarray(sample["prediction"])
                axis.plot(
                    coordinate,
                    prediction[column, :, channel],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=MODEL_LABELS[model_id],
                )
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row == 0:
                case_id = np.asarray(baseline["case_ids"])[column]
                axis.set_title(f"Test case {case_id}")
            if row == len(selected_labels) - 1:
                axis.set_xlabel(coordinate_label)
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    path = output / "test_reconstructions.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def plot_deeponet_runs(
    deeponet_run: str | Path,
    pod_deeponet_run: str | Path,
    *,
    selected_labels: Sequence[str],
    coordinate_label: str,
    output_dir: str | Path,
    cases: int = 2,
) -> tuple[Path, Path]:
    """Compare one canonical direct run with one canonical POD run."""

    runs = validate_model_comparison((deeponet_run, pod_deeponet_run))
    model_ids = tuple(str(run.manifest["model_id"]) for run in runs)
    if model_ids != ("deeponet", "pod_deeponet"):
        raise ValueError(
            "Expected deeponet_run and pod_deeponet_run in that order; got "
            + ", ".join(model_ids)
            + "."
        )
    return plot_deeponet_run_set(
        runs,
        selected_labels=selected_labels,
        coordinate_label=coordinate_label,
        output_dir=output_dir,
        cases=cases,
    )


def plot_deeponet_run_set(
    run_paths: Iterable[RunArtifacts | str | Path],
    *,
    selected_labels: Sequence[str],
    coordinate_label: str,
    output_dir: str | Path,
    cases: int = 2,
) -> tuple[Path, Path]:
    """Plot compatible CSTR operator runs from canonical artifacts."""

    runs = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in run_paths
    )
    if not runs:
        raise ValueError("At least one run is required.")
    if len(runs) > 1:
        validate_model_comparison(runs)
    model_ids = tuple(str(run.manifest["model_id"]) for run in runs)
    unknown = sorted(set(model_ids) - MODEL_STYLES.keys())
    if unknown:
        raise ValueError("Unsupported operator runs: " + ", ".join(unknown) + ".")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history = plot_experiment_histories(
        runs,
        output=output / "training_validation_loss.png",
        panels=(HistoryPanel(
            ("relative_l2",), "Validation mean per-case relative L2", ("val",),
            metrics_by_model={"fno": ("data_loss",)},
        ),),
    )
    reconstruction = _plot_deeponet_run_reconstructions(
        runs,
        output,
        selected_labels=selected_labels,
        coordinate_label=coordinate_label,
        cases=cases,
    )
    return history, reconstruction


def plot_experiment_histories(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    output: str | Path,
    metric: str | None = "relative_l2",
    split: str | None = "val",
    panels: Sequence[HistoryPanel] | None = None,
    yscale: str = "auto",
    band_alpha: float = 0.2,
) -> Path:
    """Plot configurable history metrics from one or more canonical runs.

    By default, overlay the selected ``metric`` and ``split`` across runs.
    Provide ``panels`` to select several metrics/splits and show them in
    separate panels. A panel's ``yscale`` overrides the function-level scale.
    ``auto`` uses a log scale only when every plotted value in that panel is
    positive. A multi-seed run draws the mean across its members with a
    shaded ±1 standard-deviation band (clipped to the members' range).
    """
    if yscale not in {"auto", "linear", "log"}:
        raise ValueError("yscale must be 'auto', 'linear', or 'log'.")
    loaded = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in runs
    )
    if not loaded:
        raise ValueError("At least one run is required to plot history.")
    if panels is None:
        if metric is None or split is None:
            raise ValueError("metric and split are required when panels are omitted.")
        panels = (HistoryPanel((metric,), f"{split} {metric}", (split,), yscale),)
    elif not panels:
        raise ValueError("At least one history panel is required.")
    if len(loaded) > 1 and any(
        _matches_metric("objective", metrics)
        for panel in panels
        for metrics in (panel.metrics, *(panel.metrics_by_model or {}).values())
    ):
        raise ValueError(
            "Raw objectives are model-specific and cannot be overlaid directly."
        )

    figure, axes = plt.subplots(
        1,
        len(panels),
        squeeze=False,
        figsize=(7.0 * len(panels), 4.5),
    )
    any_values = False
    for axis, panel in zip(axes[0], panels):
        panel_values: list[float] = []
        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        curve_colors: dict[tuple[int, str], str] = {}
        for run_index, run in enumerate(loaded):
            metrics = (panel.metrics_by_model or {}).get(
                str(run.manifest["model_id"]), panel.metrics
            )
            # Group every member's events by (split, metric); members that
            # lack a key simply do not contribute to that curve.
            grouped: dict[tuple[str, str], list[tuple[RunArtifacts, list]]] = {}
            for member in run.seed_members:
                member_series: defaultdict[tuple[str, str], list] = defaultdict(list)
                for event in _select_history_events(member.history, metrics, panel.splits):
                    member_series[(event.split, event.metric)].append(event)
                for key, series in member_series.items():
                    grouped.setdefault(key, []).append((member, series))
            if not grouped:
                continue
            for (event_split, event_metric), contributions in grouped.items():
                color_key = (run_index, event_metric)
                if color_key not in curve_colors:
                    curve_colors[color_key] = colors[len(curve_colors) % len(colors)]
                color = curve_colors[color_key]
                linestyle = "--" if _is_validation_split(event_split) else "-"
                label = (
                    f"{run.label} {event_split}/{event_metric}"
                    if len(loaded) > 1
                    else f"{event_split}/{event_metric}"
                )
                for _, series in contributions:
                    panel_values.extend(event.value for event in series)
                if len(contributions) == 1:
                    series = sorted(contributions[0][1], key=lambda event: event.epoch)
                    axis.plot(
                        [event.epoch for event in series],
                        [event.value for event in series],
                        color=color, linestyle=linestyle, label=label,
                    )
                    continue
                summary = aggregate_history(
                    [member for member, _ in contributions],
                    metric=event_metric, split=event_split,
                )
                epochs = np.asarray(summary.epochs)
                mean = np.asarray(summary.mean)
                std = np.asarray(summary.std)
                lower = np.maximum(mean - std, np.asarray(summary.minimum))
                upper = np.minimum(mean + std, np.asarray(summary.maximum))
                axis.plot(
                    epochs, mean, color=color, linestyle=linestyle,
                    label=f"{label} (mean ± std, n={summary.count})",
                )
                axis.fill_between(
                    epochs, lower, upper, color=color, alpha=band_alpha,
                    linewidth=0, label="_nolegend_",
                )
        if not panel_values:
            axis.set(xlabel="Epoch", title=panel.title)
            axis.text(
                0.5,
                0.5,
                "No matching history data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.grid(alpha=0.25)
            continue
        any_values = True
        effective_yscale = panel.yscale or yscale
        if effective_yscale == "auto":
            effective_yscale = (
                "log" if all(value > 0 for value in panel_values) else "linear"
            )
        if effective_yscale == "log" and any(value <= 0 for value in panel_values):
            raise ValueError(
                f"Cannot use a log y-scale for {panel.title}: values must be positive."
            )
        axis.set_yscale(effective_yscale)
        axis.set(xlabel="Epoch", ylabel=panel.title, title=panel.title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    if not any_values:
        plt.close(figure)
        raise ValueError("No history values matched the requested panels.")
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def _select_history_events(history, metrics, splits):
    """Filter history events by metric patterns and split aliases."""
    return [
        event
        for event in history
        if _matches_metric(event.metric, metrics)
        and (
            splits is None
            or event.split in splits
            or (event.split == "validation" and "val" in splits)
            or (event.split == "val" and "validation" in splits)
        )
    ]


def member_plot_dirs(run: RunArtifacts, destination: str | Path) -> tuple[tuple[RunArtifacts, Path], ...]:
    """Pair each seed member with its plot directory.

    The best member (lowest final validation loss, see
    ``best_validation_member``) is written at ``destination`` so existing
    figure paths stay stable; a multi-seed run additionally writes every
    member under ``destination/seeds/<seed>/``.
    """
    destination = Path(destination)
    pairs = [(run.best_member, destination)]
    if len(run.seed_members) > 1:
        pairs.extend(
            (member, destination / "seeds" / str(member.seed))
            for member in run.seed_members
        )
    return tuple(pairs)


def plot_experiment_matrix(
    runs: Iterable[RunArtifacts | str | Path],
    *,
    output: str | Path,
    metric: str = "relative_l2",
    split: str = "test",
) -> Path:
    """Plot problem-by-model test values from canonical metrics tables."""
    loaded = tuple(
        run if isinstance(run, RunArtifacts) else load_run(run)
        for run in runs
    )
    matrix = metric_matrix(loaded, metric=metric, split=split)
    problems = tuple(sorted(matrix))
    models = tuple(sorted({model for row in matrix.values() for model in row}))
    x = np.arange(len(problems), dtype=float)
    width = 0.8 / max(len(models), 1)
    figure, axis = plt.subplots(
        figsize=(max(7.0, 1.4 * len(problems)), 4.8)
    )
    for index, model in enumerate(models):
        values = [matrix[problem].get(model, np.nan) for problem in problems]
        axis.bar(
            x + (index - (len(models) - 1) / 2) * width,
            values,
            width=width,
            label=model,
        )
    axis.set_xticks(x, problems, rotation=20, ha="right")
    axis.set(ylabel=f"{split} {metric}")
    if all(
        value > 0
        for row in matrix.values()
        for value in row.values()
    ):
        axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def set_log_xscale(axis, values) -> None:
    """Use a log x-axis, falling back to symlog when ``values`` touch zero or go negative.

    ``values`` may be one array or a sequence of arrays (e.g. one per case).
    For symlog, the linear region ends at the power of ten at or below the
    smallest nonzero magnitude, so a grid starting at ``t = 0`` keeps its first
    point and the ``0`` tick sits a full decade from the first log tick.
    """
    arrays = values if isinstance(values, (list, tuple)) else [values]
    flat = np.concatenate([np.ravel(array) for array in arrays]) if arrays else np.empty(0)
    if flat.size and np.all(flat > 0.0):
        axis.set_xscale("log")
        return
    nonzero = np.abs(flat[flat != 0.0]).astype(np.float64)
    # A float32 threshold just under a decade makes matplotlib add a tick
    # one decade lower, right beside the 0 tick, so compute it in float64.
    linthresh = 10.0 ** math.floor(math.log10(nonzero.min())) if nonzero.size else 1.0e-12
    axis.set_xscale("symlog", linthresh=linthresh)


def plot_operator_fields(path, reference, prediction, labels, coordinates, coordinate_names,
                         *, xlog=False):
    """Draw channel-first one- or two-dimensional physical fields.

    ``xlog`` puts the first coordinate on a log (or symlog, when it includes
    zero) axis, which spreads out early transients in time-series fields.
    """
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.shape != prediction.shape or reference.shape[0] != len(labels):
        raise ValueError("Field shapes and labels do not agree.")
    dimensions = reference.ndim - 1
    if dimensions not in (1, 2) or len(coordinates) != dimensions:
        raise ValueError("Expected one or two coordinate axes.")
    # Display spatial axes in millimeters; stored coordinates stay in meters.
    axes_display = {"r": ("Radius [mm]", 1e3), "z": ("Axial position [mm]", 1e3), "t": ("Time [s]", 1.0)}
    names = [str(name) for name in coordinate_names]
    coordinate_names = [axes_display.get(name, (name, 1.0))[0] for name in names]
    coordinates = [np.asarray(values) * axes_display.get(name, (name, 1.0))[1]
                   for values, name in zip(coordinates, names)]
    if dimensions == 1:
        fig, axes = plt.subplots(len(labels), 1, squeeze=False, figsize=(8, 3 * len(labels)))
        for i, label in enumerate(labels):
            axes[i, 0].plot(coordinates[0], reference[i], label="Reference")
            axes[i, 0].plot(coordinates[0], prediction[i], "--", label="Prediction")
            axes[i, 0].set(xlabel=str(coordinate_names[0]), ylabel=str(label))
            axes[i, 0].legend()
    else:
        fig, axes = plt.subplots(len(labels), 3, squeeze=False, figsize=(14, 3.5 * len(labels)))
        for i, label in enumerate(labels):
            low = min(reference[i].min(), prediction[i].min())
            high = max(reference[i].max(), prediction[i].max())
            for j, (title, field) in enumerate((("Reference", reference[i]), ("Prediction", prediction[i]),
                                               ("Absolute error", np.abs(prediction[i] - reference[i])))):
                options = {"vmin": low, "vmax": high} if j < 2 else {"cmap": "magma"}
                artist = axes[i, j].pcolormesh(coordinates[0], coordinates[1], field.T, shading="auto", **options)
                axes[i, j].set(title=f"{label}: {title}", xlabel=str(coordinate_names[0]), ylabel=str(coordinate_names[1]))
                fig.colorbar(artist, ax=axes[i, j])
    if xlog:
        for axis in axes.flat:
            set_log_xscale(axis, coordinates[0])
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def select_channels(available: Sequence[str], labels: Sequence[str] | None) -> list[int]:
    """Return channel indices for ``labels`` (every channel when ``None``).

    Labels absent from ``available`` are ignored here; callers check that every
    requested label exists somewhere in the run.
    """
    available = [str(label) for label in available]
    if labels is None:
        return list(range(len(available)))
    return [available.index(str(label)) for label in labels if str(label) in available]


def _check_plot_labels(member: RunArtifacts, labels: Sequence[str] | None) -> None:
    if labels is None:
        return
    if not labels:
        raise ValueError("labels cannot be empty.")
    data = read_reconstructions(member.path)
    available = {str(label) for key, value in data.items()
                 if key == "labels" or key.endswith("_labels") for label in value}
    missing = [label for label in labels if str(label) not in available]
    if missing:
        raise ValueError(f"Labels not found in {member.path}: {missing}")


def plot_operator_runs(run_paths, output_dir, *, cases=2, labels=None, xlog=False):
    """Plot canonical runs using only their manifest, history, and NPZ arrays.

    ``labels`` restricts reconstruction figures to the named channels; by
    default every saved channel is drawn. ``xlog`` draws reconstructions on a
    log first-coordinate axis (see :func:`set_log_xscale`).
    """
    if cases < 1:
        raise ValueError("cases must be positive.")
    runs = tuple(run if isinstance(run, RunArtifacts) else load_run(run) for run in run_paths)
    if not runs:
        raise ValueError("At least one run is required.")
    if len(runs) > 1:
        validate_model_comparison(runs)
        baseline = read_reconstructions(runs[0].representative.path)
        for run in runs[1:]:
            other = read_reconstructions(run.representative.path)
            keys = [key for key in baseline if key in ("case_ids", "reference", "coordinates", "t", "r", "z", "labels")
                    or key.endswith("_reference") or key.endswith("_labels")]
            for key in keys:
                if key not in other or baseline[key].shape != other[key].shape:
                    raise ValueError(f"Compared runs have different reconstruction {key}.")
                equal = (np.allclose(baseline[key], other[key], rtol=1e-6, atol=1e-9)
                         if np.issubdtype(baseline[key].dtype, np.floating)
                         else np.array_equal(baseline[key], other[key]))
                if not equal:
                    raise ValueError(f"Compared runs have different reconstruction {key}.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for run in runs:
        destination = output if len(runs) == 1 else output / run.manifest["model_id"] / run.path.name
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "training_validation_loss.png"
        plot_experiment_histories(
            (run,),
            output=path,
            panels=(
                HistoryPanel(("relative_l2",), "Relative L2", yscale="auto"),
                HistoryPanel(("objective", "*_loss"), "Model objectives", yscale="auto"),
            ),
        )
        saved.append(path)
        for member, member_dir in member_plot_dirs(run, destination):
            _check_plot_labels(member, labels)
            saved.extend(_plot_member_reconstructions(member, member_dir, cases, labels,
                                                      xlog=xlog))
    if len(runs) > 1:
        fig, axis = plt.subplots(figsize=(8, 5))
        summaries = final_metric_summaries(runs)
        values = {label: summary.mean for label, summary in summaries.items()}
        axis.bar(range(len(values)), list(values.values()),
                 yerr=[summary.std for summary in summaries.values()], capsize=3)
        axis.set_xticks(range(len(values)), list(values), rotation=15, ha="right", fontsize=7)
        axis.set(ylabel="Test relative L2 (physical units)")
        fig.tight_layout()
        path = output / "comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
    return tuple(saved)


def _plot_member_reconstructions(member: RunArtifacts, destination: Path, cases: int,
                                 labels: Sequence[str] | None = None, *,
                                 xlog: bool = False) -> list[Path]:
    """Write the first ``cases`` test reconstructions of one seed member."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    data = read_reconstructions(member.path)
    count = min(cases, len(data["case_ids"]))
    for index in range(count):
        if "pfr_reference" in data:
            groups = (("pfr", ("z",)), ("wall", ("z", "r")))
            for prefix, axes in groups:
                channels = select_channels(data[f"{prefix}_labels"], labels)
                if not channels:
                    continue
                saved.append(plot_operator_fields(destination / f"case_{index:02d}_{prefix}.png",
                    data[f"{prefix}_reference"][index][channels], data[f"{prefix}_prediction"][index][channels],
                    data[f"{prefix}_labels"][channels], [data[axis][index] for axis in axes], axes,
                    xlog=xlog))
        else:
            if "coordinate_names" in data:
                axes = data["coordinate_names"].tolist()
                reference, prediction = data["reference"][index], data["prediction"][index]
                coordinates = [data[axis][index] for axis in axes]
            else:  # Canonical DeepONet arrays have channels last.
                axes = list(member.manifest["coordinates"])
                reference = np.moveaxis(data["reference"][index], -1, 0)
                prediction = np.moveaxis(data["prediction"][index], -1, 0)
                coordinates = [data["coordinates"][index].reshape(-1)]
            channels = select_channels(data["labels"], labels)
            saved.append(plot_operator_fields(destination / f"case_{index:02d}.png",
                reference[channels], prediction[channels], data["labels"][channels], coordinates, axes,
                xlog=xlog))
    return saved


TUNING_PROGRESS_METRIC = "valid_relative_l2"
TUNING_METRIC_LABELS = {
    "valid_relative_l2": r"Validation relative $L^2$",
    "best_valid_relative_l2": r"Best validation relative $L^2$",
}


def _tuning_metric(source: Path, rows: Sequence[Mapping[str, str]]) -> str:
    """Prefer the per-epoch validation error, else the study's tuning metric."""
    if any(row.get(TUNING_PROGRESS_METRIC) not in (None, "") for row in rows):
        return TUNING_PROGRESS_METRIC
    trials = source / "tuning_trials.csv"
    if trials.is_file():
        with trials.open(encoding="utf-8", newline="") as stream:
            first = next(csv.DictReader(stream), None)
        if first and first.get("metric"):
            return str(first["metric"])
    raise ValueError(
        f"{source} tuning progress has no {TUNING_PROGRESS_METRIC} and no tuning metric."
    )


def _finite(value: str | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def plot_tuning_progress(
    run_dir: str | Path,
    output: str | Path,
    *,
    metric: str | None = None,
) -> Path | None:
    """Plot every tuning trial's metric against wall-clock time on a log axis.

    Reads the study behind ``run_dir`` (see ``tuning_source_dir``), from the
    run folder's ``tuning_progress.csv`` or else Ray's ``progress.csv`` files.
    Returns ``None`` when no tuning progress can be found.
    """
    source = tuning_source_dir(run_dir)
    rows = None if source is None else read_tuning_progress(source)
    if not rows:
        return None
    assert source is not None
    metric = metric or _tuning_metric(source, rows)
    trials: dict[str, list[tuple[float, datetime, float]]] = defaultdict(list)
    for row in rows:
        stamp = _finite(row.get("timestamp"))
        value = _finite(row.get(metric))
        if stamp is None or value is None or value <= 0.0:
            continue
        iteration = _finite(row.get("training_iteration")) or 0.0
        trials[str(row["trial_id"])].append((iteration, datetime.fromtimestamp(stamp), value))
    if not trials:
        raise ValueError(f"{source} tuning progress has no positive finite {metric} values.")

    figure, axis = plt.subplots(figsize=(10.0, 5.0))
    for points in trials.values():
        points.sort(key=lambda point: (point[0], point[1]))
        axis.plot([point[1] for point in points], [point[2] for point in points],
                  linewidth=1.2)
    axis.set_yscale("log")
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", axis="y", alpha=0.15)
    stamps = [point[1] for points in trials.values() for point in points]
    # Hour-only ticks repeat on short studies, so add minutes below 12 hours.
    if (max(stamps) - min(stamps)).total_seconds() >= 12 * 3600:
        tick_format, tick_label = "%m-%d %H", "mm-dd hh"
    else:
        tick_format, tick_label = "%m-%d %H:%M", "mm-dd hh:mm"
    axis.xaxis.set_major_formatter(mdates.DateFormatter(tick_format))
    figure.autofmt_xdate(rotation=30)
    axis.set(xlabel=f"Wall time ({tick_label})",
             ylabel=TUNING_METRIC_LABELS.get(metric, metric.replace("_", " ")))
    figure.tight_layout()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination
