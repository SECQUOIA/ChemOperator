"""Plot saved training artifacts without loading datasets or models."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MODEL_LABELS = {
    "deeponet": "DeepONet",
    "pod_deeponet": "POD-DeepONet",
}
MODEL_STYLES = {
    "deeponet": ("tab:blue", "--"),
    "pod_deeponet": ("tab:orange", ":"),
}


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
    import matplotlib.pyplot as plt

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
            linestyle="-" if split == "train" else "--",
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
    import matplotlib.pyplot as plt

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
        figsize=(5.2 * n_cases, 2.6 * len(selected_labels)),
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


__all__ = ["plot_deeponet_artifacts"]
