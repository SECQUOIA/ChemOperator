"""Train direct and IPCA-POD DeepONets for non-isothermal CSTR trajectories."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cantera as ct
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from chem_operator.models import (
    CoordinateScaler,
    DeepONetBenchmarkConfig,
    DeepONetTrainingHistory,
    DeepXDEAdapter,
    PODTransform,
    deeponet_parameter_counts,
    fit_incremental_pod_dataset,
    fit_zscore_normalizer,
    make_deeponet_dataloader,
    train_deeponet_lazy,
)
from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    TargetTransformConfig,
)
from chem_operator.example_paths import ExamplePaths
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="cstr")
DATASET_NAME = "cstr"
DATASET_NAME = "cstr_non_isothermal"
MECHANISM = "n-heptane-NUIG-2016.yaml"
FIELDS = ("T", "P", "X")
CONSTANTS = ("heat_transfer_coefficient",)

EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
WIDTH = 512
LATENT_WIDTH = 64
DISPLAY_EVERY = 10
COORDINATE_STRIDE = 5
POD_VARIANCE_THRESHOLD = 0.9995
SEED = 42
PLOT_CASES = 2
MAX_TRAJECTORIES: int | None = None
DATALOADER_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


def raw_dataset(split: str) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        PATHS.data / f"{DATASET_NAME}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="t",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        index_stride=COORDINATE_STRIDE,
        dtype=torch.float32,
    )


def limited(dataset: Dataset) -> Dataset:
    if MAX_TRAJECTORIES is None:
        return dataset
    return Subset(dataset, range(min(MAX_TRAJECTORIES, len(dataset))))


def processor(normalizer: ZScoreNormalizer) -> DataProcessor:
    return DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=FIELDS,
            constant_field_order=CONSTANTS,
        ),
        normalizer=normalizer,
        normalization_config=NormalizationConfig(enabled=True),
        target_transform=TargetTransformConfig(mode="state"),
    )


def adapter(dataset: Dataset, normalizer: ZScoreNormalizer) -> DeepXDEAdapter:
    return DeepXDEAdapter(
        dataset,
        processor(normalizer),
        format="cartesian_product",
        coordinate_name="t",
        include_constants=True,
    )


def benchmark_config() -> DeepONetBenchmarkConfig:
    return DeepONetBenchmarkConfig(
        loss="relative_l2",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        width=WIDTH,
        latent_width=LATENT_WIDTH,
        display_every=DISPLAY_EVERY,
        variance_threshold=POD_VARIANCE_THRESHOLD,
        seed=SEED,
        plot_cases=PLOT_CASES,
    )


def loss_curve(values: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray([np.sum(value) for value in values], dtype=float)


def plot_losses(
    histories: Mapping[str, DeepONetTrainingHistory],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {"DeepONet": "tab:blue", "POD-DeepONet": "tab:orange"}
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, history in histories.items():
        axis.semilogy(
            history.steps,
            loss_curve(history.loss_train),
            color=colors[name],
            label=f"{name} train",
        )
        axis.semilogy(
            history.steps,
            loss_curve(history.loss_test),
            color=colors[name],
            linestyle="--",
            label=f"{name} validation",
        )
    axis.set(xlabel="Epoch", ylabel="Relative L2 loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "training_validation_loss.png", dpi=200)
    plt.close(figure)


def plot_reconstructions(
    coordinates: Sequence[np.ndarray],
    reference: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    labels: Sequence[str],
    selected_labels: Sequence[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    label_to_index = {label: index for index, label in enumerate(labels)}
    missing = [label for label in selected_labels if label not in label_to_index]
    if missing:
        raise KeyError("Unknown plot labels: " + ", ".join(missing))

    n_cases = min(PLOT_CASES, reference.shape[0])
    figure, axes = plt.subplots(
        len(selected_labels),
        n_cases,
        figsize=(5.2 * n_cases, 2.6 * len(selected_labels)),
        squeeze=False,
        sharex="col",
    )
    styles = {
        "DeepONet": ("tab:blue", "--"),
        "POD-DeepONet": ("tab:orange", ":"),
    }
    for column in range(n_cases):
        for row, label in enumerate(selected_labels):
            axis = axes[row, column]
            channel = label_to_index[label]
            axis.plot(
                coordinates[column],
                reference[column, :, channel],
                color="black",
                linewidth=1.8,
                label="Cantera",
            )
            for name, values in predictions.items():
                color, linestyle = styles[name]
                axis.plot(
                    coordinates[column],
                    values[column, :, channel],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=name,
                )
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(f"Test trajectory {column + 1}")
            if row == len(selected_labels) - 1:
                axis.set_xlabel("Time [s]")
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "test_reconstructions.png", dpi=200)
    plt.close(figure)


def evaluate_models(
    direct_model: torch.nn.Module,
    pod_model: torch.nn.Module,
    pod: PODTransform,
    test: DeepXDEAdapter,
    normalizer: ZScoreNormalizer,
    coordinate_scaler: CoordinateScaler,
) -> tuple[
    dict[str, dict[str, float]],
    list[np.ndarray],
    np.ndarray,
    dict[str, np.ndarray],
    tuple[str, ...],
]:
    loader = make_deeponet_dataloader(
        test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        coordinate_scaler=coordinate_scaler,
        num_workers=DATALOADER_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=SEED,
    )
    direct_device = next(direct_model.parameters()).device
    pod_device = next(pod_model.parameters()).device
    direct_model.eval()
    pod_model.eval()

    totals = {
        name: {
            "count": 0,
            "squared_error": 0.0,
            "reference_squared": 0.0,
            "max_absolute_error": 0.0,
        }
        for name in ("DeepONet", "POD-DeepONet")
    }
    plot_coordinates: list[np.ndarray] = []
    plot_reference: list[np.ndarray] = []
    plot_predictions: dict[str, list[np.ndarray]] = {
        "DeepONet": [],
        "POD-DeepONet": [],
    }
    labels: tuple[str, ...] | None = None

    with torch.no_grad():
        for batch in loader:
            branch = batch["branch"]
            trunk = batch["trunk"]
            target = batch["target"]

            direct_normalized = direct_model(
                (branch.to(direct_device), trunk.to(direct_device))
            ).cpu()
            if direct_normalized.ndim + 1 == target.ndim:
                direct_normalized = direct_normalized.unsqueeze(-1)

            pod_flattened = pod_model(
                (branch.to(pod_device), trunk.to(pod_device))
            )
            pod_normalized = pod.unflatten_tensor(pod_flattened).cpu()

            reference = normalizer.denormalize_flattened(target, "variable")
            predictions = {
                "DeepONet": normalizer.denormalize_flattened(
                    direct_normalized, "variable"
                ),
                "POD-DeepONet": normalizer.denormalize_flattened(
                    pod_normalized, "variable"
                ),
            }
            reference64 = reference.to(torch.float64)
            reference_squared = float(torch.sum(reference64**2))
            for name, prediction in predictions.items():
                error = prediction.to(torch.float64) - reference64
                totals[name]["count"] += error.numel()
                totals[name]["squared_error"] += float(torch.sum(error**2))
                totals[name]["reference_squared"] += reference_squared
                totals[name]["max_absolute_error"] = max(
                    totals[name]["max_absolute_error"],
                    float(torch.max(torch.abs(error))),
                )

            labels = tuple(batch["labels"])
            for index, coordinate in enumerate(batch["coordinates"]):
                if len(plot_reference) >= PLOT_CASES:
                    break
                plot_coordinates.append(coordinate.detach().cpu().numpy())
                plot_reference.append(reference[index].numpy())
                for name, prediction in predictions.items():
                    plot_predictions[name].append(prediction[index].numpy())

    if labels is None or not plot_reference:
        raise RuntimeError("The test loader produced no trajectories.")

    metrics = {
        name: {
            "rmse": float(
                np.sqrt(values["squared_error"] / max(values["count"], 1))
            ),
            "relative_l2": float(
                np.sqrt(
                    values["squared_error"]
                    / max(values["reference_squared"], 1e-30)
                )
            ),
            "max_absolute_error": values["max_absolute_error"],
        }
        for name, values in totals.items()
    }
    return (
        metrics,
        plot_coordinates,
        np.stack(plot_reference),
        {name: np.stack(values) for name, values in plot_predictions.items()},
        labels,
    )


def main() -> None:
    PATHS.output.mkdir(parents=True, exist_ok=True)
    train_raw = raw_dataset("train")
    valid_raw = raw_dataset("valid")
    test_raw = raw_dataset("test")
    try:
        train_data = limited(train_raw)
        valid_data = limited(valid_raw)
        test_data = limited(test_raw)

        print("Fitting CSTR Z-score statistics from training trajectories ...")
        normalizer = fit_zscore_normalizer(train_data, FIELDS, CONSTANTS)
        train = adapter(train_data, normalizer)
        validation = adapter(valid_data, normalizer)
        test = adapter(test_data, normalizer)
        coordinate_scaler = CoordinateScaler.fit(train[0]["trunk"].numpy())

        print("Fitting incremental trajectory POD ...")
        pod = fit_incremental_pod_dataset(
            train,
            variance_threshold=POD_VARIANCE_THRESHOLD,
            num_workers=DATALOADER_WORKERS,
        )
        print(
            f"IPCA retained {pod.n_components} components for "
            f"{pod.cumulative_explained_variance:.6%} cumulative variance."
        )

        config = benchmark_config()
        print("Training DeepONet ...")
        direct_model, direct_history = train_deeponet_lazy(
            train,
            validation,
            config=config,
            coordinate_scaler=coordinate_scaler,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        direct_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Training POD-DeepONet ...")
        pod_model, pod_history = train_deeponet_lazy(
            train,
            validation,
            config=config,
            coordinate_scaler=coordinate_scaler,
            pod=pod,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        metrics, coordinates, reference, predictions, labels = evaluate_models(
            direct_model,
            pod_model,
            pod,
            test,
            normalizer,
            coordinate_scaler,
        )
        fuel_index = ct.Solution(MECHANISM).species_index("NC7H16")
        plot_losses(
            {"DeepONet": direct_history, "POD-DeepONet": pod_history},
            PATHS.output,
        )
        plot_reconstructions(
            coordinates,
            reference,
            predictions,
            labels,
            ("T", "P", f"X[{fuel_index}]"),
            PATHS.output,
        )

        summary = {
            "dataset": DATASET_NAME,
            "config": config.to_dict(),
            "parameter_counts": {
                "DeepONet": deeponet_parameter_counts(direct_model),
                "POD-DeepONet": deeponet_parameter_counts(pod_model),
            },
            "pod_components": pod.n_components,
            "pod_cumulative_explained_variance": (
                pod.cumulative_explained_variance
            ),
            "metrics": metrics,
        }
        np.savez(
            PATHS.output / "ipca_pod_matrix.npz",
            mean=pod.mean,
            basis=pod.basis,
            explained_variance_ratio=pod.explained_variance_ratio,
            output_shape=np.asarray(pod.output_shape, dtype=np.int64),
        )
        with (PATHS.output / "metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(summary, file, indent=2)
        print(f"Results written to {PATHS.output}")
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()


if __name__ == "__main__":
    main()
