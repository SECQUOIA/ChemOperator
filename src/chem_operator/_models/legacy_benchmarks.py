"""Legacy DeepONet benchmark plotting and orchestration."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from chem_operator.normalization import ZScoreNormalizer
from chem_operator.utils import to_numpy

from .deeponet import (CoordinateScaler, DeepONetBenchmarkConfig,
                       DeepONetBenchmarkResult, _with_output_channel,
                       deeponet_parameter_counts, make_deeponet_dataloader)
from .deepxde import DeepXDEAdapter
from .pod import PODTransform, fit_incremental_pod_dataset
from .training import train_deeponet_lazy


def _loss_curve(history: Any, kind: Literal["train", "valid"]) -> np.ndarray:
    values = history.loss_train if kind == "train" else history.loss_test
    return np.asarray([np.sum(value) for value in values], dtype=float)


@dataclass
class _StreamingMetrics:
    count: int = 0
    squared_error: float = 0.0
    reference_squared: float = 0.0
    max_absolute_error: float = 0.0

    def update(self, reference: np.ndarray, prediction: np.ndarray) -> None:
        reference64 = reference.astype(np.float64, copy=False)
        error = prediction.astype(np.float64, copy=False) - reference64
        self.count += error.size
        self.squared_error += float(np.sum(error**2))
        self.reference_squared += float(np.sum(reference64**2))
        self.max_absolute_error = max(
            self.max_absolute_error,
            float(np.max(np.abs(error))),
        )

    def result(self) -> dict[str, float]:
        return {
            "rmse": float(
                np.sqrt(self.squared_error / max(self.count, 1))
            ),
            "relative_l2": float(
                np.sqrt(
                    self.squared_error
                    / max(self.reference_squared, 1e-30)
                )
            ),
            "max_absolute_error": self.max_absolute_error,
        }


def _plot_losses(histories: Mapping[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"DeepONet": "tab:blue", "POD-DeepONet": "tab:orange"}
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    for name, history in histories.items():
        steps = np.asarray(history.steps)
        axis.semilogy(
            steps,
            _loss_curve(history, "train"),
            color=colors[name],
            label=f"{name} train",
        )
        axis.semilogy(
            steps,
            _loss_curve(history, "valid"),
            color=colors[name],
            linestyle="--",
            label=f"{name} validation",
        )
    loss_names = {
        getattr(history, "loss_name", "loss")
        for history in histories.values()
    }
    if loss_names == {"relative_l2"}:
        loss_label = "Relative L2 loss"
    elif loss_names == {"mse"}:
        loss_label = "Mean-squared loss"
    else:
        loss_label = "Loss"
    axis.set(xlabel="Epoch / iteration", ylabel=loss_label)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "training_validation_loss.png", dpi=200)
    plt.close(figure)


def _plot_reconstructions(
    coordinates: Sequence[np.ndarray],
    reference: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    labels: Sequence[str],
    selected_labels: Sequence[str],
    coordinate_label: str,
    plot_cases: int,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    label_to_index = {label: index for index, label in enumerate(labels)}
    missing = [label for label in selected_labels if label not in label_to_index]
    if missing:
        raise KeyError("Unknown plot labels: " + ", ".join(missing))
    n_cases = min(plot_cases, reference.shape[0])
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
                axis.set_xlabel(coordinate_label)
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "test_reconstructions.png", dpi=200)
    plt.close(figure)


def run_deepxde_benchmark(
    train: DeepXDEAdapter,
    validation: DeepXDEAdapter,
    test: DeepXDEAdapter,
    normalizer: ZScoreNormalizer,
    *,
    output_dir: str | Path,
    plot_labels: Sequence[str],
    coordinate_label: str,
    config: DeepONetBenchmarkConfig | None = None,
    direct_config: DeepONetBenchmarkConfig | None = None,
    pod_config: DeepONetBenchmarkConfig | None = None,
    pod: PODTransform | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DeepONetBenchmarkResult:
    """Lazily train direct and state-POD DeepONets and write test plots."""

    if any(
        adapter.format != "cartesian_product"
        for adapter in (train, validation, test)
    ):
        raise ValueError(
            "The shared benchmark runner requires Cartesian-product adapters."
        )
    if config is not None and (direct_config is not None or pod_config is not None):
        raise ValueError("Use config or the two model-specific configs, not both.")
    base_config = config or DeepONetBenchmarkConfig()
    direct_config = direct_config or base_config
    pod_config = pod_config or base_config
    for model_config in (direct_config, pod_config):
        if model_config.epochs < 1 or model_config.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    first_train = train[0]
    scaler = CoordinateScaler.fit(to_numpy(first_train["trunk"]))
    if pod is None:
        pod = fit_incremental_pod_dataset(
            train,
            variance_threshold=pod_config.variance_threshold,
            num_workers=num_workers,
        )
    print(
        f"IPCA retained {pod.n_components} trajectory components for "
        f"{pod.cumulative_explained_variance:.6%} cumulative variance."
    )
    print("Training DeepONet ...")
    direct_model, direct_history = train_deeponet_lazy(
        train,
        validation,
        config=direct_config,
        coordinate_scaler=scaler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    # Keep only the actively trained model on the accelerator.
    direct_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Training POD-DeepONet ...")
    pod_model, pod_history = train_deeponet_lazy(
        train,
        validation,
        config=pod_config,
        coordinate_scaler=scaler,
        pod=pod,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = make_deeponet_dataloader(
        test,
        batch_size=max(direct_config.batch_size, pod_config.batch_size),
        shuffle=False,
        coordinate_scaler=scaler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=direct_config.seed,
    )
    direct_device = next(direct_model.parameters()).device
    pod_device = next(pod_model.parameters()).device
    direct_model.eval()
    pod_model.eval()
    metric_state = {
        "DeepONet": _StreamingMetrics(),
        "POD-DeepONet": _StreamingMetrics(),
    }
    plot_limit = max(direct_config.plot_cases, pod_config.plot_cases)
    plot_coordinates: list[np.ndarray] = []
    plot_reference: list[np.ndarray] = []
    plot_direct: list[np.ndarray] = []
    plot_pod: list[np.ndarray] = []
    labels: tuple[str, ...] | None = None
    with torch.no_grad():
        for batch in test_loader:
            branch = batch["branch"]
            trunk = batch["trunk"]
            direct_normalized = direct_model(
                (branch.to(direct_device), trunk.to(direct_device))
            )
            direct_normalized = _with_output_channel(
                direct_normalized,
                batch["target"].shape[-1],
            ).cpu()
            pod_flattened = pod_model(
                (branch.to(pod_device), trunk.to(pod_device))
            )
            pod_normalized = pod.unflatten_tensor(pod_flattened).cpu()
            reference = to_numpy(
                normalizer.denormalize_flattened(
                    batch["target"], "variable"
                )
            )
            direct_prediction = to_numpy(
                normalizer.denormalize_flattened(
                    direct_normalized, "variable"
                )
            )
            pod_prediction = to_numpy(
                normalizer.denormalize_flattened(
                    pod_normalized, "variable"
                )
            )
            metric_state["DeepONet"].update(
                reference, direct_prediction
            )
            metric_state["POD-DeepONet"].update(
                reference, pod_prediction
            )
            labels = batch["labels"]
            for index, coordinate in enumerate(batch["coordinates"]):
                if len(plot_reference) >= plot_limit:
                    break
                plot_coordinates.append(to_numpy(coordinate))
                plot_reference.append(reference[index])
                plot_direct.append(direct_prediction[index])
                plot_pod.append(pod_prediction[index])
    if labels is None or not plot_reference:
        raise RuntimeError("The test loader produced no trajectories.")
    metrics = {
        name: state.result() for name, state in metric_state.items()
    }
    reference_plot = np.stack(plot_reference)
    prediction_plots = {
        "DeepONet": np.stack(plot_direct),
        "POD-DeepONet": np.stack(plot_pod),
    }

    _plot_losses(
        {"DeepONet": direct_history, "POD-DeepONet": pod_history},
        output_path,
    )
    _plot_reconstructions(
        plot_coordinates,
        reference_plot,
        prediction_plots,
        labels,
        plot_labels,
        coordinate_label,
        max(direct_config.plot_cases, pod_config.plot_cases),
        output_path,
    )
    summary = {
        "direct_config": asdict(direct_config),
        "pod_config": asdict(pod_config),
        "parameter_counts": {
            "DeepONet": deeponet_parameter_counts(direct_model),
            "POD-DeepONet": deeponet_parameter_counts(pod_model),
        },
        "pod_components": pod.n_components,
        "pod_cumulative_explained_variance": pod.cumulative_explained_variance,
        "metrics": metrics,
    }
    np.savez(
        output_path / "ipca_pod_matrix.npz",
        mean=pod.mean,
        basis=pod.basis,
        explained_variance_ratio=pod.explained_variance_ratio,
        output_shape=np.asarray(pod.output_shape, dtype=np.int64),
    )
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return DeepONetBenchmarkResult(
        direct_model=direct_model,
        pod_model=pod_model,
        pod=pod,
        metrics=metrics,
    )
