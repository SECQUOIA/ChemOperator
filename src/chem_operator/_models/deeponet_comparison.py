"""Headless training, evaluation, and artifacts for DeepONet comparisons."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from chem_operator._normalization.base import Normalizer

from .deeponet import (
    CoordinateScaler,
    DeepONetTrainingConfig,
    DeepONetTrainingHistory,
    _with_output_channel,
    deeponet_parameter_counts,
    make_deeponet_dataloader,
)
from .deepxde import DeepONetAdapter
from .pod import PODTransform, fit_incremental_pod_dataset
from .training import train_deeponet_lazy

DIRECT_MODEL_ID = "deeponet"
POD_MODEL_ID = "pod_deeponet"
MODEL_IDS = (DIRECT_MODEL_ID, POD_MODEL_ID)
ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReconstructionSamples:
    """A bounded set of physical-unit predictions used for later plotting."""

    coordinates: np.ndarray
    reference: np.ndarray
    predictions: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class DeepONetComparisonResult:
    """Models and serializable results from a direct/POD comparison."""

    direct_model: torch.nn.Module
    pod_model: torch.nn.Module
    direct_config: DeepONetTrainingConfig
    pod_config: DeepONetTrainingConfig
    direct_history: DeepONetTrainingHistory
    pod_history: DeepONetTrainingHistory
    coordinate_scaler: CoordinateScaler
    pod: PODTransform
    metrics: Mapping[str, Mapping[str, float]]
    reconstructions: ReconstructionSamples


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
            "rmse": float(np.sqrt(self.squared_error / max(self.count, 1))),
            "relative_l2": float(
                np.sqrt(
                    self.squared_error
                    / max(self.reference_squared, 1.0e-30)
                )
            ),
            "max_absolute_error": self.max_absolute_error,
        }


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def run_deeponet_comparison(
    train: DeepONetAdapter,
    validation: DeepONetAdapter,
    test: DeepONetAdapter,
    normalizer: Normalizer,
    *,
    direct_config: DeepONetTrainingConfig,
    pod_config: DeepONetTrainingConfig | None = None,
    pod: PODTransform | None = None,
    pod_variance_threshold: float = 0.999,
    reconstruction_cases: int = 2,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DeepONetComparisonResult:
    """Train and evaluate direct and POD DeepONets without writing files."""

    adapters = (train, validation, test)
    if any(adapter.format != "cartesian_product" for adapter in adapters):
        raise ValueError(
            "DeepONet comparison requires Cartesian-product adapters."
        )
    if reconstruction_cases < 1:
        raise ValueError("reconstruction_cases must be positive.")
    pod_config = pod_config or direct_config
    first_train = train[0]
    scaler = CoordinateScaler.fit(_numpy(first_train["trunk"]))
    if pod is None:
        pod = fit_incremental_pod_dataset(
            train,
            variance_threshold=pod_variance_threshold,
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

    loader = make_deeponet_dataloader(
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
    metric_states = {
        DIRECT_MODEL_ID: _StreamingMetrics(),
        POD_MODEL_ID: _StreamingMetrics(),
    }
    coordinates: list[np.ndarray] = []
    references: list[np.ndarray] = []
    direct_predictions: list[np.ndarray] = []
    pod_predictions: list[np.ndarray] = []
    labels: tuple[str, ...] | None = None

    with torch.no_grad():
        for batch in loader:
            branch = batch["branch"]
            trunk = batch["trunk"]
            target = batch["target"]
            direct_normalized = direct_model(
                (branch.to(direct_device), trunk.to(direct_device))
            )
            direct_normalized = _with_output_channel(
                direct_normalized,
                target.shape[-1],
            ).cpu()
            pod_flattened = pod_model(
                (branch.to(pod_device), trunk.to(pod_device))
            )
            pod_normalized = pod.unflatten_tensor(pod_flattened).cpu()
            reference = _numpy(
                normalizer.denormalize_flattened(target, "variable")
            )
            direct_prediction = _numpy(
                normalizer.denormalize_flattened(
                    direct_normalized,
                    "variable",
                )
            )
            pod_prediction = _numpy(
                normalizer.denormalize_flattened(
                    pod_normalized,
                    "variable",
                )
            )
            metric_states[DIRECT_MODEL_ID].update(
                reference,
                direct_prediction,
            )
            metric_states[POD_MODEL_ID].update(reference, pod_prediction)

            batch_labels = tuple(batch["labels"])
            if labels is None:
                labels = batch_labels
            elif batch_labels != labels:
                raise ValueError("Test trajectories have inconsistent labels.")
            for index, coordinate in enumerate(batch["coordinates"]):
                if len(references) >= reconstruction_cases:
                    break
                coordinates.append(_numpy(coordinate))
                references.append(reference[index])
                direct_predictions.append(direct_prediction[index])
                pod_predictions.append(pod_prediction[index])

    if labels is None or not references:
        raise RuntimeError("The test loader produced no trajectories.")
    reconstructions = ReconstructionSamples(
        coordinates=np.stack(coordinates),
        reference=np.stack(references),
        predictions=np.stack(
            (
                np.stack(direct_predictions),
                np.stack(pod_predictions),
            )
        ),
        labels=labels,
    )
    return DeepONetComparisonResult(
        direct_model=direct_model,
        pod_model=pod_model,
        direct_config=direct_config,
        pod_config=pod_config,
        direct_history=direct_history,
        pod_history=pod_history,
        coordinate_scaler=scaler,
        pod=pod,
        metrics={
            model_id: state.result()
            for model_id, state in metric_states.items()
        },
        reconstructions=reconstructions,
    )


def _history_rows(
    model_id: str,
    history: DeepONetTrainingHistory,
) -> list[dict[str, str | int | float]]:
    if not (
        len(history.steps)
        == len(history.loss_train)
        == len(history.loss_test)
    ):
        raise ValueError(f"{model_id} history lengths do not match.")
    rows: list[dict[str, str | int | float]] = []
    for epoch, train_values, valid_values in zip(
        history.steps,
        history.loss_train,
        history.loss_test,
    ):
        rows.extend(
            (
                {
                    "model": model_id,
                    "epoch": epoch,
                    "split": "train",
                    "metric": history.loss_name,
                    "value": float(np.sum(train_values)),
                },
                {
                    "model": model_id,
                    "epoch": epoch,
                    "split": "validation",
                    "metric": history.loss_name,
                    "value": float(np.sum(valid_values)),
                },
            )
        )
    return rows


def save_deeponet_comparison(
    result: DeepONetComparisonResult,
    output_dir: str | Path,
    *,
    problem: str,
) -> None:
    """Write one versioned, plot-independent comparison artifact set."""

    if not problem:
        raise ValueError("problem must be a non-empty identifier.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parameter_counts = {
        DIRECT_MODEL_ID: deeponet_parameter_counts(result.direct_model),
        POD_MODEL_ID: deeponet_parameter_counts(result.pod_model),
    }
    configs = {
        DIRECT_MODEL_ID: asdict(result.direct_config),
        POD_MODEL_ID: asdict(result.pod_config),
    }
    summary = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "problem": problem,
        "models": {
            model_id: {
                "config": configs[model_id],
                "parameter_counts": parameter_counts[model_id],
                "metrics": dict(result.metrics[model_id]),
            }
            for model_id in MODEL_IDS
        },
        "pod": {
            "components": result.pod.n_components,
            "cumulative_explained_variance": (
                result.pod.cumulative_explained_variance
            ),
        },
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    rows = [
        *_history_rows(DIRECT_MODEL_ID, result.direct_history),
        *_history_rows(POD_MODEL_ID, result.pod_history),
    ]
    with (output / "history.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "epoch", "split", "metric", "value"),
        )
        writer.writeheader()
        writer.writerows(rows)

    samples = result.reconstructions
    np.savez(
        output / "reconstructions.npz",
        coordinates=samples.coordinates,
        reference=samples.reference,
        predictions=samples.predictions,
        model_ids=np.asarray(MODEL_IDS),
        labels=np.asarray(samples.labels),
    )
    np.savez(
        output / "ipca_pod_matrix.npz",
        mean=result.pod.mean,
        basis=result.pod.basis,
        explained_variance_ratio=result.pod.explained_variance_ratio,
        output_shape=np.asarray(result.pod.output_shape, dtype=np.int64),
    )
