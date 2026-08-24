"""Tune, train, and benchmark a Fourier neural operator for Q2D CMR data.

The current operator maps the two independently varied case parameters
``T0`` and ``SCCM`` to axial velocity and the CH4/H2 mole-fraction fields.
Channel extraction is declarative: extending ``INPUT_CHANNELS`` or
``OUTPUT_CHANNELS`` is sufficient to expose another parameter, constant,
scalar field, or species field to the rest of the workflow.

Converged solution fields should only be added to ``INPUT_CHANNELS`` when
they are genuinely known at inference time. Otherwise, doing so leaks the
prediction target into the model input.
"""

# Environment variables must be set before importing plotting and Ray.
# pylint: disable=wrong-import-position

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
from neuralop.losses import LpLoss
from neuralop.models import FNO
import numpy as np
import optuna
import pandas as pd
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import (
    FNOAdapter,
    FNOChannel,
    fit_fno_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


SLIDE_DPI = 220
plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 20,
        "figure.titlesize": 24,
        "lines.linewidth": 2.6,
        "lines.markersize": 8,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.9,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white"
    }
)

PATHS = ExamplePaths.from_script(__file__, dataset="q2d_cmr")

FILE_STEM = "q2d_cmr"
SEED = 42
METRIC = "best_valid_loss"

# Run-mode flags. Leave both false for tuning, training, and evaluation.
TRAIN_BEST_CONFIG_ONLY = False
PLOT_SAVED_MODEL_ONLY = True

TUNE_SAMPLES = 25
TUNE_EPOCHS = 30
FINAL_EPOCHS = 100
EVALUATION_BATCH_SIZE = 2

CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
MAX_CONCURRENT_TRIALS = 1

LATENCY_WARMUPS = 5
LATENCY_REPEATS = 20
RELATIVE_L2_EPS = 1.0e-8
AMORTIZED_EVALUATION_COUNTS = (1, 10, 100, 1000)
REPRESENTATIVE_RESOLUTION_COUNT = 4

MESH_FILE_PATTERN = re.compile(r"q2d_cmr_(\d+)_(\d+)_test\.h5")

INPUT_CHANNELS = (
    FNOChannel(
        "T0",
        "parameter",
        "T0",
        display_name="Inlet temperature",
        unit="K",
    ),
    FNOChannel(
        "SCCM",
        "constant",
        "SCCM",
        display_name="Inlet flow rate",
        unit="sccm",
    ),
)

OUTPUT_CHANNELS = (
    # FNOChannel(
    #     "velocity_axial",
    #     "field",
    #     "velocity_axial",
    #     display_name="Axial velocity",
    #     unit="m/s",
    # ),
    FNOChannel(
        "X_CH4",
        "species",
        "X",
        species="CH4",
        display_name="Methane",
        unit="-",
    ),
    FNOChannel(
        "X_H2",
        "species",
        "X",
        species="H2",
        display_name="Hydrogen",
        unit="-",
    ),
    FNOChannel(
        "theta_C(s)",
        "species",
        "theta",
        species="C(s)",
        display_name="Carbon Accumulation",
        unit="-",
    ),
)


@dataclass(frozen=True)
class MeshFile:
    """One resolution-sweep HDF5 file."""

    path: Path
    n_z: int
    n_r: int

    @property
    def mesh_points(self) -> int:
        return self.n_z * self.n_r


def raw_dataset(path: Path) -> ChemOperatorDataset:
    """Open one complete steady Q2D field per HDF5 case."""
    field_names = FNOAdapter.required_field_names(
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
    )
    if not field_names:
        raise ValueError("At least one configured channel must read an HDF5 field.")
    return ChemOperatorDataset(
        path,
        task="field_map",
        coordinate_name="z",
        input_fields=field_names,
        output_fields=field_names,
        constant_inputs=FNOAdapter.required_constant_names(
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        ),
        n_steps_input=1,
        dtype=torch.float32,
    )


def _case_geometry(metadata: Mapping[str, Any]) -> tuple[float, float]:
    """Return physical axial length and lumen radius for domain validation."""
    inputs = metadata.get("input_values", {})
    try:
        return float(inputs["LENGTH_LUMEN"]), float(inputs["CHANNELRAD_LUMEN"])
    except KeyError as exc:
        raise KeyError("Q2D metadata is missing its physical domain geometry.") from exc


def adapter_spatial_shape(
    dataset: FNOAdapter,
    dataset_name: str,
) -> tuple[int, int]:
    """Return and validate the common spatial shape stored by an adapter."""
    if len(dataset) == 0:
        raise RuntimeError(f"{dataset_name} contains no cases.")
    expected: tuple[int, int] | None = None
    for index in range(len(dataset)):
        physical = dataset.physical_item(index)
        input_shape = tuple(int(value) for value in physical["x"].shape[-2:])
        output_shape = tuple(int(value) for value in physical["y"].shape[-2:])
        if input_shape != output_shape:
            raise ValueError(
                f"{dataset_name} case {index} has input shape {input_shape} "
                f"and output shape {output_shape}."
            )
        if expected is None:
            expected = output_shape
        elif output_shape != expected:
            raise ValueError(
                f"{dataset_name} mixes spatial shapes {expected} and "
                f"{output_shape}."
            )
    assert expected is not None
    return expected


def fit_normalizer(
    path: Path,
) -> tuple[ZScoreNormalizer, tuple[float, float], tuple[int, int]]:
    """Fit training-only statistics and discover the training mesh shape."""
    dataset = raw_dataset(path)
    geometry: tuple[float, float] | None = None
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            current_geometry = _case_geometry(sample["metadata"])
            if geometry is None:
                geometry = current_geometry
            elif not np.allclose(
                current_geometry,
                geometry,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("Training cases do not share one physical domain.")
        normalizer = fit_fno_zscore_normalizer(
            dataset,
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        )
        training_adapter = FNOAdapter(
            dataset,
            normalizer,
            input_channels=INPUT_CHANNELS,
            output_channels=OUTPUT_CHANNELS,
            coordinate_names=("z", "r"),
        )
        training_shape = adapter_spatial_shape(
            training_adapter,
            path.name,
        )
    finally:
        dataset.close()

    if geometry is None:
        raise RuntimeError("The training dataset contains no cases.")
    return normalizer, geometry, training_shape


def normalizer_state(
    normalizer: ZScoreNormalizer,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a CPU-only serializable normalizer state."""
    return {
        "mean": {
            name: value.detach().cpu() for name, value in normalizer.means.items()
        },
        "std": {
            name: value.detach().cpu() for name, value in normalizer.stds.items()
        },
        "mean_delta": {
            name: value.detach().cpu()
            for name, value in normalizer.delta_means.items()
        },
        "std_delta": {
            name: value.detach().cpu()
            for name, value in normalizer.delta_stds.items()
        },
    }


def normalizer_from_state(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> ZScoreNormalizer:
    """Reconstruct the configured ChemOperator normalizer."""
    return ZScoreNormalizer(
        state,
        variable_field_order=tuple(spec.label for spec in OUTPUT_CHANNELS),
        constant_field_order=tuple(spec.label for spec in INPUT_CHANNELS),
    )


def make_adapter(
    path: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
) -> tuple[ChemOperatorDataset, FNOAdapter]:
    """Open a raw dataset and its normalized Q2D adapter."""
    raw = raw_dataset(path)
    current_geometry = _case_geometry(raw[0]["metadata"])
    if not np.allclose(
        current_geometry,
        geometry,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raw.close()
        raise ValueError(
            f"Dataset geometry {current_geometry} differs from training "
            f"geometry {geometry}."
        )
    return raw, FNOAdapter(
        raw,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("z", "r"),
    )


def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured two-dimensional NeuralOperator FNO."""
    return FNO(
        n_modes=(int(config["modes_z"]), int(config["modes_r"])),
        in_channels=len(INPUT_CHANNELS),
        out_channels=len(OUTPUT_CHANNELS),
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
        domain_padding=float(config.get("domain_padding", 0.0)),
    ).to(device)


def count_parameters(model: nn.Module) -> int:
    """Return the number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def validation_loss(
    model: nn.Module,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return mean global normalized relative L2 over a complete split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_function = LpLoss(d=3, p=2, reduction="mean", eps=RELATIVE_L2_EPS)
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["y"].to(device)
            prediction = model(batch["x"].to(device))
            loss = loss_function(prediction, target)
            count = target.shape[0]
            total += float(loss) * count
            samples += count
    return total / max(samples, 1)


def train_model(  # pylint: disable=too-many-arguments,too-many-locals
    config: Mapping[str, Any],
    train_data: FNOAdapter,
    valid_data: FNOAdapter,
    *,
    epochs: int,
    device: torch.device,
    report_to_ray: bool = False,
    print_epochs: bool = False,
) -> tuple[FNO, dict[str, list[float]], float]:
    """Train one FNO and restore its best validation checkpoint."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = model_from_config(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = LpLoss(d=3, p=2, reduction="mean", eps=RELATIVE_L2_EPS)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history = {"train_loss": [], "valid_loss": []}
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    tic = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        samples = 0
        for batch in loader:
            model_input = batch["x"].to(device)
            target = batch["y"].to(device)
            prediction = model(model_input)
            loss = loss_function(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = target.shape[0]
            train_total += float(loss.detach()) * count
            samples += count

        train_value = train_total / max(samples, 1)
        valid_value = validation_loss(
            model,
            valid_data,
            batch_size=int(config["batch_size"]),
            device=device,
        )
        history["train_loss"].append(train_value)
        history["valid_loss"].append(valid_value)
        if valid_value < best_loss:
            best_loss = valid_value
            best_state = deepcopy(model.state_dict())
        if print_epochs:
            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"train={train_value:.6e}, valid={valid_value:.6e}"
            )
        if report_to_ray:
            tune.report(
                {
                    "train_loss": train_value,
                    "valid_loss": valid_value,
                    METRIC: best_loss,
                    "n_params": count_parameters(model),
                }
            )

    elapsed = time.perf_counter() - tic
    model.load_state_dict(best_state)
    return model.eval(), history, elapsed


def ray_trial(
    config: Mapping[str, Any],
    *,
    data_dir: str,
    normalization: Mapping[str, Mapping[str, torch.Tensor]],
    geometry: tuple[float, float],
) -> None:
    """Ray trainable that creates process-local lazy HDF5 readers."""
    torch.set_num_threads(max(1, CPUS_PER_TRIAL))
    normalizer = normalizer_from_state(normalization)
    train_raw, train_data = make_adapter(
        Path(data_dir) / f"{FILE_STEM}_train.h5",
        normalizer,
        geometry,
    )
    valid_raw, valid_data = make_adapter(
        Path(data_dir) / f"{FILE_STEM}_valid.h5",
        normalizer,
        geometry,
    )
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_model(
            config,
            train_data,
            valid_data,
            epochs=TUNE_EPOCHS,
            device=device,
            report_to_ray=True,
        )
    finally:
        train_raw.close()
        valid_raw.close()


def tune_hyperparameters(
    data_dir: Path,
    output_dir: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
) -> dict[str, Any]:
    """Run Optuna/ASHA tuning and return the best final configuration."""
    search = OptunaSearch(
        metric=METRIC,
        mode="min",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=4,
            multivariate=True,
        ),
    )
    scheduler = ASHAScheduler(
        metric=METRIC,
        mode="min",
        time_attr="training_iteration",
        max_t=TUNE_EPOCHS,
        grace_period=min(8, TUNE_EPOCHS),
        reduction_factor=2,
    )
    parameterized = tune.with_parameters(
        ray_trial,
        data_dir=str(data_dir.resolve()),
        normalization=normalizer_state(normalizer),
        geometry=geometry,
    )
    trainable = tune.with_resources(
        parameterized,
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )
    run_name = time.strftime("q2d_fno_%Y%m%d_%H%M%S")
    tuner = tune.Tuner(
        trainable,
        param_space={
            "modes_z": tune.choice([4, 5, 6]),
            "modes_r": tune.choice([2, 3, 4]),
            "hidden_channels": tune.choice([16, 20, 24, 28, 32]),
            "n_layers": tune.choice([6, 7, 8, 9]),
            "learning_rate": tune.loguniform(1.0e-4, 4.0e-3),
            "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
            "batch_size": 2, #tune.choice([2, 4]),
            "domain_padding": tune.choice([0.0, 0.05, 0.1, 0.15]),
        },
        tune_config=tune.TuneConfig(
            search_alg=search,
            scheduler=scheduler,
            num_samples=TUNE_SAMPLES,
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=run_name,
            storage_path=str((output_dir / "ray_results").resolve()),
            verbose=1,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=METRIC, mode="min", scope="last")
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in best.config.items()
    }


def save_checkpoint(
    path: Path,
    model: FNO,
    config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
) -> None:
    """Save model tensors and all data-interface configuration."""
    tensor_state = {
        name: value
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    torch.save(
        {
            "state_dict": tensor_state,
            "model_config": dict(config),
            "input_channels": [asdict(spec) for spec in INPUT_CHANNELS],
            "output_channels": [asdict(spec) for spec in OUTPUT_CHANNELS],
            "normalization": normalizer_state(normalizer),
            "geometry": tuple(geometry),
            "training_shape": tuple(training_shape),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FNO, ZScoreNormalizer, tuple[float, float], tuple[int, int]]:
    """Load and validate a serialized Q2D FNO."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected_inputs = [asdict(spec) for spec in INPUT_CHANNELS]
    expected_outputs = [asdict(spec) for spec in OUTPUT_CHANNELS]
    if checkpoint["input_channels"] != expected_inputs:
        raise RuntimeError("Checkpoint input-channel configuration has changed.")
    if checkpoint["output_channels"] != expected_outputs:
        raise RuntimeError("Checkpoint output-channel configuration has changed.")
    training_shape = tuple(int(value) for value in checkpoint["training_shape"])
    if len(training_shape) != 2 or any(value < 2 for value in training_shape):
        raise RuntimeError(
            f"Checkpoint training mesh {training_shape} is invalid."
        )
    model = model_from_config(checkpoint["model_config"], device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "FNO checkpoint tensors are incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    geometry = tuple(float(value) for value in checkpoint["geometry"])
    return (
        model.eval(),
        normalizer_from_state(checkpoint["normalization"]),
        geometry,
        training_shape,
    )


def _relative_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=1)
    reference = target.flatten(start_dim=1)
    return torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(RELATIVE_L2_EPS)


def _relative_l2_per_field(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=2)
    reference = target.flatten(start_dim=2)
    return torch.linalg.vector_norm(difference, dim=2) / torch.linalg.vector_norm(
        reference, dim=2
    ).clamp_min(RELATIVE_L2_EPS)


def evaluate(
    model: FNO,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate mean per-case normalized relative L2."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    global_values: list[torch.Tensor] = []
    field_values: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["y"].to(device)
            prediction = model(batch["x"].to(device))
            global_values.append(
                _relative_l2_per_sample(prediction, target).detach().cpu()
            )
            field_values.append(
                _relative_l2_per_field(prediction, target).detach().cpu()
            )
    global_tensor = torch.cat(global_values)
    fields_tensor = torch.cat(field_values)
    metrics = {"normalized_relative_l2": float(global_tensor.mean())}
    for index, spec in enumerate(OUTPUT_CHANNELS):
        metrics[f"normalized_relative_l2_{spec.label}"] = float(
            fields_tensor[:, index].mean()
        )
    return metrics


def resize_model_input(
    model_input: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Resize channel-first inputs to a requested FNO evaluation mesh.

    Broadcast scalar channels remain exactly constant. This interpolation also
    makes future continuous physical input fields configuration-compatible
    with zero-shot superresolution. Evaluating directly on the requested mesh
    also keeps NeuralOperator domain padding and unpadding resolution-consistent.
    """
    if tuple(model_input.shape[-2:]) == tuple(shape):
        return model_input
    return torch_functional.interpolate(
        model_input.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=True,
    )[0]


def write_history(
    path: Path,
    history: Mapping[str, list[float]],
) -> None:
    """Write per-epoch training and validation losses."""
    frame = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history["train_loss"]) + 1),
            "train_loss": history["train_loss"],
            "valid_loss": history["valid_loss"],
        }
    )
    frame.to_csv(path, index=False)


def read_history(path: Path) -> dict[str, list[float]]:
    """Load and validate a previously saved loss history."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Training history not found at {path}. Train the model first."
        )
    frame = pd.read_csv(path)
    required = {"train_loss", "valid_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Training history {path} is missing columns {sorted(missing)}."
        )
    history = {
        name: frame[name].astype(float).tolist()
        for name in ("train_loss", "valid_loss")
    }
    if not history["train_loss"] or any(
        not math.isfinite(value)
        for values in history.values()
        for value in values
    ):
        raise ValueError(f"Training history {path} is empty or non-finite.")
    return history


def plot_history(
    path: Path,
    history: Mapping[str, list[float]],
) -> None:
    """Plot normalized relative-L2 histories."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    axis.semilogy(epochs, history["train_loss"], label="Training")
    axis.semilogy(epochs, history["valid_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized relative L2")
    axis.set_title("FNO training and validation loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def plot_field_comparison(  # pylint: disable=too-many-arguments
    path: Path,
    truth: torch.Tensor,
    prediction: torch.Tensor,
    z: torch.Tensor,
    r: torch.Tensor,
    *,
    title: str,
) -> None:
    """Plot solver truth, FNO output, and absolute error for three fields."""
    exact = truth.detach().cpu().numpy()
    reconstructed = prediction.detach().cpu().numpy()
    error = np.abs(reconstructed - exact)
    z_mm = 1.0e3 * z.detach().cpu().numpy()
    r_mm = 1.0e3 * r.detach().cpu().numpy()
    figure, axes = plt.subplots(
        len(OUTPUT_CHANNELS),
        3,
        figsize=(16, 12),
        squeeze=False,
        constrained_layout=True,
    )
    for row, spec in enumerate(OUTPUT_CHANNELS):
        low = min(float(exact[row].min()), float(reconstructed[row].min()))
        high = max(float(exact[row].max()), float(reconstructed[row].max()))
        images = (
            axes[row, 0].pcolormesh(
                z_mm,
                r_mm,
                exact[row].T,
                shading="auto",
                vmin=low,
                vmax=high,
            ),
            axes[row, 1].pcolormesh(
                z_mm,
                r_mm,
                reconstructed[row].T,
                shading="auto",
                vmin=low,
                vmax=high,
            ),
            axes[row, 2].pcolormesh(
                z_mm,
                r_mm,
                error[row].T,
                shading="auto",
                cmap="magma",
            ),
        )
        axes[row, 0].set_ylabel(
            r"$\bf{" + spec.title + "}$\nRadius [mm]",
            fontsize=22
        )
        colorbar_label = spec.unit if spec.unit != "-" else "Mole fraction [-]"
        for column, column_title in enumerate(
            ("Numerical Solver", "FNO", "Absolute error")
        ):
            axes[0, column].set_title(column_title)
            if row == len(OUTPUT_CHANNELS) - 1:
                axes[row, column].set_xlabel("Axial position [mm]")

        # The solver and FNO use the same limits and share the colorbar beside
        # the FNO column. The absolute-error colorbar remains independent.
        figure.colorbar(images[1], ax=axes[row, 1])
        figure.colorbar(
            images[2],
            ax=axes[row, 2],
            label=colorbar_label,
        )
    figure.suptitle(title)
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def discover_mesh_files(data_dir: Path) -> list[MeshFile]:
    """Return all available ``q2d_cmr_{n_z}_{n_r}_test`` files."""
    files: list[MeshFile] = []
    for path in data_dir.glob("q2d_cmr_*_*_test.h5"):
        match = MESH_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            files.append(MeshFile(path, int(match.group(1)), int(match.group(2))))
    files.sort(key=lambda item: (item.n_z, item.n_r))
    discovered = {(item.n_z, item.n_r) for item in files}
    axial_points = {item.n_z for item in files}
    radial_points = {item.n_r for item in files}
    expected = {
        (n_z, n_r)
        for n_z in axial_points
        for n_r in radial_points
    }
    missing = sorted(expected - discovered)
    if missing:
        print(
            "Mesh benchmark files not present (continuing): "
            + ", ".join(f"{n_z}x{n_r}" for n_z, n_r in missing)
        )
    if not files:
        raise FileNotFoundError("No Q2D resolution-sweep test files were found.")
    return files


def _case_controls(metadata: Mapping[str, Any]) -> tuple[float, float]:
    params = metadata.get("params", {})
    try:
        return float(params["T0"]), float(params["sccm"])
    except KeyError as exc:
        raise KeyError("Case metadata is missing T0 or sccm.") from exc


def _controls_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    first_t0, first_sccm = _case_controls(first)
    second_t0, second_sccm = _case_controls(second)
    return math.isclose(first_t0, second_t0, rel_tol=1.0e-10, abs_tol=1.0e-8) and (
        math.isclose(
            first_sccm,
            second_sccm,
            rel_tol=1.0e-10,
            abs_tol=1.0e-8,
        )
    )


def find_superresolution_pair(
    mesh_files: Sequence[MeshFile],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
) -> tuple[MeshFile, int, MeshFile, int]:
    """Find a training-mesh case with truth on the finest matching mesh."""
    try:
        base_file = next(
            item
            for item in mesh_files
            if (item.n_z, item.n_r) == training_shape
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            "No mesh-sweep test file matches the discovered training mesh "
            f"{training_shape[0]}x{training_shape[1]}."
        ) from exc
    base_raw, base_data = make_adapter(base_file.path, normalizer, geometry)
    candidates = sorted(
        (
            item
            for item in mesh_files
            if item.mesh_points > math.prod(training_shape)
            and item.n_z >= training_shape[0]
            and item.n_r >= training_shape[1]
        ),
        key=lambda item: (item.mesh_points, item.n_r, item.n_z),
        reverse=True,
    )
    try:
        base_metadata = [
            base_data.physical_item(index)["metadata"]
            for index in range(len(base_data))
        ]
        for candidate in candidates:
            fine_raw, fine_data = make_adapter(
                candidate.path,
                normalizer,
                geometry,
            )
            try:
                for base_index, base_case in enumerate(base_metadata):
                    for fine_index in range(len(fine_data)):
                        fine_case = fine_data.physical_item(fine_index)["metadata"]
                        if _controls_match(base_case, fine_case):
                            return base_file, base_index, candidate, fine_index
            finally:
                fine_raw.close()
    finally:
        base_raw.close()
    raise RuntimeError("No matched coarse/fine case exists for superresolution.")


def make_reconstruction_figures(  # pylint: disable=too-many-locals
    model: FNO,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    mesh_files: Sequence[MeshFile],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Create matched coarse and solver-truth superresolution figures."""
    base_file, base_index, fine_file, fine_index = find_superresolution_pair(
        mesh_files,
        normalizer,
        geometry,
        training_shape,
    )
    base_raw, base_data = make_adapter(
        base_file.path,
        normalizer,
        geometry,
    )
    fine_raw, fine_data = make_adapter(fine_file.path, normalizer, geometry)
    try:
        base = base_data[base_index]
        fine = fine_data[fine_index]
        with torch.no_grad():
            coarse_prediction = model(base["x"].unsqueeze(0).to(device)).cpu()[0]
            fine_input = resize_model_input(
                base["x"],
                (fine_file.n_z, fine_file.n_r),
            )
            fine_prediction = model(fine_input.unsqueeze(0).to(device)).cpu()[0]
        coarse_truth = base_data.denormalize_output(base["y"])
        coarse_prediction = base_data.denormalize_output(coarse_prediction)
        fine_truth = fine_data.denormalize_output(fine["y"])
        fine_prediction = fine_data.denormalize_output(fine_prediction)
        base_metadata = base_data.physical_item(base_index)["metadata"]
        t0, sccm = _case_controls(base_metadata)
        plot_field_comparison(
            PATHS.output / "test_reconstructions.png",
            coarse_truth,
            coarse_prediction,
            base["z"],
            base["r"],
            title=(
                "Test reconstruction: "
                f"{training_shape[0]}×{training_shape[1]}, "
                f"$T_0$={t0:.2f} K, $Q_s$={sccm:.2f} sccm"
            ),
        )
        plot_field_comparison(
            PATHS.output / "test_superresolution.png",
            fine_truth,
            fine_prediction,
            fine["z"],
            fine["r"],
            title=(
                "2D CMR FNO superresolution: "
                f"{training_shape[0]}×{training_shape[1]} to "
                f"{fine_file.n_z}×{fine_file.n_r}"
            ),
        )
    finally:
        base_raw.close()
        fine_raw.close()
    return {
        "training_file": base_file.path.name,
        "training_shape": list(training_shape),
        "base_case_index": base_index,
        "fine_file": fine_file.path.name,
        "fine_case_index": fine_index,
        "fine_shape": [fine_file.n_z, fine_file.n_r],
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _complete_fno_evaluation(
    model: FNO,
    physical_input: torch.Tensor,
    dataset: FNOAdapter,
    training_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Run preprocessing, resampling, inference, and postprocessing once."""
    normalized_input = torch.stack(
        [
            dataset.normalizer.normalize(
                physical_input[index],
                channel.label,
            )
            for index, channel in enumerate(dataset.input_channels)
        ]
    )
    training_input = resize_model_input(normalized_input, training_shape)
    evaluation_input = resize_model_input(training_input, output_shape)
    normalized_output = model(evaluation_input.unsqueeze(0).to(device)).cpu()[0]
    return dataset.denormalize_output(normalized_output)


def inference_latency(
    model: FNO,
    physical_input: torch.Tensor,
    dataset: FNOAdapter,
    training_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    device: torch.device,
) -> float:
    """Return median end-to-end single-case FNO evaluation time."""
    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUPS):
            _complete_fno_evaluation(
                model,
                physical_input,
                dataset,
                training_shape,
                output_shape,
                device=device,
            )
        _synchronize(device)
        durations = []
        for _ in range(LATENCY_REPEATS):
            _synchronize(device)
            tic = time.perf_counter()
            _complete_fno_evaluation(
                model,
                physical_input,
                dataset,
                training_shape,
                output_shape,
                device=device,
            )
            _synchronize(device)
            durations.append(time.perf_counter() - tic)
    return float(np.median(durations))


def benchmark_meshes(  # pylint: disable=too-many-locals
    model: FNO,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    mesh_files: Sequence[MeshFile],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate all runs in every discovered resolution-sweep file."""
    rows: list[dict[str, Any]] = []
    model.eval()
    for mesh_file in mesh_files:
        raw, dataset = make_adapter(mesh_file.path, normalizer, geometry)
        try:
            for case_index in range(len(dataset)):
                sample = dataset[case_index]
                target = sample["y"].unsqueeze(0).to(device)
                target_shape = tuple(int(value) for value in target.shape[-2:])
                if target_shape != (mesh_file.n_z, mesh_file.n_r):
                    raise ValueError(
                        f"{mesh_file.path.name} declares "
                        f"{mesh_file.n_z}x{mesh_file.n_r} but stores {target_shape}."
                    )
                training_input = resize_model_input(sample["x"], training_shape)
                prediction_input = resize_model_input(
                    training_input,
                    target_shape,
                )
                with torch.no_grad():
                    prediction = model(prediction_input.unsqueeze(0).to(device))
                global_l2 = float(
                    _relative_l2_per_sample(prediction, target).cpu()[0]
                )
                field_l2 = _relative_l2_per_field(prediction, target).cpu()[0]
                physical = dataset.physical_item(case_index)
                metadata = physical["metadata"]
                t0, sccm = _case_controls(metadata)
                row: dict[str, Any] = {
                    "dataset_file": mesh_file.path.name,
                    "case_index": case_index,
                    "n_z": mesh_file.n_z,
                    "n_r": mesh_file.n_r,
                    "mesh_points": mesh_file.mesh_points,
                    "T0_K": t0,
                    "SCCM": sccm,
                    "normalized_relative_l2": global_l2,
                    "solver_wall_time_s": float(metadata["wall_time"]),
                    "fno_wall_time_s": inference_latency(
                        model,
                        physical["x"],
                        dataset,
                        training_shape,
                        target_shape,
                        device=device,
                    ),
                }
                for field_index, spec in enumerate(OUTPUT_CHANNELS):
                    row[f"normalized_relative_l2_{spec.label}"] = float(
                        field_l2[field_index]
                    )
                rows.append(row)
        finally:
            raw.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("The mesh benchmark produced no rows.")
    return frame.sort_values(
        ["mesh_points", "n_z", "n_r", "case_index"],
        ignore_index=True,
    )


def _grouped_summary(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("mesh_points", sort=True)[column]
    return (
        grouped.mean().index.to_numpy(dtype=float),
        grouped.mean().to_numpy(dtype=float),
        grouped.min().to_numpy(dtype=float),
        grouped.max().to_numpy(dtype=float),
    )


def dataset_generation_costs() -> dict[str, float | int]:
    """Return numerical-simulation costs for training and validation data."""
    result: dict[str, float | int] = {}
    total_seconds = 0.0
    total_cases = 0
    for split in ("train", "valid"):
        dataset = raw_dataset(PATHS.data / f"{FILE_STEM}_{split}.h5")
        try:
            wall_times = [
                float(dataset[index]["metadata"]["wall_time"])
                for index in range(len(dataset))
            ]
        finally:
            dataset.close()
        if not wall_times or any(
            not math.isfinite(value) or value < 0.0 for value in wall_times
        ):
            raise ValueError(
                f"{split} data contain missing or invalid solver wall times."
            )
        split_seconds = sum(wall_times)
        result[f"{split}_data_generation_seconds"] = split_seconds
        result[f"{split}_data_cases"] = len(wall_times)
        total_seconds += split_seconds
        total_cases += len(wall_times)
    result["data_generation_seconds"] = total_seconds
    result["data_generation_cases"] = total_cases
    return result


def _resolution_cost_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate benchmark costs for each exact axial-radial resolution."""
    summary = (
        frame.groupby(["n_z", "n_r"], as_index=False, sort=True)
        .agg(
            mesh_points=("mesh_points", "first"),
            benchmark_cases=("case_index", "size"),
            numerical_solver_seconds=("solver_wall_time_s", "mean"),
            fno_evaluation_seconds=("fno_wall_time_s", "mean"),
            normalized_relative_l2=("normalized_relative_l2", "mean"),
        )
        .sort_values(["mesh_points", "n_z", "n_r"], ignore_index=True)
    )
    records: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        records.append(
            {
                "n_z": int(row.n_z),
                "n_r": int(row.n_r),
                "mesh_points": int(row.mesh_points),
                "benchmark_cases": int(row.benchmark_cases),
                "numerical_solver_seconds": float(
                    row.numerical_solver_seconds
                ),
                "fno_evaluation_seconds": float(row.fno_evaluation_seconds),
                "normalized_relative_l2": float(row.normalized_relative_l2),
            }
        )
    return records


def _representative_resolutions(
    records: Sequence[Mapping[str, Any]],
    training_shape: tuple[int, int],
) -> list[Mapping[str, Any]]:
    """Select dataset-derived resolutions spanning the benchmark range."""
    if len(records) <= REPRESENTATIVE_RESOLUTION_COUNT:
        selected = list(records)
    else:
        indices = np.linspace(
            0,
            len(records) - 1,
            REPRESENTATIVE_RESOLUTION_COUNT,
        ).round().astype(int)
        selected = [records[index] for index in dict.fromkeys(indices)]
    training_record = next(
        (
            record
            for record in records
            if (record["n_z"], record["n_r"]) == training_shape
        ),
        None,
    )
    if training_record is not None and training_record not in selected:
        if len(selected) < 3:
            selected.append(training_record)
        else:
            replacement = min(
                range(1, len(selected) - 1),
                key=lambda index: abs(
                    selected[index]["mesh_points"]
                    - training_record["mesh_points"]
                ),
            )
            selected[replacement] = training_record
    return sorted(
        selected,
        key=lambda record: (
            record["mesh_points"],
            record["n_z"],
            record["n_r"],
        ),
    )


def build_cost_model(
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
    *,
    ray_tuning_seconds: float,
    final_training_seconds: float,
) -> dict[str, Any]:
    """Build the persisted offline, online, and break-even cost model."""
    data_costs = dataset_generation_costs()
    offline_seconds = (
        float(data_costs["data_generation_seconds"])
        + ray_tuning_seconds
        + final_training_seconds
    )
    records = _resolution_cost_records(frame)
    for record in records:
        difference = (
            record["numerical_solver_seconds"]
            - record["fno_evaluation_seconds"]
        )
        record["break_even_evaluations"] = (
            offline_seconds / difference if difference > 0.0 else None
        )
        record["asymptotic_speedup"] = (
            record["numerical_solver_seconds"]
            / record["fno_evaluation_seconds"]
        )
    representatives = _representative_resolutions(records, training_shape)
    return {
        "definitions": {
            "fno_total": (
                "T_data + T_tune + T_final_train + "
                "N * t_FNO(resolution)"
            ),
            "numerical_total": "N * t_num(resolution)",
            "fno_amortized": (
                "T_offline / N + t_FNO(resolution)"
            ),
            "speedup": "T_num(N, resolution) / T_FNO(N, resolution)",
            "data_generation_source": (
                "sum of per-case solver wall_time metadata for the complete "
                "training and validation datasets"
            ),
            "ray_tuning_scope": (
                "elapsed wall time around all Ray Tune trials, including "
                "pruned or unsuccessful trials"
            ),
            "resolution_aggregation": (
                "mean per-case wall time for each exact (n_z, n_r) mesh"
            ),
            "fno_evaluation_includes": [
                "input normalization",
                "interpolation or resampling",
                "device transfer",
                "model inference",
                "output denormalization",
            ],
        },
        **data_costs,
        "ray_tuning_seconds": float(ray_tuning_seconds),
        "final_training_seconds": float(final_training_seconds),
        "offline_seconds": float(offline_seconds),
        "amortized_evaluation_counts": list(AMORTIZED_EVALUATION_COUNTS),
        "representative_resolutions": [
            {
                "n_z": int(record["n_z"]),
                "n_r": int(record["n_r"]),
                "mesh_points": int(record["mesh_points"]),
            }
            for record in representatives
        ],
        "resolutions": records,
    }


def plot_cumulative_break_even(
    path: Path,
    cost_model: Mapping[str, Any],
) -> None:
    """Plot cumulative numerical and FNO costs with break-even markers."""
    representative_keys = {
        (item["n_z"], item["n_r"])
        for item in cost_model["representative_resolutions"]
    }
    representatives = [
        record
        for record in cost_model["resolutions"]
        if (record["n_z"], record["n_r"]) in representative_keys
    ]
    finite_break_evens = [
        float(record["break_even_evaluations"])
        for record in representatives
        if record["break_even_evaluations"] is not None
    ]
    maximum_cases = max(
        AMORTIZED_EVALUATION_COUNTS[-1],
        1.1 * max(finite_break_evens, default=0.0),
    )
    requested_cases = np.linspace(0.0, maximum_cases, 500)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        squeeze=False,
        constrained_layout=True,
    )
    offline_seconds = float(cost_model["offline_seconds"])
    for axis, record in zip(axes.flat, representatives):
        numerical = requested_cases * record["numerical_solver_seconds"]
        fno = (
            offline_seconds
            + requested_cases * record["fno_evaluation_seconds"]
        )
        axis.plot(requested_cases, numerical, label="Numerical solver")
        axis.plot(requested_cases, fno, linestyle="--", label="FNO")
        break_even = record["break_even_evaluations"]
        if break_even is not None:
            intersection_time = (
                break_even * record["numerical_solver_seconds"]
            )
            axis.scatter(
                [break_even],
                [intersection_time],
                color="black",
                marker="X",
                s=90,
                zorder=4,
                label=f"Break-even: N={break_even:.1f}",
            )
        axis.set_title(
            f"{record['n_z']}×{record['n_r']} "
            f"({record['mesh_points']} points)"
        )
        axis.set_xlabel("Number of new cases, N")
        axis.set_ylabel("Cumulative wall time [s]")
        axis.grid(alpha=0.28)
        axis.legend()
    for axis in axes.flat[len(representatives):]:
        axis.set_visible(False)
    figure.suptitle(
        "Cumulative cost and FNO break-even "
        f"(offline cost = {offline_seconds:.1f} s)"
    )
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def _pareto_indices(
    times: np.ndarray,
    mesh_points: np.ndarray,
) -> np.ndarray:
    """Return indices not dominated in lower-time/higher-resolution space."""
    keep = []
    for index, (wall_time, resolution) in enumerate(
        zip(times, mesh_points)
    ):
        dominated = np.any(
            (times <= wall_time)
            & (mesh_points >= resolution)
            & ((times < wall_time) | (mesh_points > resolution))
        )
        if not dominated:
            keep.append(index)
    return np.asarray(
        sorted(keep, key=lambda item: times[item]),
        dtype=int,
    )


def plot_amortized_pareto(
    path: Path,
    cost_model: Mapping[str, Any],
) -> None:
    """Plot amortized cost-resolution Pareto frontiers."""
    records = cost_model["resolutions"]
    mesh_points = np.asarray(
        [record["mesh_points"] for record in records],
        dtype=float,
    )
    numerical = np.asarray(
        [record["numerical_solver_seconds"] for record in records],
        dtype=float,
    )
    fno_online = np.asarray(
        [record["fno_evaluation_seconds"] for record in records],
        dtype=float,
    )
    offline_seconds = float(cost_model["offline_seconds"])
    figure, axes = plt.subplots(
        1,
        len(AMORTIZED_EVALUATION_COUNTS),
        figsize=(21, 6),
        sharey=True,
        constrained_layout=True,
    )
    numerical_frontier = _pareto_indices(numerical, mesh_points)
    for axis, evaluations in zip(axes, AMORTIZED_EVALUATION_COUNTS):
        fno_amortized = offline_seconds / evaluations + fno_online
        fno_frontier = _pareto_indices(fno_amortized, mesh_points)
        axis.scatter(numerical, mesh_points, color="tab:blue", alpha=0.25)
        axis.scatter(fno_amortized, mesh_points, color="tab:orange", alpha=0.25)
        axis.plot(
            numerical[numerical_frontier],
            mesh_points[numerical_frontier],
            color="tab:blue",
            marker="o",
            label="Numerical frontier",
        )
        axis.plot(
            fno_amortized[fno_frontier],
            mesh_points[fno_frontier],
            color="tab:orange",
            marker="o",
            label="FNO frontier",
        )
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10.0))
        axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(f"N = {evaluations}")
        axis.set_xlabel("Amortized wall time per case [s]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Requested resolution [mesh points]")
    axes[-1].legend()
    figure.suptitle("Amortized cost-resolution Pareto frontiers")
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def plot_break_even_map(
    path: Path,
    cost_model: Mapping[str, Any],
) -> None:
    """Plot speedup across exact resolutions and evaluation counts."""
    records = cost_model["resolutions"]
    numerical = np.asarray(
        [record["numerical_solver_seconds"] for record in records],
        dtype=float,
    )
    fno_online = np.asarray(
        [record["fno_evaluation_seconds"] for record in records],
        dtype=float,
    )
    finite_break_evens = [
        float(record["break_even_evaluations"])
        for record in records
        if record["break_even_evaluations"] is not None
    ]
    maximum_cases = max(
        AMORTIZED_EVALUATION_COUNTS[-1],
        2.0 * max(finite_break_evens, default=0.0),
    )
    evaluation_counts = np.geomspace(1.0, maximum_cases, 240)
    offline_seconds = float(cost_model["offline_seconds"])
    speedup = (
        evaluation_counts[:, None] * numerical[None, :]
        / (
            offline_seconds
            + evaluation_counts[:, None] * fno_online[None, :]
        )
    )
    resolution_index = np.arange(len(records))
    positive_minimum = max(float(speedup.min()), np.finfo(float).tiny)
    maximum = float(speedup.max())
    figure, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.set_yscale("log")
    image = axis.pcolormesh(
        resolution_index,
        evaluation_counts,
        speedup,
        shading="nearest",
        cmap="coolwarm",
        norm=LogNorm(vmin=positive_minimum, vmax=maximum),
    )
    if positive_minimum <= 1.0 <= maximum:
        contour = axis.contour(
            resolution_index,
            evaluation_counts,
            speedup,
            levels=[1.0],
            colors="black",
            linewidths=2,
        )
        label_candidates = [
            (float(index), float(record["break_even_evaluations"]))
            for index, record in enumerate(records)
            if record["break_even_evaluations"] is not None
        ]
        label_location = label_candidates[len(label_candidates) // 3]
        axis.clabel(
            contour,
            fmt={1.0: "break-even"},
            inline=True,
            inline_spacing=8,
            manual=[label_location],
        )
    tick_step = max(1, math.ceil(len(records) / 12))
    ticks = resolution_index[::tick_step]
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [
            f"{records[index]['n_z']}×{records[index]['n_r']}"
            for index in ticks
        ],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Requested axial × radial resolution")
    axis.set_ylabel("Number of new evaluations, N")
    axis.set_title("FNO-to-numerical cumulative-cost speedup")
    figure.colorbar(image, ax=axis, label="Speedup")
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def _resolution_plot_summary(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Aggregate one benchmark quantity by exact axial-radial resolution."""
    return (
        frame.groupby(["n_z", "n_r"], as_index=False, sort=True)
        .agg(
            mesh_points=("mesh_points", "first"),
            mean=(column, "mean"),
            minimum=(column, "min"),
            maximum=(column, "max"),
        )
        .sort_values(["mesh_points", "n_z", "n_r"], ignore_index=True)
    )


def _set_resolution_ticks(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    maximum_ticks: int = 14,
) -> None:
    """Label an ordered categorical axial-by-radial resolution axis."""
    step = max(1, math.ceil(len(summary) / maximum_ticks))
    ticks = np.arange(0, len(summary), step)
    if ticks[-1] != len(summary) - 1:
        ticks = np.append(ticks, len(summary) - 1)
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [
            f"{int(summary.iloc[index]['n_z'])}×"
            f"{int(summary.iloc[index]['n_r'])}"
            for index in ticks
        ],
        rotation=42,
        ha="right",
    )
    axis.set_xlim(-0.6, len(summary) - 0.4)
    axis.set_xlabel("Axial × radial resolution")


def _mark_training_resolution(
    axis: plt.Axes,
    summary: pd.DataFrame,
    training_shape: tuple[int, int],
) -> None:
    """Mark the categorical resolution used to train the FNO."""
    matches = summary.index[
        (summary["n_z"] == training_shape[0])
        & (summary["n_r"] == training_shape[1])
    ].tolist()
    if not matches:
        raise ValueError(
            "The training resolution is absent from the mesh benchmark."
        )
    axis.axvline(
        matches[0],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=(
            "FNO training resolution "
            f"({training_shape[0]}×{training_shape[1]})"
        ),
    )


def plot_mesh_l2(
    path: Path,
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
) -> None:
    """Plot normalized relative L2 against exact requested resolution."""
    summary = _resolution_plot_summary(
        frame,
        "normalized_relative_l2",
    )
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    axis.plot(x, summary["mean"], marker="o", label="FNO mean")
    axis.fill_between(
        x,
        summary["minimum"],
        summary["maximum"],
        alpha=0.22,
        label="Case min–max",
    )
    axis.set_ylabel("Normalized relative L2")
    axis.set_yscale("log")
    axis.set_title("FNO error across requested mesh resolutions")
    _set_resolution_ticks(axis, summary)
    _mark_training_resolution(axis, summary, training_shape)
    axis.grid(alpha=0.28)
    axis.legend(ncols=3, loc="upper left")
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def plot_mesh_wall_time(
    path: Path,
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
    cost_model: Mapping[str, Any],
) -> None:
    """Plot online times and grouped offline-cost bars on a shared axis."""
    solver = _resolution_plot_summary(frame, "solver_wall_time_s")
    fno = _resolution_plot_summary(frame, "fno_wall_time_s")
    if not solver[["n_z", "n_r"]].equals(fno[["n_z", "n_r"]]):
        raise ValueError("Solver and FNO resolution summaries do not align.")
    x = np.arange(len(solver))
    figure, (axis, offline_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 12),
        gridspec_kw={"width_ratios": (3.2, 2.3)},
        sharey=True,
        constrained_layout=True,
    )
    for summary, label, color in (
        (solver, "Numerical solver per case", "tab:blue"),
        (fno, "FNO per case", "tab:orange"),
    ):
        axis.plot(x, summary["mean"], marker="o", color=color, label=label)
        axis.fill_between(
            x,
            summary["minimum"],
            summary["maximum"],
            color=color,
            alpha=0.18,
        )
    ray_tuning_seconds = float(cost_model["ray_tuning_seconds"])
    final_training_seconds = float(cost_model["final_training_seconds"])
    axis.set_ylabel("Wall time [s]")
    positive_minimum = float(fno["minimum"].min())
    axis.set_yscale("symlog", linthresh=max(positive_minimum / 2.0, 1.0e-6))
    axis.set_title("Online evaluation cost by resolution")
    _set_resolution_ticks(axis, solver)
    _mark_training_resolution(axis, solver, training_shape)
    axis.grid(alpha=0.28)
    axis.legend()
    # axis.legend(ncols=2, loc="upper left")

    components = (
        (
            float(cost_model["data_generation_seconds"]),
            r"Data Generation",
            "tab:blue",
        ),
        (
            ray_tuning_seconds,
            r"Tuning Model",
            "tab:purple",
        ),
        (
            final_training_seconds,
            r"Final training",
            "tab:green",
        ),
    )
    for seconds, label, _ in components:
        if not math.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(f"{label} wall time must be positive and finite.")

    bar_positions = np.array((-0.34, 0.0, 0.34))
    bars = offline_axis.bar(
        bar_positions,
        [component[0] for component in components],
        color=[component[2] for component in components],
        width=0.28,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, (seconds, label, _) in zip(bars, components):
        offline_axis.annotate(
            f"{seconds:.1f} s",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    total_offline_seconds = sum(component[0] for component in components)
    offline_axis.text(
        0.5,
        0.05,
        rf"Total offline time = {total_offline_seconds:.1f} s",
        transform=offline_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=20,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "0.35",
            "alpha": 0.94,
        },
    )
    offline_axis.set_xlim(-0.65, 0.65)
    offline_axis.set_xticks(
        bar_positions,
        [component[1] for component in components],
        fontsize=14
    )
    offline_axis.set_title("One-time offline cost components")
    offline_axis.grid(axis="y", alpha=0.28)
    offline_axis.tick_params(axis="y", labelleft=False)
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def load_json_mapping(path: Path, description: str) -> dict[str, Any]:
    """Load a required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found at {path}. Run the full workflow first."
        )
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{description} at {path} must contain a JSON object.")
    return value


def train_best_config(  # pylint: disable=too-many-arguments
    best_config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> tuple[dict[str, list[float]], float]:
    """Train and save one model using an already selected configuration."""
    train_raw, train_data = make_adapter(
        PATHS.data / f"{FILE_STEM}_train.h5",
        normalizer,
        geometry,
    )
    valid_raw, valid_data = make_adapter(
        PATHS.data / f"{FILE_STEM}_valid.h5",
        normalizer,
        geometry,
    )
    try:
        stored_training_shape = adapter_spatial_shape(
            train_data,
            f"{FILE_STEM}_train.h5",
        )
        valid_shape = adapter_spatial_shape(
            valid_data,
            f"{FILE_STEM}_valid.h5",
        )
        if stored_training_shape != training_shape:
            raise ValueError(
                f"Training mesh changed from {training_shape} to "
                f"{stored_training_shape}."
            )
        if valid_shape != training_shape:
            raise ValueError(
                f"Validation mesh {valid_shape} differs from the discovered "
                f"training mesh {training_shape}."
            )
        model, history, training_seconds = train_model(
            best_config,
            train_data,
            valid_data,
            epochs=FINAL_EPOCHS,
            device=device,
            print_epochs=True,
        )
    finally:
        train_raw.close()
        valid_raw.close()
    print(f"Final FNO training time: {training_seconds:.6f} s")

    save_checkpoint(
        PATHS.output / "fno.pt",
        model,
        best_config,
        normalizer,
        geometry,
        training_shape,
    )
    write_history(PATHS.output / "history.csv", history)
    plot_history(PATHS.output / "training_validation_loss.png", history)
    return history, training_seconds


def use_saved_model(  # pylint: disable=too-many-locals
    device: torch.device,
    *,
    calculate_metrics: bool,
    ray_tuning_seconds: float,
    final_training_seconds: float,
) -> dict[str, Any]:
    """Load the checkpoint and regenerate Q2D evaluations and figures."""
    checkpoint_path = PATHS.output / "fno.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {checkpoint_path}. Train it first."
        )
    model, normalizer, geometry, training_shape = load_checkpoint(
        checkpoint_path,
        device,
    )

    training_raw, training_data = make_adapter(
        PATHS.data / f"{FILE_STEM}_train.h5",
        normalizer,
        geometry,
    )
    try:
        stored_training_shape = adapter_spatial_shape(
            training_data,
            f"{FILE_STEM}_train.h5",
        )
    finally:
        training_raw.close()
    if stored_training_shape != training_shape:
        raise RuntimeError(
            f"Checkpoint training mesh {training_shape} differs from the "
            f"current training dataset mesh {stored_training_shape}."
        )

    test_raw, test_data = make_adapter(
        PATHS.data / f"{FILE_STEM}_test.h5",
        normalizer,
        geometry,
    )
    try:
        test_metrics = (
            evaluate(
                model,
                test_data,
                batch_size=EVALUATION_BATCH_SIZE,
                device=device,
            )
            if calculate_metrics
            else {}
        )
    finally:
        test_raw.close()

    plot_history(
        PATHS.output / "training_validation_loss.png",
        read_history(PATHS.output / "history.csv"),
    )

    mesh_files = discover_mesh_files(PATHS.data)
    reconstruction = make_reconstruction_figures(
        model,
        normalizer,
        geometry,
        mesh_files,
        training_shape,
        device=device,
    )
    benchmark_path = PATHS.output / "mesh_benchmark.csv"
    if calculate_metrics:
        benchmark = benchmark_meshes(
            model,
            normalizer,
            geometry,
            mesh_files,
            training_shape,
            device=device,
        )
        benchmark.to_csv(benchmark_path, index=False)
    else:
        if not benchmark_path.is_file():
            raise FileNotFoundError(
                f"Mesh benchmark not found at {benchmark_path}. "
                "Run the full workflow first."
            )
        benchmark = pd.read_csv(benchmark_path)
    if calculate_metrics:
        print(benchmark.to_string(index=False))
    cost_model = build_cost_model(
        benchmark,
        training_shape,
        ray_tuning_seconds=ray_tuning_seconds,
        final_training_seconds=final_training_seconds,
    )
    plot_mesh_l2(
        PATHS.output / "mesh_l2_vs_points.png",
        benchmark,
        training_shape,
    )
    plot_mesh_wall_time(
        PATHS.output / "mesh_wall_time_vs_points.png",
        benchmark,
        training_shape,
        cost_model,
    )
    plot_cumulative_break_even(
        PATHS.output / "cumulative_break_even.png",
        cost_model,
    )
    plot_amortized_pareto(
        PATHS.output / "amortized_pareto_frontiers.png",
        cost_model,
    )
    plot_break_even_map(
        PATHS.output / "break_even_speedup_map.png",
        cost_model,
    )
    return {
        **test_metrics,
        "parameters": count_parameters(model),
        "training_shape": list(training_shape),
        "training_mesh_points": math.prod(training_shape),
        "mesh_benchmark_rows": int(len(benchmark)),
        "reconstruction": reconstruction,
        "cost_model": cost_model,
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Run the selected tuning, training, or saved-model plotting workflow."""
    if TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY:
        raise ValueError(
            "TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY cannot both be true."
        )

    PATHS.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_path = PATHS.output / "metrics.json"
    if PLOT_SAVED_MODEL_ONLY:
        saved_metrics = load_json_mapping(metrics_path, "Saved metrics")
        plot_metrics = use_saved_model(
            device,
            calculate_metrics=False,
            ray_tuning_seconds=float(saved_metrics["ray_tuning_seconds"]),
            final_training_seconds=float(
                saved_metrics["final_training_seconds"]
            ),
        )
        saved_metrics.update(plot_metrics)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(saved_metrics, file, indent=2)
        print(f"Plots written to {PATHS.output}")
        return

    print("Fitting scalar training-only normalization statistics ...")
    normalizer, geometry, training_shape = fit_normalizer(
        PATHS.data / f"{FILE_STEM}_train.h5"
    )
    print(
        "Discovered FNO training mesh: "
        f"{training_shape[0]}x{training_shape[1]} "
        f"({math.prod(training_shape)} points)"
    )
    best_config_path = PATHS.output / "best_config.json"
    if TRAIN_BEST_CONFIG_ONLY:
        best_config = load_json_mapping(best_config_path, "Best configuration")
        history, training_seconds = train_best_config(
            best_config,
            normalizer,
            geometry,
            training_shape,
            device=device,
        )
        saved_metrics = (
            load_json_mapping(metrics_path, "Saved metrics")
            if metrics_path.is_file()
            else {}
        )
        saved_metrics.update(
            {
                "parameters": count_parameters(
                    load_checkpoint(PATHS.output / "fno.pt", device)[0]
                ),
                "final_training_seconds": training_seconds,
                "best_epoch": int(np.argmin(history["valid_loss"]) + 1),
                "best_validation_loss": float(min(history["valid_loss"])),
                "domain_padding": float(
                    best_config.get("domain_padding", 0.0)
                ),
                "training_shape": list(training_shape),
                "training_mesh_points": math.prod(training_shape),
            }
        )
        saved_metrics.pop("cost_model", None)
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(saved_metrics, file, indent=2)
        print(f"Model and loss history written to {PATHS.output}")
        return

    (PATHS.output / "ray_results").mkdir(parents=True, exist_ok=True)
    PATHS.ray.mkdir(parents=True, exist_ok=True)
    worker_pythonpath = os.pathsep.join(
        filter(
            None,
            (
                str((PATHS.root / "scripts").resolve()),
                str((PATHS.root / "src").resolve()),
                os.environ.get("PYTHONPATH"),
            ),
        )
    )
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(PATHS.ray.resolve()),
        runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
    )
    tune_tic = time.perf_counter()
    try:
        best_config = tune_hyperparameters(
            PATHS.data,
            PATHS.output,
            normalizer,
            geometry,
        )
    finally:
        ray.shutdown()
    tune_seconds = time.perf_counter() - tune_tic
    with best_config_path.open("w", encoding="utf-8") as file:
        json.dump(best_config, file, indent=2)

    history, training_seconds = train_best_config(
        best_config,
        normalizer,
        geometry,
        training_shape,
        device=device,
    )
    evaluation = use_saved_model(
        device,
        calculate_metrics=True,
        ray_tuning_seconds=tune_seconds,
        final_training_seconds=training_seconds,
    )

    metrics = {
        **evaluation,
        "final_training_seconds": training_seconds,
        "ray_tuning_seconds": tune_seconds,
        "best_epoch": int(np.argmin(history["valid_loss"]) + 1),
        "best_validation_loss": float(min(history["valid_loss"])),
        "domain_padding": float(best_config.get("domain_padding", 0.0)),
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {PATHS.output}")


if __name__ == "__main__":
    main()
