"""Tune and train a 2D FNO for transient Hagen--Poiseuille flow."""

# Environment variables must be set before importing plotting and Ray.
# pylint: disable=wrong-import-position

from __future__ import annotations

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import matplotlib.pyplot as plt
from neuralop.losses import LpLoss
from neuralop.models import FNO
import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import (
    FNOAdapter,
    FNOChannel,
    fit_fno_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="pipe_flow_transient")

FIELD_NAMES = ("velocity",)
CONSTANT_NAMES = (
    "radius",
    "length",
    "dynamic_viscosity",
    "pressure_drop",
)
INPUT_CHANNELS = tuple(
    FNOChannel(name, "constant", name) for name in CONSTANT_NAMES
)
OUTPUT_CHANNELS = tuple(
    FNOChannel(name, "field", name) for name in FIELD_NAMES
)
FILE_STEM = "transient_hagen_poiseuille_pipe_flow"
METRIC = "best_valid_loss"
SEED = 42

# Run-mode flags. Leave both false for tuning, training, and evaluation.
TRAIN_BEST_CONFIG_ONLY = False
PLOT_SAVED_MODEL_ONLY = True

# CPU-conscious defaults. Set a limit to None to use an entire split.
MAX_TRAIN_TRAJECTORIES: int | None = None #2000
MAX_VALID_TRAJECTORIES: int | None = None #500
MAX_TEST_TRAJECTORIES: int | None = None #500
TUNE_SAMPLES = 12
TUNE_EPOCHS = 6
FINAL_EPOCHS = 10
EVALUATION_BATCH_SIZE = 4
PLOT_CASES = 2

CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
MAX_CONCURRENT_TRIALS = 1


def raw_dataset(data_dir: Path, split: str) -> ChemOperatorDataset:
    """Open one complete transient trajectory per HDF5 case."""
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="t",
        input_fields=FIELD_NAMES,
        output_fields=FIELD_NAMES,
        constant_inputs=CONSTANT_NAMES,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )


def make_adapter(
    data_dir: Path,
    split: str,
    normalizer: ZScoreNormalizer,
    maximum: int | None,
) -> tuple[ChemOperatorDataset, FNOAdapter]:
    """Return an open raw dataset and its FNO adapter."""
    dataset = raw_dataset(data_dir, split)
    adapter = FNOAdapter(
        dataset,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("t", "r"),
        max_trajectories=maximum,
    )
    return dataset, adapter


def fit_normalizer(data_dir: Path) -> ZScoreNormalizer:
    """Fit all preprocessing statistics using only the training subset."""
    dataset = raw_dataset(data_dir, "train")
    count = len(dataset)
    if MAX_TRAIN_TRAJECTORIES is not None:
        count = min(count, MAX_TRAIN_TRAJECTORIES)
    try:
        return fit_fno_zscore_normalizer(
            Subset(dataset, range(count)),
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        )
    finally:
        dataset.close()


def normalizer_state(
    normalizer: ZScoreNormalizer,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return the serializable state needed to reconstruct a normalizer."""
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
    """Reconstruct the ChemOperator normalizer stored with a checkpoint."""
    return ZScoreNormalizer(
        state,
        variable_field_order=FIELD_NAMES,
        constant_field_order=CONSTANT_NAMES,
    )


def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured NeuralOperator FNO."""
    modes = int(config["modes"])
    return FNO(
        n_modes=(modes, modes),
        in_channels=len(CONSTANT_NAMES),
        out_channels=len(FIELD_NAMES),
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
    ).to(device)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable and frozen model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def validation_loss(
    model: nn.Module,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return mean 2D relative L2 loss for a complete split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loss_function = LpLoss(d=2, p=2, reduction="mean")
    model.eval()
    total = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["x"].to(device))
            target = batch["y"].to(device)
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
    torch.set_default_device("cpu")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = model_from_config(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_function = LpLoss(d=2, p=2, reduction="mean")
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
) -> None:
    """Ray trainable that opens its own lazy HDF5 readers."""
    torch.set_num_threads(max(1, CPUS_PER_TRIAL))
    normalizer = normalizer_from_state(normalization)
    train_raw, train_data = make_adapter(
        Path(data_dir),
        "train",
        normalizer,
        MAX_TRAIN_TRAJECTORIES,
    )
    valid_raw, valid_data = make_adapter(
        Path(data_dir),
        "valid",
        normalizer,
        MAX_VALID_TRAJECTORIES,
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
) -> dict[str, Any]:
    """Run Optuna search with ASHA early stopping and return the best config."""
    search = OptunaSearch(
        metric=METRIC,
        mode="min",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=2,
            multivariate=True,
        ),
    )
    scheduler = ASHAScheduler(
        metric=METRIC,
        mode="min",
        time_attr="training_iteration",
        max_t=TUNE_EPOCHS,
        grace_period=max(1, TUNE_EPOCHS // 3),
        reduction_factor=2,
    )
    parameterized = tune.with_parameters(
        ray_trial,
        data_dir=str(data_dir.resolve()),
        normalization=normalizer_state(normalizer),
    )
    trainable = tune.with_resources(
        parameterized,
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )
    tuner = tune.Tuner(
        trainable,
        param_space={
            "modes": tune.choice([8, 12]),
            "hidden_channels": tune.choice([8, 16]),
            "n_layers": 3, #tune.choice([2, 3]),
            "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
            "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
            "batch_size": tune.choice([2, 4, 8, 16, 32]),
        },
        tune_config=tune.TuneConfig(
            search_alg=search,
            scheduler=scheduler,
            num_samples=TUNE_SAMPLES,
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name="transient_pipe_flow_fno",
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
) -> None:
    """Save the model, configuration, and training-only normalization."""
    # NeuralOperator adds a callable-rich ``_metadata`` entry to state_dict.
    # The model configuration below reconstructs that metadata, so persist only
    # tensors and retain compatibility with PyTorch's safe weights-only loader.
    tensor_state = {
        name: value
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    torch.save(
        {
            "state_dict": tensor_state,
            "model_config": dict(config),
            "field_names": FIELD_NAMES,
            "constant_names": CONSTANT_NAMES,
            "normalization": normalizer_state(normalizer),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FNO, ZScoreNormalizer]:
    """Load a serialized FNO and its normalization state."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = model_from_config(checkpoint["model_config"], device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "FNO checkpoint is missing model tensors: "
            f"{incompatible.missing_keys}."
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "FNO checkpoint has unexpected tensors: "
            f"{incompatible.unexpected_keys}."
        )
    return model.eval(), normalizer_from_state(checkpoint["normalization"])


def evaluate(
    model: FNO,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate global physical-unit relative L2 and RMSE."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    squared_error = 0.0
    squared_target = 0.0
    points = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["x"].to(device)).cpu()
            target = batch["y"]
            prediction = dataset.denormalize_output(prediction)
            target = dataset.denormalize_output(target)
            squared_error += float((prediction - target).square().sum())
            squared_target += float(target.square().sum())
            points += target.numel()
    return {
        "relative_l2": (squared_error / max(squared_target, 1.0e-24)) ** 0.5,
        "rmse": (squared_error / max(points, 1)) ** 0.5,
    }


def write_history(
    path: Path,
    history: Mapping[str, list[float]],
) -> None:
    """Write per-epoch training and validation losses as CSV."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "train_loss", "valid_loss"),
        )
        writer.writeheader()
        for index, (train_value, valid_value) in enumerate(
            zip(history["train_loss"], history["valid_loss"]),
            start=1,
        ):
            writer.writerow(
                {
                    "epoch": index,
                    "train_loss": train_value,
                    "valid_loss": valid_value,
                }
            )


def plot_history(
    path: Path,
    history: Mapping[str, list[float]],
) -> None:
    """Plot training and validation loss against epoch."""
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.semilogy(epochs, history["train_loss"], label="Training")
    axis.semilogy(epochs, history["valid_loss"], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Relative L2 loss")
    axis.set_title("Transient pipe-flow FNO")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def read_history(path: Path) -> dict[str, list[float]]:
    """Load previously saved per-epoch losses."""
    history = {"train_loss": [], "valid_loss": []}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            history["train_loss"].append(float(row["train_loss"]))
            history["valid_loss"].append(float(row["valid_loss"]))
    return history


def plot_reconstructions(  # pylint: disable=too-many-locals
    path: Path,
    model: FNO,
    dataset: FNOAdapter,
    *,
    cases: int,
    device: torch.device,
) -> None:
    """Plot exact, reconstructed, and absolute-error velocity surfaces."""
    cases = min(cases, len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, cases).round().int().tolist()
    figure, axes = plt.subplots(
        cases,
        3,
        figsize=(12, 3.3 * cases),
        squeeze=False,
        constrained_layout=True,
    )
    model.eval()
    for row, index in enumerate(indices):
        sample = dataset[index]
        with torch.no_grad():
            prediction = model(sample["x"].unsqueeze(0).to(device)).cpu()[0]
        prediction = dataset.denormalize_output(prediction)[0].numpy()
        exact = dataset.denormalize_output(sample["y"])[0].numpy()
        error = abs(prediction - exact)
        radius_mm = 1.0e3 * sample["r"].numpy()
        time_values = sample["t"].numpy()
        extent = (
            float(time_values[0]),
            float(time_values[-1]),
            float(radius_mm[0]),
            float(radius_mm[-1]),
        )
        low = min(float(exact.min()), float(prediction.min()))
        high = max(float(exact.max()), float(prediction.max()))
        images = (
            axes[row, 0].imshow(
                exact.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=low,
                vmax=high,
            ),
            axes[row, 1].imshow(
                prediction.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                vmin=low,
                vmax=high,
            ),
            axes[row, 2].imshow(
                error.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
            ),
        )
        for column, title in enumerate(("Exact", "FNO", "Absolute error")):
            axes[row, column].set_title(f"Case {index}: {title}")
            axes[row, column].set_xlabel("Time [s]")
            axes[row, column].set_ylabel("Radius [mm]")
            figure.colorbar(images[column], ax=axes[row, column], label="m/s")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_best_config(path: Path) -> dict[str, Any]:
    """Load a previously tuned FNO configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Best configuration not found at {path}. Run tuning first."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def train_best_config(
    best_config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    device: torch.device,
) -> tuple[dict[str, list[float]], float]:
    """Train and save one model using an already selected configuration."""
    train_raw, train_data = make_adapter(
        PATHS.data,
        "train",
        normalizer,
        MAX_TRAIN_TRAJECTORIES,
    )
    valid_raw, valid_data = make_adapter(
        PATHS.data,
        "valid",
        normalizer,
        MAX_VALID_TRAJECTORIES,
    )
    try:
        model, history, elapsed = train_model(
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

    save_checkpoint(
        PATHS.output / "fno.pt",
        model,
        best_config,
        normalizer,
    )
    write_history(PATHS.output / "history.csv", history)
    plot_history(PATHS.output / "training_validation_loss.png", history)
    return history, elapsed


def use_saved_model(
    device: torch.device,
    *,
    calculate_metrics: bool,
) -> dict[str, float]:
    """Load the saved model, make reconstruction plots, and optionally score."""
    checkpoint_path = PATHS.output / "fno.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {checkpoint_path}. Train it first."
        )
    model, normalizer = load_checkpoint(checkpoint_path, device)
    test_raw, test_data = make_adapter(
        PATHS.data,
        "test",
        normalizer,
        MAX_TEST_TRAJECTORIES,
    )
    try:
        metrics = (
            evaluate(
                model,
                test_data,
                batch_size=EVALUATION_BATCH_SIZE,
                device=device,
            )
            if calculate_metrics
            else {}
        )
        plot_reconstructions(
            PATHS.output / "test_reconstructions.png",
            model,
            test_data,
            cases=PLOT_CASES,
            device=device,
        )
    finally:
        test_raw.close()

    history_path = PATHS.output / "history.csv"
    if history_path.is_file():
        plot_history(
            PATHS.output / "training_validation_loss.png",
            read_history(history_path),
        )
    return metrics


def main() -> None:
    """Run the selected tuning, training, or saved-model plotting workflow."""
    if TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY:
        raise ValueError(
            "TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY cannot both be true."
        )

    PATHS.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if PLOT_SAVED_MODEL_ONLY:
        use_saved_model(device, calculate_metrics=False)
        print(f"Plots written to {PATHS.output}")
        return

    print("Fitting training-only normalization statistics ...")
    normalizer = fit_normalizer(PATHS.data)
    best_config_path = PATHS.output / "best_config.json"
    if TRAIN_BEST_CONFIG_ONLY:
        best_config = load_best_config(best_config_path)
        train_best_config(best_config, normalizer, device)
        print(f"Model and loss history written to {PATHS.output}")
        return

    (PATHS.output / "ray_results").mkdir(parents=True, exist_ok=True)
    PATHS.ray.mkdir(parents=True, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(PATHS.ray.resolve()),
    )
    try:
        best_config = tune_hyperparameters(PATHS.data, PATHS.output, normalizer)
    finally:
        ray.shutdown()
    with best_config_path.open("w", encoding="utf-8") as file:
        json.dump(best_config, file, indent=2)

    history, elapsed = train_best_config(best_config, normalizer, device)
    metrics = {
        **use_saved_model(device, calculate_metrics=True),
        "training_seconds": elapsed,
        "best_epoch": int(
            torch.tensor(history["valid_loss"]).argmin().item() + 1
        ),
        "best_validation_loss": min(history["valid_loss"]),
    }
    model, _ = load_checkpoint(PATHS.output / "fno.pt", device)
    metrics["parameters"] = count_parameters(model)
    with (PATHS.output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Results written to {PATHS.output}")


if __name__ == "__main__":
    main()
