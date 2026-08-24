"""Compare data-driven and physics-informed DeepONets for pipe flow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import os
from pathlib import Path
import time
from typing import Any, Mapping

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.pipe_flow.dataset_generator import (
    HagenPoiseuille,
    hagen_poiseuille_velocity,
)


PATHS = ExamplePaths.from_script(__file__, dataset="pipe_flow")
BRANCH_NAMES = ("radius", "length", "dynamic_viscosity", "pressure_drop")
ALL_CONSTANTS = BRANCH_NAMES + ("density", "pressure_gradient")
METRIC = "valid_relative_l2"


@dataclass(frozen=True)
class Normalization:
    branch_mean: torch.Tensor
    branch_std: torch.Tensor
    velocity_mean: torch.Tensor
    velocity_std: torch.Tensor


@dataclass(frozen=True)
class PipeData:
    branch: torch.Tensor
    coordinates: torch.Tensor
    radius: torch.Tensor
    viscosity: torch.Tensor
    pressure_gradient: torch.Tensor
    target: torch.Tensor
    target_normalized: torch.Tensor | None = None

    def normalized(self, statistics: Normalization) -> "PipeData":
        return replace(
            self,
            branch=(self.branch - statistics.branch_mean)
            / statistics.branch_std,
            target_normalized=(self.target - statistics.velocity_mean)
            / statistics.velocity_std,
        )

    def dataset(self) -> TensorDataset:
        if self.target_normalized is None:
            raise RuntimeError("Normalize a split before constructing a loader.")
        return TensorDataset(
            self.branch,
            self.coordinates,
            self.radius,
            self.viscosity,
            self.pressure_gradient,
            self.target_normalized,
            self.target,
        )


class DeepONet(nn.Module):
    """PhysicsNeMo branch/trunk MLPs with a scalar DeepONet product."""

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        latent_width: int,
        activation: str,
    ) -> None:
        super().__init__()
        options = {
            "layer_size": width,
            "out_features": latent_width,
            "num_layers": depth,
            "activation_fn": activation,
        }
        self.branch = FullyConnected(in_features=len(BRANCH_NAMES), **options)
        self.trunk = FullyConnected(in_features=1, **options)
        self.bias = nn.Parameter(torch.zeros(1))

    def points(
        self,
        branch: torch.Tensor,
        coordinates: torch.Tensor,
        radius: torch.Tensor,
        case_index: torch.Tensor,
    ) -> torch.Tensor:
        branch_latent = self.branch(branch)[case_index]
        case_radius = radius.reshape(-1, 1)[case_index]
        relative_coordinate = 2.0 * coordinates / case_radius - 1.0
        trunk_latent = self.trunk(relative_coordinate)
        return (branch_latent * trunk_latent).sum(dim=-1, keepdim=True) + self.bias

    def forward(
        self,
        branch: torch.Tensor,
        coordinates: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_points = coordinates.shape[:2]
        case_index = torch.arange(batch_size, device=branch.device).repeat_interleave(
            n_points
        )
        values = self.points(
            branch,
            coordinates.reshape(-1, 1),
            radius,
            case_index,
        )
        return values.reshape(batch_size, n_points, 1)


def project_path(value: str) -> Path:
    return PATHS.resolve(value)


def load_split(data_dir: Path, split: str, maximum: int | None) -> PipeData:
    torch.set_default_device("cpu")
    dataset = ChemOperatorDataset(
        data_dir / f"hagen_poiseuille_pipe_flow_{split}.h5",
        task="operator_cartesian",
        coordinate_name="r",
        input_fields=("velocity",),
        output_fields=("velocity",),
        constant_inputs=ALL_CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )
    count = len(dataset) if maximum is None else min(maximum, len(dataset))
    branch: list[torch.Tensor] = []
    coordinates: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    viscosities: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    try:
        for index in range(count):
            sample = dataset[index]
            constants = sample["constant_inputs"]
            branch.append(torch.stack([constants[name] for name in BRANCH_NAMES]))
            radii.append(constants["radius"].reshape(1))
            viscosities.append(constants["dynamic_viscosity"].reshape(1))
            gradients.append(constants["pressure_gradient"].reshape(1))
            coordinates.append(
                torch.cat(
                    (
                        sample["input_coordinates"]["r"],
                        sample["output_coordinates"]["r"],
                    )
                ).reshape(-1, 1)
            )
            targets.append(
                torch.cat(
                    (
                        sample["input_fields"]["velocity"],
                        sample["output_fields"]["velocity"],
                    )
                ).reshape(-1, 1)
            )
    finally:
        dataset.close()
    return PipeData(
        branch=torch.stack(branch),
        coordinates=torch.stack(coordinates),
        radius=torch.stack(radii),
        viscosity=torch.stack(viscosities),
        pressure_gradient=torch.stack(gradients),
        target=torch.stack(targets),
    )


def fit_normalization(data: PipeData) -> Normalization:
    return Normalization(
        branch_mean=data.branch.mean(dim=0),
        branch_std=data.branch.std(dim=0, unbiased=False).clamp_min(1.0e-8),
        velocity_mean=data.target.mean(),
        velocity_std=data.target.std(unbiased=False).clamp_min(1.0e-8),
    )


def model_from_config(config: Mapping[str, Any], device: torch.device) -> DeepONet:
    return DeepONet(
        width=int(config["width"]),
        depth=int(config["depth"]),
        latent_width=int(config["latent_width"]),
        activation=str(config["activation"]),
    ).to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.linalg.vector_norm((prediction - target).reshape(-1))
    scale = torch.linalg.vector_norm(target.reshape(-1)).clamp_min(1.0e-12)
    return error / scale


def physics_mse(
    model: DeepONet,
    physics: PhysicsInformer,
    branch: torch.Tensor,
    coordinates: torch.Tensor,
    radius: torch.Tensor,
    viscosity: torch.Tensor,
    pressure_gradient: torch.Tensor,
    statistics: Normalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized predictions and a dimensionless residual MSE."""

    batch_size, n_points = coordinates.shape[:2]
    point_coordinates = coordinates.reshape(-1, 1).detach().clone()
    point_coordinates.requires_grad_(True)
    case_index = torch.arange(batch_size, device=branch.device).repeat_interleave(
        n_points
    )
    prediction_normalized = model.points(
        branch, point_coordinates, radius, case_index
    )
    velocity = (
        prediction_normalized * statistics.velocity_std.to(branch.device)
        + statistics.velocity_mean.to(branch.device)
    )
    point_radius = radius[case_index]
    point_viscosity = viscosity[case_index]
    point_gradient = pressure_gradient[case_index]
    residual = physics.forward(
        {
            "coordinates": point_coordinates,
            "x": point_coordinates,
            "velocity": velocity,
            "dynamic_viscosity": point_viscosity,
            "pressure_gradient": point_gradient,
        }
    )["momentum"]
    residual_scale = (point_gradient.abs() * point_radius).clamp_min(1.0e-12)
    return (
        prediction_normalized.reshape(batch_size, n_points, 1),
        (residual / residual_scale).square().mean(),
    )


def validation_error(
    model: DeepONet,
    data: PipeData,
    statistics: Normalization,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    squared_error = 0.0
    squared_target = 0.0
    loader = DataLoader(data.dataset(), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for branch, coordinates, radius, _, _, _, target in loader:
            prediction = model(
                branch.to(device), coordinates.to(device), radius.to(device)
            )
            prediction = (
                prediction * statistics.velocity_std.to(device)
                + statistics.velocity_mean.to(device)
            )
            target = target.to(device)
            squared_error += float((prediction - target).square().sum())
            squared_target += float(target.square().sum())
    return (squared_error / max(squared_target, 1.0e-24)) ** 0.5


def train_model(
    config: Mapping[str, Any],
    train_data: PipeData,
    validation_data: PipeData,
    statistics: Normalization,
    *,
    physics_weight: float,
    epochs: int,
    device: torch.device,
    report_to_ray: bool,
) -> tuple[DeepONet, dict[str, list[float]], float]:
    # Ray and PhysicsNeMo may leave a CUDA default-device context active in a
    # worker. DataLoader indices and its RNG must remain CPU-side.
    torch.set_default_device("cpu")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = model_from_config(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    informer = (
        PhysicsInformer(
            required_outputs=["momentum"],
            equations=HagenPoiseuille(),
            grad_method="autodiff",
            device=str(device),
        )
        if physics_weight > 0.0
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = DataLoader(
        train_data.dataset(),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history = {"data_loss": [], "physics_loss": [], "valid_relative_l2": []}
    best_error = float("inf")
    best_state = deepcopy(model.state_dict())
    tic = time.perf_counter()

    for _epoch in range(1, epochs + 1):
        model.train()
        data_total = 0.0
        physics_total = 0.0
        samples = 0
        for branch, coordinates, radius, viscosity, gradient, target_n, _ in loader:
            branch = branch.to(device)
            coordinates = coordinates.to(device)
            radius = radius.to(device)
            target_n = target_n.to(device)
            if informer is None:
                prediction_n = model(branch, coordinates, radius)
                residual_loss = prediction_n.new_zeros(())
            else:
                prediction_n, residual_loss = physics_mse(
                    model,
                    informer,
                    branch,
                    coordinates,
                    radius,
                    viscosity.to(device),
                    gradient.to(device),
                    statistics,
                )
            data_loss = torch.nn.functional.mse_loss(prediction_n, target_n)
            loss = data_loss + physics_weight * residual_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = branch.shape[0]
            data_total += float(data_loss.detach()) * count
            physics_total += float(residual_loss.detach()) * count
            samples += count

        valid_error = validation_error(
            model,
            validation_data,
            statistics,
            int(config["batch_size"]),
            device,
        )
        history["data_loss"].append(data_total / samples)
        history["physics_loss"].append(physics_total / samples)
        history["valid_relative_l2"].append(valid_error)
        if valid_error < best_error:
            best_error = valid_error
            best_state = deepcopy(model.state_dict())
        if report_to_ray:
            tune.report(
                {
                    METRIC: valid_error,
                    "data_loss": history["data_loss"][-1],
                    "physics_loss": history["physics_loss"][-1],
                    "n_params": count_parameters(model),
                }
            )

    elapsed = time.perf_counter() - tic
    model.load_state_dict(best_state)
    return model.eval(), history, elapsed


def ray_trial(
    config: Mapping[str, Any],
    *,
    train_data: PipeData,
    validation_data: PipeData,
    statistics: Normalization,
    epochs: int,
    physics_weight: float | None = None,
    fixed_config: Mapping[str, Any] | None = None,
) -> None:
    model_config = dict(config if fixed_config is None else fixed_config)
    weight = float(config["physics_weight"]) if physics_weight is None else physics_weight
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model_config,
        train_data,
        validation_data,
        statistics,
        physics_weight=weight,
        epochs=epochs,
        device=device,
        report_to_ray=True,
    )


def tuner(
    trainable,
    parameter_space: Mapping[str, Any],
    *,
    name: str,
    samples: int,
    epochs: int,
    cfg: DictConfig,
    storage: Path,
) -> tune.Tuner:
    scheduler = ASHAScheduler(
        metric=METRIC,
        mode="min",
        max_t=epochs,
        grace_period=min(int(cfg.tune.grace_period), epochs),
        reduction_factor=int(cfg.tune.reduction_factor),
    )
    return tune.Tuner(
        tune.with_resources(
            trainable,
            resources={
                "cpu": int(cfg.ray.cpus_per_trial),
                "gpu": float(cfg.ray.gpus_per_trial)
                if torch.cuda.is_available()
                else 0,
            },
        ),
        param_space=dict(parameter_space),
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            num_samples=samples,
            max_concurrent_trials=int(cfg.ray.max_concurrent_trials),
        ),
        run_config=tune.RunConfig(
            name=name,
            storage_path=str(storage.resolve()),
            verbose=int(cfg.ray.verbose),
        ),
    )


def best_result(results: tune.ResultGrid) -> tune.Result:
    return results.get_best_result(metric=METRIC, mode="min", scope="last")


def serializable(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in values.items()
    }


def save_checkpoint(
    path: Path,
    model: DeepONet,
    config: Mapping[str, Any],
    statistics: Normalization,
    physics_weight: float,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": serializable(config),
            "branch_names": BRANCH_NAMES,
            "normalization": {
                "branch_mean": statistics.branch_mean,
                "branch_std": statistics.branch_std,
                "velocity_mean": statistics.velocity_mean,
                "velocity_std": statistics.velocity_std,
            },
            "physics_weight": physics_weight,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[DeepONet, Normalization]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = model_from_config(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["state_dict"])
    normalizer = Normalization(**checkpoint["normalization"])
    return model.eval(), normalizer


def predict(
    model: DeepONet,
    data: PipeData,
    statistics: Normalization,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    torch.set_default_device("cpu")
    predictions: list[torch.Tensor] = []
    loader = DataLoader(data.dataset(), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for branch, coordinates, radius, *_ in loader:
            value = model(branch.to(device), coordinates.to(device), radius.to(device))
            value = value * statistics.velocity_std.to(device) + statistics.velocity_mean.to(
                device
            )
            predictions.append(value.detach().to(device="cpu"))
    torch.set_default_device("cpu")
    return torch.cat(predictions).to(device="cpu")


def residual_metric(
    model: DeepONet,
    data: PipeData,
    statistics: Normalization,
    batch_size: int,
    device: torch.device,
) -> float:
    physics = PhysicsInformer(
        ["momentum"], HagenPoiseuille(), "autodiff", device=str(device)
    )
    total = 0.0
    points = 0
    loader = DataLoader(data.dataset(), batch_size=batch_size, shuffle=False)
    for branch, coordinates, radius, viscosity, gradient, *_ in loader:
        _, loss = physics_mse(
            model,
            physics,
            branch.to(device),
            coordinates.to(device),
            radius.to(device),
            viscosity.to(device),
            gradient.to(device),
            statistics,
        )
        count = branch.shape[0] * coordinates.shape[1]
        total += float(loss.detach()) * count
        points += count
    return (total / points) ** 0.5


def evaluate(
    model: DeepONet,
    data: PipeData,
    statistics: Normalization,
    batch_size: int,
    device: torch.device,
    elapsed: float,
) -> tuple[dict[str, float | int], torch.Tensor]:
    prediction = predict(model, data, statistics, batch_size, device).to(device="cpu")
    target = data.target.to(device="cpu")
    return (
        {
            "relative_l2": float(relative_l2(prediction, target)),
            "rmse": float((prediction - target).square().mean().sqrt()),
            "normalized_physics_residual": residual_metric(
                model, data, statistics, batch_size, device
            ),
            "parameters": count_parameters(model),
            "training_seconds": elapsed,
        },
        prediction,
    )


def plot_history(
    output: Path,
    data_history: Mapping[str, list[float]],
    physics_history: Mapping[str, list[float]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    epochs = range(1, len(data_history["data_loss"]) + 1)
    axes[0].semilogy(epochs, data_history["data_loss"], label="Data-driven")
    axes[0].semilogy(epochs, physics_history["data_loss"], label="PI")
    axes[0].set_title("Normalized data MSE")
    axes[1].semilogy(
        epochs, physics_history["physics_loss"], color="tab:orange", label="PI"
    )
    axes[1].set_title("PI normalized residual MSE")
    axes[2].semilogy(epochs, data_history["valid_relative_l2"], label="Data-driven")
    axes[2].semilogy(epochs, physics_history["valid_relative_l2"], label="PI")
    axes[2].set_title("Validation relative L2")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_predictions(
    output: Path,
    data: PipeData,
    data_prediction: torch.Tensor,
    physics_prediction: torch.Tensor,
    cases: int,
) -> None:
    cases = min(cases, len(data.target))
    figure, axes = plt.subplots(cases, 1, figsize=(7, 3.2 * cases), squeeze=False)
    for index, axis in enumerate(axes[:, 0]):
        radius_mm = (
            1.0e3 * data.coordinates[index, :, 0].detach().to("cpu").numpy()
        )
        exact = data.target[index, :, 0].detach().to("cpu").numpy()
        data_values = data_prediction[index, :, 0].detach().to("cpu").numpy()
        physics_values = physics_prediction[index, :, 0].detach().to("cpu").numpy()
        axis.plot(radius_mm, exact, "k", label="Exact")
        axis.plot(radius_mm, data_values, "--", label="Data-driven")
        axis.plot(radius_mm, physics_values, ":", label="PI")
        axis.set_xlabel("Radius [mm]")
        axis.set_ylabel("Velocity [m/s]")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def verify_exact_residual() -> None:
    coordinate = torch.linspace(0.0, 1.0e-3, 32, dtype=torch.float64).reshape(-1, 1)
    coordinate.requires_grad_(True)
    viscosity = torch.full_like(coordinate, 1.0e-3)
    gradient = torch.full_like(coordinate, -80.0)
    velocity = hagen_poiseuille_velocity(
        coordinate,
        radius=1.0e-3,
        dynamic_viscosity=viscosity,
        pressure_gradient=gradient,
    )
    physics = PhysicsInformer(["momentum"], HagenPoiseuille(), "autodiff")
    residual = physics.forward(
        {
            "coordinates": coordinate,
            "x": coordinate,
            "velocity": velocity,
            "dynamic_viscosity": viscosity,
            "pressure_gradient": gradient,
        }
    )["momentum"]
    if float(residual.detach().abs().max()) > 1.0e-10:
        raise RuntimeError("PhysicsInformer failed the exact-profile residual check.")


@hydra.main(version_base="1.3", config_path=".", config_name="deeponet")
def main(cfg: DictConfig) -> None:
    torch.set_default_device("cpu")
    verify_exact_residual()
    output_dir = project_path(str(cfg.output.directory))
    storage = output_dir / "ray_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    storage.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "run_config.yaml")

    maximum = cfg.data.max_trajectories
    maximum = None if maximum is None else int(maximum)
    data_dir = project_path(str(cfg.data.directory))
    print("Loading pipe-flow profiles ...")
    train_raw = load_split(data_dir, "train", maximum)
    validation_raw = load_split(data_dir, "valid", maximum)
    test_raw = load_split(data_dir, "test", maximum)
    statistics = fit_normalization(train_raw)
    train_data = train_raw.normalized(statistics)
    validation_data = validation_raw.normalized(statistics)
    test_data = test_raw.normalized(statistics)

    ray_temp = project_path(str(cfg.ray.temp_directory))
    ray_temp.mkdir(parents=True, exist_ok=True)
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=str(ray_temp.resolve()),
    )
    try:
        base_trainable = tune.with_parameters(
            ray_trial,
            train_data=train_data,
            validation_data=validation_data,
            statistics=statistics,
            epochs=int(cfg.tune.max_epochs),
            physics_weight=0.0,
        )
        search_space = {
            "width": tune.choice(list(cfg.search.width)),
            "depth": tune.choice(list(cfg.search.depth)),
            "latent_width": tune.choice(list(cfg.search.latent_width)),
            "activation": tune.choice(list(cfg.search.activation)),
            "learning_rate": tune.loguniform(*map(float, cfg.search.learning_rate)),
            "weight_decay": tune.loguniform(*map(float, cfg.search.weight_decay)),
            "batch_size": tune.choice(list(cfg.search.batch_size)),
            "seed": int(cfg.seed),
        }
        base_results = tuner(
            base_trainable,
            search_space,
            name="data_driven",
            samples=int(cfg.tune.num_samples),
            epochs=int(cfg.tune.max_epochs),
            cfg=cfg,
            storage=storage,
        ).fit()
        best_base = best_result(base_results)
        best_config = serializable(best_base.config)

        physics_trainable = tune.with_parameters(
            ray_trial,
            train_data=train_data,
            validation_data=validation_data,
            statistics=statistics,
            epochs=int(cfg.tune.max_epochs),
            fixed_config=best_config,
        )
        physics_results = tuner(
            physics_trainable,
            {
                "physics_weight": tune.loguniform(
                    float(cfg.physics.min_weight), float(cfg.physics.max_weight)
                )
            },
            name="physics_weight",
            samples=int(cfg.physics.num_samples),
            epochs=int(cfg.tune.max_epochs),
            cfg=cfg,
            storage=storage,
        ).fit()
        best_physics = best_result(physics_results)
        physics_weight = float(best_physics.config["physics_weight"])
    finally:
        ray.shutdown()

    best_summary = {
        "shared_model": best_config,
        "data_driven_validation_relative_l2": float(best_base.metrics[METRIC]),
        "physics_weight": physics_weight,
        "physics_informed_validation_relative_l2": float(
            best_physics.metrics[METRIC]
        ),
    }
    OmegaConf.save(OmegaConf.create(best_summary), output_dir / "best_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_epochs = int(cfg.final.epochs)
    data_model, data_history, data_seconds = train_model(
        best_config,
        train_data,
        validation_data,
        statistics,
        physics_weight=0.0,
        epochs=final_epochs,
        device=device,
        report_to_ray=False,
    )
    physics_model, physics_history, physics_seconds = train_model(
        best_config,
        train_data,
        validation_data,
        statistics,
        physics_weight=physics_weight,
        epochs=final_epochs,
        device=device,
        report_to_ray=False,
    )
    data_checkpoint = output_dir / "data_driven.pt"
    physics_checkpoint = output_dir / "physics_informed.pt"
    save_checkpoint(data_checkpoint, data_model, best_config, statistics, 0.0)
    save_checkpoint(
        physics_checkpoint,
        physics_model,
        best_config,
        statistics,
        physics_weight,
    )

    # Evaluate the serialized artifacts, not the in-memory training objects.
    data_model, data_statistics = load_checkpoint(data_checkpoint, device)
    physics_model, physics_statistics = load_checkpoint(physics_checkpoint, device)
    evaluation_batch_size = int(cfg.final.evaluation_batch_size)
    data_metrics, data_prediction = evaluate(
        data_model,
        test_data,
        data_statistics,
        evaluation_batch_size,
        device,
        data_seconds,
    )
    physics_metrics, physics_prediction = evaluate(
        physics_model,
        test_data,
        physics_statistics,
        evaluation_batch_size,
        device,
        physics_seconds,
    )
    if data_metrics["parameters"] != physics_metrics["parameters"]:
        raise RuntimeError("The compared models do not have matching architectures.")
    metrics = {
        "data_driven": data_metrics,
        "physics_informed": {
            **physics_metrics,
            "physics_weight": physics_weight,
        },
        "relative_l2_change_percent": 100.0
        * (
            float(physics_metrics["relative_l2"])
            / float(data_metrics["relative_l2"])
            - 1.0
        ),
    }
    OmegaConf.save(OmegaConf.create(metrics), output_dir / "metrics.yaml")
    plot_history(
        output_dir / "training_validation_loss.png", data_history, physics_history
    )
    plot_predictions(
        output_dir / "test_reconstructions.png",
        test_data,
        data_prediction,
        physics_prediction,
        int(cfg.final.plot_cases),
    )
    print(OmegaConf.to_yaml(OmegaConf.create(metrics)))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
