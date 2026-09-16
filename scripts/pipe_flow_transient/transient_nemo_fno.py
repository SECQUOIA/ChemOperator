"""Train a physics-informed PhysicsNeMo FNO for transient pipe flow.

The governing equation is selected from each generated case's
``metadata["physicsnemo_pde"]`` entry.  Supervised velocity error is combined
with the nondimensional transient Hagen--Poiseuille momentum residual and its
initial/boundary conditions.
"""

# pylint: disable=too-many-lines

# Environment variables must be set before importing plotting, Ray, and
# PhysicsNeMo.  pylint: disable=wrong-import-position

from __future__ import annotations

import argparse
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
import optuna
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
from torch import nn
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader, Subset

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.experiments import add_workflow_arguments, resolve_device
from chem_operator.models import (
    FNOAdapter,
    FNOChannel,
    fit_fno_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.reactors.pipe_flow_transient.dataset_generator import (
    TransientHagenPoiseuille,
)


PATHS = ExamplePaths.from_script(__file__, dataset="pipe_flow_transient")

FIELD_NAMES = ("velocity",)
MODEL_CONSTANT_NAMES = (
    "radius",
    "length",
    "dynamic_viscosity",
    "pressure_drop",
)
PHYSICS_CONSTANT_NAMES = MODEL_CONSTANT_NAMES + ("density",)
INPUT_CHANNELS = tuple(
    FNOChannel(name, "constant", name) for name in MODEL_CONSTANT_NAMES
)
OUTPUT_CHANNELS = (FNOChannel("velocity", "field", "velocity"),)
FILE_STEM = "transient_hagen_poiseuille_pipe_flow"
METRIC = "best_valid_relative_l2"
SEED = 42

PDE_REGISTRY: dict[str, type[PDE]] = {
    TransientHagenPoiseuille.__name__: TransientHagenPoiseuille,
}

# Run-mode flags. Leave both false for tuning, training, and evaluation.
TRAIN_BEST_CONFIG_ONLY = False
PLOT_SAVED_MODEL_ONLY = False

MAX_TRAIN_TRAJECTORIES: int | None = None
MAX_VALID_TRAJECTORIES: int | None = None
MAX_TEST_TRAJECTORIES: int | None = None
TUNE_SAMPLES = 3
TUNE_EPOCHS = 10
FINAL_EPOCHS = 15
EVALUATION_BATCH_SIZE = 8
PLOT_CASES = 2

CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
MAX_CONCURRENT_TRIALS = 1

HISTORY_FIELDS = (
    "data_loss",
    "physics_loss",
    "constraint_loss",
    "total_loss",
    "valid_relative_l2",
    "valid_physics_loss",
    "valid_constraint_loss",
)


def parse_args() -> argparse.Namespace:
    """Parse the only command-line interface used by this model script."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    return parser.parse_args()


def generate_missing_data() -> None:
    """Create absent dataset splits without replacing existing split files."""
    from scripts.pipe_flow_transient.generate_dataset import (
        pipe_flow_transient_simulator,
    )
    from chem_operator.datasets import SimulationDatasetGenerator

    generator = SimulationDatasetGenerator(pipe_flow_transient_simulator, PATHS.data)
    generated = generator.generate_missing_splits(n_cases=10_000)
    print(
        "Generated splits: " + ", ".join(generated)
        if generated
        else "All dataset splits already exist; nothing was overwritten."
    )


class PhysicsFNOAdapter(FNOAdapter):
    """Add PDE provenance and dimensional constants to an FNO sample."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Build from physical_item directly so each lazy HDF5 case is read once.
        physical = self.physical_item(index)
        metadata = physical.get("metadata", {})
        if "physicsnemo_pde" not in metadata:
            raise KeyError(
                "Dataset metadata is missing required 'physicsnemo_pde'."
            )
        parameters = metadata.get("params", {})
        try:
            physics_constants = torch.stack(
                [
                    torch.as_tensor(parameters[name], dtype=physical["x"].dtype)
                    .reshape(())
                    for name in PHYSICS_CONSTANT_NAMES
                ]
            )
        except KeyError as exc:
            raise KeyError(
                f"Dataset metadata params are missing physics constant "
                f"{exc.args[0]!r}."
            ) from exc
        model_input = torch.stack(
            [
                self.normalizer.normalize(physical["x"][channel], definition.label)
                for channel, definition in enumerate(self.input_channels)
            ]
        )
        target = torch.stack(
            [
                self.normalizer.normalize(physical["y"][channel], definition.label)
                for channel, definition in enumerate(self.output_channels)
            ]
        )
        return {
            "x": model_input,
            "y": target,
            **{name: physical[name] for name in self.coordinate_names},
            "physics_constants": physics_constants,
            "physicsnemo_pde": str(metadata["physicsnemo_pde"]),
        }


def raw_dataset(data_dir: Path, split: str) -> ChemOperatorDataset:
    """Open one complete generated transient trajectory per HDF5 case."""
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="t",
        input_fields=FIELD_NAMES,
        output_fields=FIELD_NAMES,
        constant_inputs=PHYSICS_CONSTANT_NAMES,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )


def dataset_pde_name(
    dataset: ChemOperatorDataset,
    maximum: int | None = None,
) -> str:
    """Return the single supported PDE declared by a dataset subset."""
    count = len(dataset) if maximum is None else min(len(dataset), maximum)
    if count == 0:
        raise RuntimeError("Cannot resolve a PDE from an empty dataset.")
    names: set[str] = set()
    for index in range(count):
        metadata = dataset[index].get("metadata", {})
        try:
            names.add(str(metadata["physicsnemo_pde"]))
        except KeyError as exc:
            raise KeyError(
                "Dataset metadata is missing required 'physicsnemo_pde'."
            ) from exc
    if len(names) != 1:
        raise ValueError(
            "Dataset trajectories must declare exactly one physicsnemo_pde; "
            f"found {sorted(names)}."
        )
    name = names.pop()
    if name not in PDE_REGISTRY:
        raise ValueError(
            f"Unsupported metadata physicsnemo_pde {name!r}; "
            f"supported values are {sorted(PDE_REGISTRY)}."
        )
    return name


def make_adapter(
    data_dir: Path,
    split: str,
    normalizer: ZScoreNormalizer,
    maximum: int | None,
) -> tuple[ChemOperatorDataset, PhysicsFNOAdapter]:
    """Return an open raw split and its physics-aware FNO adapter."""
    dataset = raw_dataset(data_dir, split)
    adapter = PhysicsFNOAdapter(
        dataset,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("t", "r"),
        max_trajectories=maximum,
    )
    return dataset, adapter


def fit_normalizer(data_dir: Path) -> ZScoreNormalizer:
    """Fit preprocessing statistics using only the training subset."""
    dataset = raw_dataset(data_dir, "train")
    count = len(dataset)
    if MAX_TRAIN_TRAJECTORIES is not None:
        count = min(count, MAX_TRAIN_TRAJECTORIES)
    try:
        dataset_pde_name(dataset, 1)
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
    """Reconstruct the normalizer stored with a checkpoint."""
    return ZScoreNormalizer(
        state,
        variable_field_order=FIELD_NAMES,
        constant_field_order=MODEL_CONSTANT_NAMES,
    )


def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured two-dimensional PhysicsNeMo FNO."""
    modes = int(config["modes"])
    return FNO(
        in_channels=len(MODEL_CONSTANT_NAMES),
        out_channels=len(FIELD_NAMES),
        dimension=2,
        latent_channels=int(config["latent_channels"]),
        num_fno_layers=int(config["n_layers"]),
        num_fno_modes=[modes, modes],
        padding=int(config.get("padding", 4)),
        decoder_layers=int(config.get("decoder_layers", 1)),
        decoder_layer_size=int(config.get("decoder_layer_size", 32)),
        coord_features=True,
    ).to(device)


def count_parameters(model: nn.Module) -> int:
    """Return the number of model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def make_physics_informer(
    pde_name: str,
    *,
    radial_spacing: float,
    device: torch.device,
) -> PhysicsInformer:
    """Construct a metadata-selected cylindrical momentum evaluator."""
    try:
        equation = PDE_REGISTRY[pde_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported PhysicsNeMo PDE {pde_name!r}.") from exc
    return PhysicsInformer(
        required_outputs=["momentum"],
        equations=equation,
        grad_method="meshless_finite_difference",
        fd_dx=radial_spacing,
        device=str(device),
    )


def adapter_radial_spacing(dataset: PhysicsFNOAdapter) -> float:
    """Return the first case's validated dimensionless radial spacing.

    Every training/evaluation batch is checked again by :func:`physics_losses`,
    avoiding a separate eager pass through a large lazy HDF5 split.
    """
    if len(dataset) == 0:
        raise RuntimeError("Cannot determine grid spacing from an empty adapter.")
    sample = dataset[0]
    radius = sample["physics_constants"][0]
    x = sample["r"] / radius
    differences = x[1:] - x[:-1]
    if not torch.allclose(
        differences,
        differences[0].expand_as(differences),
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("Physics-informed FNO requires a uniform radial grid.")
    return float(differences[0])


def _validate_batch_pde(batch: Mapping[str, Any], expected: str) -> None:
    names = batch["physicsnemo_pde"]
    if isinstance(names, str):
        names = [names]
    observed = {str(name) for name in names}
    if observed != {expected}:
        raise ValueError(
            f"Batch PDE metadata {sorted(observed)} does not match {expected!r}."
        )


def physics_losses(  # pylint: disable=too-many-arguments,too-many-locals
    prediction_normalized: torch.Tensor,
    batch: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    informer: PhysicsInformer,
    *,
    pde_name: str,
    radial_spacing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nondimensional PDE-residual and condition losses.

    ``TransientHagenPoiseuille`` uses ``x=r/R`` and
    ``tau=nu*t/R**2``. PhysicsInformer computes the radial derivatives from
    meshless centered stencils; the centered time derivative is supplied as
    ``velocity__t`` because the PDE intentionally declares time as an external
    derivative.
    """
    _validate_batch_pde(batch, pde_name)
    if prediction_normalized.ndim != 4 or prediction_normalized.shape[1] != 1:
        raise ValueError(
            "Expected normalized velocity with shape (batch, 1, time, radius); "
            f"received {tuple(prediction_normalized.shape)}."
        )
    if prediction_normalized.shape[-2] < 3 or prediction_normalized.shape[-1] < 3:
        raise ValueError("Physics loss requires at least three points per axis.")

    velocity = normalizer.denormalize(
        prediction_normalized[:, 0], "velocity"
    )
    constants = batch["physics_constants"].to(
        device=velocity.device,
        dtype=velocity.dtype,
    )
    radius, length, viscosity, pressure_drop, density = constants.unbind(dim=1)
    steady_max_velocity = pressure_drop * radius.square() / (
        4.0 * viscosity * length
    )
    dimensionless_velocity = velocity / steady_max_velocity[:, None, None]

    time_values = batch["t"].to(device=velocity.device, dtype=velocity.dtype)
    radial_values = batch["r"].to(device=velocity.device, dtype=velocity.dtype)
    kinematic_viscosity = viscosity / density
    tau = kinematic_viscosity[:, None] * time_values / radius[:, None].square()
    x = radial_values / radius[:, None]

    radial_differences = x[:, 1:] - x[:, :-1]
    expected_dx = torch.as_tensor(
        radial_spacing,
        device=velocity.device,
        dtype=velocity.dtype,
    )
    if not torch.allclose(
        radial_differences,
        expected_dx.expand_as(radial_differences),
        rtol=1.0e-4,
        atol=1.0e-6,
    ):
        raise ValueError("Batch does not match the PhysicsInformer radial spacing.")
    tau_step = tau[:, 2:] - tau[:, :-2]
    if torch.any(tau_step <= 0.0):
        raise ValueError("Dimensionless time coordinates must be increasing.")

    velocity_t = (
        dimensionless_velocity[:, 2:, 1:-1]
        - dimensionless_velocity[:, :-2, 1:-1]
    ) / tau_step[:, :, None]
    center = dimensionless_velocity[:, 1:-1, 1:-1]
    radial_coordinate = x[:, None, 1:-1].expand_as(center)
    residual = informer.forward(
        {
            "velocity": center,
            "velocity>>x::1": dimensionless_velocity[:, 1:-1, 2:],
            "velocity>>x::-1": dimensionless_velocity[:, 1:-1, :-2],
            "velocity__t": velocity_t,
            "x": radial_coordinate,
        }
    )["momentum"]
    physics_loss = residual.square().mean()

    initial_loss = dimensionless_velocity[:, 0, :].square().mean()
    wall_loss = dimensionless_velocity[:, :, -1].square().mean()
    center_gradient = (
        -3.0 * dimensionless_velocity[:, :, 0]
        + 4.0 * dimensionless_velocity[:, :, 1]
        - dimensionless_velocity[:, :, 2]
    ) / (2.0 * expected_dx)
    constraint_loss = (
        initial_loss + wall_loss + center_gradient.square().mean()
    ) / 3.0
    return physics_loss, constraint_loss


def relative_l2_batch(  # pylint: disable=not-callable
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the mean per-case relative L2 error."""
    error = torch.linalg.vector_norm(
        (prediction - target).flatten(start_dim=1), dim=1
    )
    scale = torch.linalg.vector_norm(
        target.flatten(start_dim=1), dim=1
    ).clamp_min(1.0e-12)
    return (error / scale).mean()


def validation_metrics(  # pylint: disable=too-many-arguments,too-many-locals
    model: nn.Module,
    dataset: PhysicsFNOAdapter,
    normalizer: ZScoreNormalizer,
    informer: PhysicsInformer,
    *,
    pde_name: str,
    radial_spacing: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate data, residual, and condition metrics on a complete split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    totals = {"relative_l2": 0.0, "physics_loss": 0.0, "constraint_loss": 0.0}
    samples = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch["x"].to(device))
            target = batch["y"].to(device)
            physics_loss, constraint_loss = physics_losses(
                prediction,
                batch,
                normalizer,
                informer,
                pde_name=pde_name,
                radial_spacing=radial_spacing,
            )
            count = target.shape[0]
            totals["relative_l2"] += float(
                relative_l2_batch(prediction, target)
            ) * count
            totals["physics_loss"] += float(physics_loss) * count
            totals["constraint_loss"] += float(constraint_loss) * count
            samples += count
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_model(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    config: Mapping[str, Any],
    train_data: PhysicsFNOAdapter,
    valid_data: PhysicsFNOAdapter,
    normalizer: ZScoreNormalizer,
    *,
    pde_name: str,
    epochs: int,
    device: torch.device,
    report_to_ray: bool = False,
    print_epochs: bool = False,
) -> tuple[FNO, dict[str, list[float]], float]:
    """Train one PI-FNO and restore its best validation checkpoint."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Subsequent batches are checked by physics_losses, so one eager metadata
    # read per split is sufficient here.
    train_pde = dataset_pde_name(train_data.dataset, 1)
    valid_pde = dataset_pde_name(valid_data.dataset, 1)
    if train_pde != pde_name or valid_pde != pde_name:
        raise ValueError(
            "Training and validation PDE metadata must agree with the selected PDE."
        )
    train_spacing = adapter_radial_spacing(train_data)
    valid_spacing = adapter_radial_spacing(valid_data)
    if abs(train_spacing - valid_spacing) > 1.0e-6:
        raise ValueError("Training and validation radial spacings must match.")

    model = model_from_config(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    informer = make_physics_informer(
        pde_name,
        radial_spacing=train_spacing,
        device=device,
    )
    physics_weight = float(config.get("physics_weight", 1.0e-2))
    constraint_weight = float(config.get("constraint_weight", 1.0e-2))
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    loader = DataLoader(
        train_data,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    history = {name: [] for name in HISTORY_FIELDS}
    best_error = float("inf")
    best_state = deepcopy(model.state_dict())
    tic = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        totals = {
            "data_loss": 0.0,
            "physics_loss": 0.0,
            "constraint_loss": 0.0,
            "total_loss": 0.0,
        }
        samples = 0
        for batch in loader:
            model_input = batch["x"].to(device)
            target = batch["y"].to(device)
            prediction = model(model_input)
            data_loss = torch_functional.mse_loss(prediction, target)
            physics_loss, constraint_loss = physics_losses(
                prediction,
                batch,
                normalizer,
                informer,
                pde_name=pde_name,
                radial_spacing=train_spacing,
            )
            loss = (
                data_loss
                + physics_weight * physics_loss
                + constraint_weight * constraint_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = target.shape[0]
            for name, value in (
                ("data_loss", data_loss),
                ("physics_loss", physics_loss),
                ("constraint_loss", constraint_loss),
                ("total_loss", loss),
            ):
                totals[name] += float(value.detach()) * count
            samples += count

        validation = validation_metrics(
            model,
            valid_data,
            normalizer,
            informer,
            pde_name=pde_name,
            radial_spacing=valid_spacing,
            batch_size=int(config["batch_size"]),
            device=device,
        )
        for name, total in totals.items():
            history[name].append(total / max(samples, 1))
        history["valid_relative_l2"].append(validation["relative_l2"])
        history["valid_physics_loss"].append(validation["physics_loss"])
        history["valid_constraint_loss"].append(validation["constraint_loss"])

        valid_error = validation["relative_l2"]
        if valid_error < best_error:
            best_error = valid_error
            best_state = deepcopy(model.state_dict())
        if print_epochs:
            print(
                f"Epoch {epoch:03d}/{epochs}: "
                f"data={history['data_loss'][-1]:.4e}, "
                f"pde={history['physics_loss'][-1]:.4e}, "
                f"bc={history['constraint_loss'][-1]:.4e}, "
                f"valid={valid_error:.4e}"
            )
        if report_to_ray:
            tune.report(
                {
                    **{name: history[name][-1] for name in HISTORY_FIELDS},
                    METRIC: best_error,
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
    pde_name: str,
) -> None:
    """Ray trainable that owns its lazy HDF5 readers."""
    torch.set_num_threads(max(1, CPUS_PER_TRIAL))
    normalizer = normalizer_from_state(normalization)
    train_raw, train_data = make_adapter(
        Path(data_dir), "train", normalizer, MAX_TRAIN_TRAJECTORIES
    )
    valid_raw, valid_data = make_adapter(
        Path(data_dir), "valid", normalizer, MAX_VALID_TRAJECTORIES
    )
    try:
        device = resolve_device("auto")
        train_model(
            config,
            train_data,
            valid_data,
            normalizer,
            pde_name=pde_name,
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
    pde_name: str,
) -> dict[str, Any]:
    """Tune architecture, optimizer, and physics weighting with Optuna/ASHA."""
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
        pde_name=pde_name,
    )
    trainable = tune.with_resources(
        parameterized,
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )
    parameter_space = {
            "modes": tune.choice([6, 8, 10]),
            "latent_channels": tune.choice([8, 16]),
            "n_layers": tune.choice([3, 4]),
            "padding": tune.choice([0, 4, 8]),
            "decoder_layers": tune.choice([1, 2]),
            "decoder_layer_size": tune.choice([8, 16, 32]),
            "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
            "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
            "batch_size": tune.choice([2, 4, 8]),
            "physics_weight": tune.loguniform(1.0e-4, 1.0e-1),
            "constraint_weight": tune.loguniform(1.0e-4, 1.0e-1),
        }
    experiment_name = "transient_pipe_flow_physicsnemo_fno"
    storage = (output_dir / "ray_results").resolve()
    experiment = storage / experiment_name
    if tune.Tuner.can_restore(str(experiment)):
        tuner = tune.Tuner.restore(
            str(experiment),
            trainable=trainable,
            resume_unfinished=True,
            resume_errored=True,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=parameter_space,
            tune_config=tune.TuneConfig(
                search_alg=search,
                scheduler=scheduler,
                num_samples=TUNE_SAMPLES,
                max_concurrent_trials=MAX_CONCURRENT_TRIALS,
                reuse_actors=False,
            ),
            run_config=tune.RunConfig(
                name=experiment_name,
                storage_path=str(storage),
                verbose=1,
            ),
        )
    best = tuner.fit().get_best_result(metric=METRIC, mode="min", scope="last")
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in best.config.items()
    }


def save_checkpoint(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    path: Path,
    model: FNO,
    config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    pde_name: str,
    radial_spacing: float,
) -> None:
    """Save model, normalization, channel, PDE, and stencil provenance."""
    torch.save(
        {
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "model_config": dict(config),
            "field_names": FIELD_NAMES,
            "model_constant_names": MODEL_CONSTANT_NAMES,
            "physics_constant_names": PHYSICS_CONSTANT_NAMES,
            "coordinate_names": ("t", "r"),
            "physicsnemo_pde": pde_name,
            "radial_spacing": float(radial_spacing),
            "normalization": normalizer_state(normalizer),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[FNO, ZScoreNormalizer, str, float]:
    """Load and strictly validate a serialized PhysicsNeMo PI-FNO."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if tuple(checkpoint["field_names"]) != FIELD_NAMES:
        raise ValueError("Checkpoint output fields do not match this workflow.")
    if tuple(checkpoint["model_constant_names"]) != MODEL_CONSTANT_NAMES:
        raise ValueError("Checkpoint input constants do not match this workflow.")
    pde_name = str(checkpoint["physicsnemo_pde"])
    if pde_name not in PDE_REGISTRY:
        raise ValueError(f"Checkpoint declares unsupported PDE {pde_name!r}.")
    model = model_from_config(checkpoint["model_config"], device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return (
        model.eval(),
        normalizer_from_state(checkpoint["normalization"]),
        pde_name,
        float(checkpoint["radial_spacing"]),
    )


def evaluate(  # pylint: disable=too-many-arguments,too-many-locals
    model: FNO,
    dataset: PhysicsFNOAdapter,
    normalizer: ZScoreNormalizer,
    *,
    pde_name: str,
    radial_spacing: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate physical-unit errors plus nondimensional physics diagnostics."""
    informer = make_physics_informer(
        pde_name, radial_spacing=radial_spacing, device=device
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    squared_error = 0.0
    squared_target = 0.0
    points = 0
    physics_total = 0.0
    constraint_total = 0.0
    samples = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prediction_n = model(batch["x"].to(device))
            target_n = batch["y"].to(device)
            physics_loss, constraint_loss = physics_losses(
                prediction_n,
                batch,
                normalizer,
                informer,
                pde_name=pde_name,
                radial_spacing=radial_spacing,
            )
            prediction = normalizer.denormalize(prediction_n, "velocity")
            target = normalizer.denormalize(target_n, "velocity")
            squared_error += float((prediction - target).square().sum())
            squared_target += float(target.square().sum())
            points += target.numel()
            count = target.shape[0]
            physics_total += float(physics_loss) * count
            constraint_total += float(constraint_loss) * count
            samples += count
    return {
        "relative_l2": (squared_error / max(squared_target, 1.0e-24)) ** 0.5,
        "rmse_m_per_s": (squared_error / max(points, 1)) ** 0.5,
        "momentum_residual_mse": physics_total / max(samples, 1),
        "condition_mse": constraint_total / max(samples, 1),
    }


def write_history(path: Path, history: Mapping[str, list[float]]) -> None:
    """Write every data and physics metric in shared long-form."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "split", "metric", "value"),
        )
        writer.writeheader()
        for index in range(len(history[HISTORY_FIELDS[0]])):
            for name in HISTORY_FIELDS:
                split = "val" if name.startswith("valid_") else "train"
                metric = name.removeprefix("valid_")
                writer.writerow(
                    {
                        "epoch": index + 1,
                        "split": split,
                        "metric": metric,
                        "value": history[name][index],
                    }
                )


def read_history(path: Path) -> dict[str, list[float]]:
    """Load a saved PI-FNO training history."""
    history = {name: [] for name in HISTORY_FIELDS}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if "split" not in row:  # Read pre-contract artifacts during migration.
                for name in HISTORY_FIELDS:
                    history[name].append(float(row[name]))
                continue
            name = (
                f"valid_{row['metric']}"
                if row["split"] in {"val", "validation"}
                else row["metric"]
            )
            if name in history:
                history[name].append(float(row["value"]))
    return history


def plot_history(path: Path, history: Mapping[str, list[float]]) -> None:
    """Plot supervised, physics, condition, total, and validation losses."""
    epochs = range(1, len(history["data_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, label in (
        ("data_loss", "Data"),
        ("physics_loss", "PDE residual"),
        ("constraint_loss", "Initial/boundary"),
        ("total_loss", "Weighted total"),
    ):
        axes[0].semilogy(epochs, history[name], label=label)
    axes[0].set_title("Physics-informed training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for name, label in (
        ("valid_relative_l2", "Relative L2"),
        ("valid_physics_loss", "PDE residual"),
        ("valid_constraint_loss", "Initial/boundary"),
    ):
        axes[1].semilogy(epochs, history[name], label=label)
    axes[1].set_title("Validation diagnostics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_reconstructions(  # pylint: disable=too-many-arguments,too-many-locals
    path: Path,
    model: FNO,
    dataset: PhysicsFNOAdapter,
    normalizer: ZScoreNormalizer,
    *,
    cases: int,
    device: torch.device,
) -> None:
    """Plot analytical truth, PI-FNO prediction, and absolute error."""
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
            prediction_n = model(sample["x"].unsqueeze(0).to(device)).cpu()[0]
        prediction = normalizer.denormalize(prediction_n, "velocity")[0].numpy()
        exact = normalizer.denormalize(sample["y"], "velocity")[0].numpy()
        error = abs(prediction - exact)
        extent = (
            float(sample["t"][0]),
            float(sample["t"][-1]),
            float(1.0e3 * sample["r"][0]),
            float(1.0e3 * sample["r"][-1]),
        )
        low = min(float(exact.min()), float(prediction.min()))
        high = max(float(exact.max()), float(prediction.max()))
        images = (
            axes[row, 0].imshow(
                exact.T, origin="lower", aspect="auto", extent=extent,
                vmin=low, vmax=high,
            ),
            axes[row, 1].imshow(
                prediction.T, origin="lower", aspect="auto", extent=extent,
                vmin=low, vmax=high,
            ),
            axes[row, 2].imshow(
                error.T, origin="lower", aspect="auto", extent=extent,
                cmap="magma",
            ),
        )
        for column, title in enumerate(
            ("Analytical", "PhysicsNeMo PI-FNO", "Absolute error")
        ):
            axes[row, column].set_title(f"Case {index}: {title}")
            axes[row, column].set_xlabel("Time [s]")
            axes[row, column].set_ylabel("Radius [mm]")
            figure.colorbar(images[column], ax=axes[row, column], label="m/s")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_best_config(path: Path) -> dict[str, Any]:
    """Load a previously tuned PhysicsNeMo FNO configuration."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Best configuration not found at {path}. Run tuning first."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def train_best_config(
    best_config: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    pde_name: str,
    device: torch.device,
) -> tuple[dict[str, list[float]], float]:
    """Train and save one PI-FNO using the selected configuration."""
    train_raw, train_data = make_adapter(
        PATHS.data, "train", normalizer, MAX_TRAIN_TRAJECTORIES
    )
    valid_raw, valid_data = make_adapter(
        PATHS.data, "valid", normalizer, MAX_VALID_TRAJECTORIES
    )
    try:
        model, history, elapsed = train_model(
            best_config,
            train_data,
            valid_data,
            normalizer,
            pde_name=pde_name,
            epochs=FINAL_EPOCHS,
            device=device,
            print_epochs=True,
        )
        spacing = adapter_radial_spacing(train_data)
    finally:
        train_raw.close()
        valid_raw.close()
    save_checkpoint(
        PATHS.output / "physicsnemo_fno.pt",
        model,
        best_config,
        normalizer,
        pde_name,
        spacing,
    )
    write_history(PATHS.output / "history.csv", history)
    return history, elapsed


def use_saved_model(
    device: torch.device,
    *,
    calculate_metrics: bool,
    plot_cases: int = PLOT_CASES,
) -> dict[str, float]:
    """Load the saved PI-FNO, validate PDE metadata, plot, and optionally score."""
    checkpoint_path = PATHS.output / "physicsnemo_fno.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {checkpoint_path}. Train it first."
        )
    model, normalizer, pde_name, spacing = load_checkpoint(checkpoint_path, device)
    test_raw, test_data = make_adapter(
        PATHS.data, "test", normalizer, MAX_TEST_TRAJECTORIES
    )
    try:
        test_pde = dataset_pde_name(test_raw, 1)
        if test_pde != pde_name:
            raise ValueError(
                f"Test PDE {test_pde!r} does not match checkpoint PDE {pde_name!r}."
            )
        actual_spacing = adapter_radial_spacing(test_data)
        if abs(actual_spacing - spacing) > 1.0e-6:
            raise ValueError("Test grid spacing does not match the checkpoint.")
        metrics = (
            evaluate(
                model,
                test_data,
                normalizer,
                pde_name=pde_name,
                radial_spacing=spacing,
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
            normalizer,
            cases=plot_cases,
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


def training_pde_name(data_dir: Path) -> str:
    """Resolve the governing PDE from the generated training data."""
    dataset = raw_dataset(data_dir, "train")
    try:
        return dataset_pde_name(dataset, 1)
    finally:
        dataset.close()


def main() -> None:
    """Run tuning, final training, evaluation, or saved-model plotting."""
    if TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY:
        raise ValueError(
            "TRAIN_BEST_CONFIG_ONLY and PLOT_SAVED_MODEL_ONLY cannot both be true."
        )
    args = parse_args()
    PATHS.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    if args.generate:
        generate_missing_data()
    if args.plot and not args.tune and not args.train:
        use_saved_model(device, calculate_metrics=False, plot_cases=args.plot_cases)
        print(f"Plots written to {PATHS.output}")
        return

    pde_name = training_pde_name(PATHS.data)
    print(f"Using metadata physicsnemo_pde={pde_name!r}")
    best_config_path = PATHS.output / "best_config.json"
    normalizer = None
    best_config: dict[str, Any] | None = None
    if args.tune:
        print("Fitting training-only normalization statistics ...")
        normalizer = fit_normalizer(PATHS.data)
        (PATHS.output / "ray_results").mkdir(parents=True, exist_ok=True)
        PATHS.ray.mkdir(parents=True, exist_ok=True)
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            _temp_dir=str(PATHS.ray.resolve()),
        )
        try:
            best_config = tune_hyperparameters(
                PATHS.data, PATHS.output, normalizer, pde_name
            )
        finally:
            ray.shutdown()
        with best_config_path.open("w", encoding="utf-8") as file:
            json.dump(best_config, file, indent=2)

    history: dict[str, list[float]] | None = None
    elapsed = 0.0
    if args.train:
        best_config = best_config or load_best_config(
            best_config_path
            if args.train_config == "best"
            else Path(args.train_config)
        )
        if normalizer is None:
            print("Fitting training-only normalization statistics ...")
            normalizer = fit_normalizer(PATHS.data)
        history, elapsed = train_best_config(
            best_config, normalizer, pde_name, device
        )
    if args.plot:
        metrics = use_saved_model(
            device,
            calculate_metrics=args.train,
            plot_cases=args.plot_cases,
        )
        if history is not None:
            metrics.update(
                {
                    "training_seconds": elapsed,
                    "best_epoch": int(
                        torch.tensor(history["valid_relative_l2"]).argmin().item()
                        + 1
                    ),
                    "best_validation_relative_l2": min(
                        history["valid_relative_l2"]
                    ),
                    "physicsnemo_pde": pde_name,
                    "parameters": count_parameters(
                        load_checkpoint(
                            PATHS.output / "physicsnemo_fno.pt", device
                        )[0]
                    ),
                }
            )
            with (PATHS.output / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2)
            print(json.dumps(metrics, indent=2))
    print(f"Selected stages completed in {PATHS.output}")


if __name__ == "__main__":
    main()
