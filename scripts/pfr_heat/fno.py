"""Train coupled PhysicsNeMo FNOs for a non-isothermal PFR and its wall.

A 1D FNO predicts ``(F_A, F_B, F_C, T_g)(z)``.  Its gas-temperature
prediction is broadcast across radius and supplied to a 2D FNO predicting
``T_s(z, r)``.  Supervised, PDE, and boundary losses train both networks.
"""

# pylint: disable=wrong-import-position,too-many-lines,too-many-locals
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=too-many-arguments,too-many-positional-arguments,not-callable
# pylint: disable=import-error,import-outside-toplevel,consider-using-enumerate
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")
os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp")

import matplotlib.pyplot as plt
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ray import tune
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.experiments import (
    RayRuntimeConfig,
    RunContext,
    RunPaths,
    Tuner,
    TuningConfig,
    add_workflow_arguments,
    resolve_device,
)
from chem_operator.normalization import ZScoreNormalizer, normalizer_from_state_dict
from chem_operator.reactors.pfr_heat.dataset_generator import (
    CylindricalWall,
    ModelConstants,
    PlugFlowReactor,
)

PATHS = ExamplePaths.from_script(__file__, dataset="pfr_heat")
FILE_STEM = "pfr_cylindrical_heat"
CHECKPOINT = "coupled_fno.pt"
SEED = 42
METRIC = "best_valid_relative_l2"

PFR_INPUTS = (
    "inlet_flow_a", "inlet_concentration_a", "inlet_temperature",
    "outer_temperature", "volumetric_heat_transfer", "wall_aspect_ratio_sq",
    "interface_biot", "inner_radius", "outer_radius",
)
PFR_OUTPUTS = ("F_A", "F_B", "F_C", "T_gas")
WALL_CONDITIONS = (
    "outer_temperature", "wall_aspect_ratio_sq", "interface_biot",
    "inner_radius", "outer_radius",
)
WALL_GAS = "wall_T_gas"
WALL_OUTPUTS = ("T_solid",)
PHYSICS_CONSTANTS = PFR_INPUTS + ("flow_scale", "temperature_scale")
EXPECTED_PDES = {"reactor": "PlugFlowReactor", "solid": "CylindricalWall"}

TUNE_SAMPLES = 6
TUNE_EPOCHS = 40
FINAL_EPOCHS = 75
EVALUATION_BATCH_SIZE = 4
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = int(torch.cuda.is_available())
MAX_CONCURRENT_TRIALS = 1
HISTORY_KEYS = (
    "data_loss", "species_loss", "gas_loss", "solid_loss", "bc_loss",
    "total_loss", "valid_relative_l2", "valid_species_loss",
    "valid_gas_loss", "valid_solid_loss", "valid_bc_loss",
)


def generate_missing_data() -> None:
    """Create absent splits without replacing existing solver output."""
    generator_path = Path(__file__).with_name("generate_dataset.py")
    specification = importlib.util.spec_from_file_location(
        "pfr_heat_generate_dataset", generator_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load dataset generator at {generator_path}.")
    generator_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(generator_module)
    from chem_operator.datasets import SimulationDatasetGenerator

    generator = SimulationDatasetGenerator(
        generator_module.pfr_heat_simulator, PATHS.data, seed=SEED
    )
    generated = generator.generate_missing_splits(n_cases=20)
    print(
        "Generated splits: " + ", ".join(generated)
        if generated
        else "All dataset splits already exist; nothing was overwritten."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    return parser.parse_args()

def raw_dataset(data_dir: Path, split: str) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="field_map", coordinate_name="z",
        input_fields=("F", "T_gas", "T_solid"),
        output_fields=("F", "T_gas", "T_solid"),
        constant_inputs=PHYSICS_CONSTANTS, n_steps_input=1,
        dtype=torch.float32,
    )


def complete_field(sample: Mapping[str, Any], name: str) -> torch.Tensor:
    return torch.cat((sample["input_fields"][name], sample["output_fields"][name]))


def coordinate(sample: Mapping[str, Any], name: str) -> torch.Tensor:
    first = sample["input_coordinates"][name].reshape(-1)
    second = sample["output_coordinates"][name].reshape(-1)
    if name == "z":
        return torch.cat((first, second))
    if first.shape != second.shape or not torch.equal(first, second):
        raise ValueError(f"Coordinate {name!r} changes within a steady case.")
    return first


def uniform_spacing(values: torch.Tensor, name: str) -> float:
    if values.ndim != 1 or values.numel() < 5:
        raise ValueError(f"{name} needs at least five 1D points.")
    differences = values[1:] - values[:-1]
    if torch.any(differences <= 0) or not torch.allclose(
        differences, differences[0].expand_as(differences), rtol=1e-5, atol=1e-7
    ):
        raise ValueError(f"{name} must be strictly increasing and uniform.")
    return float(differences[0])


def validate_sample(
    sample: Mapping[str, Any], expected: tuple[int, int] | None = None
) -> tuple[int, int]:
    metadata = sample.get("metadata", {})
    if metadata.get("physicsnemo_pdes") != EXPECTED_PDES:
        raise ValueError("Dataset PDE provenance does not match the coupled model.")
    if tuple(metadata.get("species", ())) != ("A", "B", "C"):
        raise ValueError("Species metadata must be ordered as A, B, C.")
    missing = [n for n in PHYSICS_CONSTANTS if n not in sample["constant_inputs"]]
    if missing:
        raise KeyError(f"Missing constants: {', '.join(missing)}")
    z, r = coordinate(sample, "z"), coordinate(sample, "r")
    uniform_spacing(z, "z")
    uniform_spacing(r, "r")
    shape = (z.numel(), r.numel())
    shapes = {
        "F": tuple(complete_field(sample, "F").shape),
        "T_gas": tuple(complete_field(sample, "T_gas").shape),
        "T_solid": tuple(complete_field(sample, "T_solid").shape),
    }
    wanted = {"F": (shape[0], 3), "T_gas": (shape[0],), "T_solid": shape}
    if shapes != wanted:
        raise ValueError(f"Invalid coupled field shapes {shapes}; expected {wanted}.")
    if expected is not None and shape != expected:
        raise ValueError(f"Expected grid {expected}, received {shape}.")
    return shape


class Moments:
    def __init__(self) -> None:
        self.count = 0
        self.total = torch.tensor(0.0, dtype=torch.float64)
        self.squares = torch.tensor(0.0, dtype=torch.float64)

    def update(self, value: torch.Tensor) -> None:
        value = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
        self.count += value.numel()
        self.total += value.sum()
        self.squares += value.square().sum()

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.count:
            raise RuntimeError("Cannot normalize an empty field.")
        mean = self.total / self.count
        variance = (self.squares / self.count - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float()


def make_normalizer(
    moments: Mapping[str, Moments], outputs: Sequence[str], inputs: Sequence[str]
) -> ZScoreNormalizer:
    pairs = {name: value.result() for name, value in moments.items()}
    return ZScoreNormalizer(
        {
            "mean": {name: value[0] for name, value in pairs.items()},
            "std": {name: value[1] for name, value in pairs.items()},
            "mean_delta": {name: torch.tensor(0.0) for name in outputs},
            "std_delta": {name: torch.tensor(1.0) for name in outputs},
        }, outputs, inputs,
    )


def fit_normalizers(
    dataset: ChemOperatorDataset,
) -> tuple[ZScoreNormalizer, ZScoreNormalizer, tuple[int, int]]:
    pfr_moments = {name: Moments() for name in PFR_INPUTS + PFR_OUTPUTS}
    wall_moments = {
        name: Moments()
        for name in (WALL_GAS,) + WALL_CONDITIONS + WALL_OUTPUTS
    }
    shape = None
    for index in range(len(dataset)):
        sample = dataset[index]
        current = validate_sample(sample, shape)
        shape = shape or current
        values = sample["constant_inputs"]
        flows, gas, solid = (
            complete_field(sample, "F"), complete_field(sample, "T_gas"),
            complete_field(sample, "T_solid"),
        )
        for name in PFR_INPUTS:
            pfr_moments[name].update(values[name])
        for channel, name in enumerate(PFR_OUTPUTS[:3]):
            pfr_moments[name].update(flows[:, channel])
        pfr_moments["T_gas"].update(gas)
        wall_moments[WALL_GAS].update(gas)
        for name in WALL_CONDITIONS:
            wall_moments[name].update(values[name])
        wall_moments["T_solid"].update(solid)
    if shape is None:
        raise RuntimeError("Training dataset is empty.")
    return (
        make_normalizer(pfr_moments, PFR_OUTPUTS, PFR_INPUTS),
        make_normalizer(
            wall_moments, WALL_OUTPUTS, (WALL_GAS,) + WALL_CONDITIONS
        ),
        shape,
    )


class CoupledPFRHeatDataset(Dataset):
    """Normalized, lazy coupled view of a PFR-heat HDF5 split."""

    def __init__(self, dataset, pfr_normalizer, wall_normalizer, shape):
        self.dataset = dataset
        self.pfr_normalizer = pfr_normalizer
        self.wall_normalizer = wall_normalizer
        self.shape = tuple(shape)
        for index in range(len(dataset)):
            validate_sample(dataset[index], self.shape)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        n_z, n_r = self.shape
        constants = sample["constant_inputs"]
        flows = complete_field(sample, "F")
        gas = complete_field(sample, "T_gas")
        solid = complete_field(sample, "T_solid")
        pfr_x = torch.stack([
            self.pfr_normalizer.normalize(constants[name], name).expand(n_z)
            for name in PFR_INPUTS
        ])
        pfr_y = torch.stack([
            self.pfr_normalizer.normalize(value, name)
            for name, value in zip(
                PFR_OUTPUTS, (flows[:, 0], flows[:, 1], flows[:, 2], gas),
                strict=True,
            )
        ])
        wall_conditions = torch.stack([
            self.wall_normalizer.normalize(constants[name], name).expand(n_z, n_r)
            for name in WALL_CONDITIONS
        ])
        return {
            "pfr_x": pfr_x, "pfr_y": pfr_y,
            "wall_conditions": wall_conditions,
            "wall_y": self.wall_normalizer.normalize(solid, "T_solid").unsqueeze(0),
            "z": coordinate(sample, "z"), "r": coordinate(sample, "r"),
            "physics_constants": torch.stack([
                constants[name].reshape(()) for name in PHYSICS_CONSTANTS
            ]),
        }


class CoupledFNO(nn.Module):
    """Acyclic PFR-to-wall pair of PhysicsNeMo FNOs."""

    def __init__(self, config, pfr_normalizer, wall_normalizer):
        super().__init__()
        self.pfr_normalizer = pfr_normalizer
        self.wall_normalizer = wall_normalizer
        common = {
            "latent_channels": int(config["latent_channels"]),
            "num_fno_layers": int(config["n_layers"]),
            "padding": int(config.get("padding", 4)),
            "decoder_layers": int(config.get("decoder_layers", 1)),
            "decoder_layer_size": int(config.get("decoder_layer_size", 32)),
            "coord_features": True,
        }
        self.pfr = FNO(
            in_channels=len(PFR_INPUTS), out_channels=4, dimension=1,
            num_fno_modes=[int(config["pfr_modes"])], **common,
        )
        self.wall = FNO(
            in_channels=1 + len(WALL_CONDITIONS), out_channels=1, dimension=2,
            num_fno_modes=[int(config["wall_modes_z"]), int(config["wall_modes_r"])],
            **common,
        )

    def forward(self, pfr_x, wall_conditions) -> dict[str, torch.Tensor]:
        pfr = self.pfr(pfr_x)
        gas = self.pfr_normalizer.denormalize(pfr[:, 3], "T_gas")
        gas = self.wall_normalizer.normalize(gas, WALL_GAS)
        gas = gas[..., None].expand(-1, -1, wall_conditions.shape[-1])
        wall = self.wall(torch.cat((gas[:, None], wall_conditions), dim=1))
        return {"pfr": pfr, "wall": wall}


def model_from_config(config, pfr_normalizer, wall_normalizer, device):
    return CoupledFNO(config, pfr_normalizer, wall_normalizer).to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class InformerCache:
    """Case-coefficient cache for finite-difference PhysicsInformers."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reactors = {}
        self.walls = {}

    def reactor(self, c: Sequence[float], dz: float) -> PhysicsInformer:
        key = (c[1], c[2], c[4], dz)
        if key not in self.reactors:
            self.reactors[key] = PhysicsInformer(
                ["flow_a", "flow_b", "flow_c", "gas_energy"],
                PlugFlowReactor(ModelConstants(), c[1], c[2], c[4]),
                "finite_difference", fd_dx=dz, device=str(self.device),
            )
        return self.reactors[key]

    def wall(self, c: Sequence[float], dz: float, dr: float) -> PhysicsInformer:
        key = (c[5], c[6], dz, dr)
        if key not in self.walls:
            self.walls[key] = PhysicsInformer(
                ["solid_heat"], CylindricalWall(c[5], c[6]),
                "finite_difference", fd_dx=[dz, dr], device=str(self.device),
            )
        return self.walls[key]


def physical_predictions(prediction, pfr_normalizer, wall_normalizer):
    flows = torch.stack([
        pfr_normalizer.denormalize(prediction["pfr"][:, i], name)
        for i, name in enumerate(PFR_OUTPUTS[:3])
    ], dim=1)
    gas = pfr_normalizer.denormalize(prediction["pfr"][:, 3], "T_gas")
    solid = wall_normalizer.denormalize(prediction["wall"][:, 0], "T_solid")
    return flows, gas, solid


def physics_losses(prediction, batch, pfr_normalizer, wall_normalizer, cache):
    """Return species, gas-energy, wall-conduction, and BC losses."""
    flows, gas, solid = physical_predictions(
        prediction, pfr_normalizer, wall_normalizer
    )
    constants = batch["physics_constants"].to(flows.device, flows.dtype)
    z, r = batch["z"].to(flows.device), batch["r"].to(flows.device)
    totals = {name: flows.new_zeros(()) for name in ("species", "gas", "solid", "bc")}
    for i in range(flows.shape[0]):
        c = [float(v) for v in constants[i].detach().cpu()]
        dz, dr = float(z[i, 1] - z[i, 0]), float(r[i, 1] - r[i, 0])
        flow_scale, temperature_scale = c[-2:]
        fi = flows[i:i + 1] / flow_scale
        tg = gas[i:i + 1, None] / temperature_scale
        ts = solid[i:i + 1, None] / temperature_scale
        residuals = cache.reactor(c, dz).forward({
            "f_a": fi[:, 0:1], "f_b": fi[:, 1:2], "f_c": fi[:, 2:3],
            "t_gas": tg, "t_wall": ts[..., 0],
        })
        totals["species"] += torch.stack([
            residuals[name][..., 2:-2].square().mean()
            for name in ("flow_a", "flow_b", "flow_c")
        ]).mean()
        totals["gas"] += residuals["gas_energy"][..., 2:-2].square().mean()
        radial = r[i][None, None, None, :].expand_as(ts)
        wall_residual = cache.wall(c, dz, dr).forward(
            {"t_solid": ts, "y": radial}
        )["solid_heat"]
        totals["solid"] += wall_residual[..., 2:-2, 2:-2].square().mean()

        inlet = fi.new_tensor([c[0] / flow_scale, 0.0, 0.0])
        inlet_flow = (fi[0, :, 0] - inlet).square().mean()
        inlet_gas = (tg[0, 0, 0] - c[2] / temperature_scale).square()
        outer = (ts[0, 0, :, -1] - c[3] / temperature_scale).square().mean()
        radial_gradient = (-3 * ts[..., 0] + 4 * ts[..., 1] - ts[..., 2]) / (2 * dr)
        interface = (radial_gradient - c[6] * (ts[..., 0] - tg)).square().mean()
        flux_0 = (-3 * ts[..., 0, :] + 4 * ts[..., 1, :] - ts[..., 2, :]) / (2 * dz)
        flux_1 = (3 * ts[..., -1, :] - 4 * ts[..., -2, :] + ts[..., -3, :]) / (2 * dz)
        axial = 0.5 * (flux_0.square().mean() + flux_1.square().mean())
        totals["bc"] += torch.stack((inlet_flow, inlet_gas, outer, interface, axial)).mean()
    return {name: value / flows.shape[0] for name, value in totals.items()}


def supervised_loss(prediction, batch):
    return 0.5 * (
        F.mse_loss(prediction["pfr"], batch["pfr_y"])
        + F.mse_loss(prediction["wall"], batch["wall_y"])
    )


def relative_channels(prediction: torch.Tensor, target: torch.Tensor):
    error = torch.linalg.vector_norm((prediction - target).flatten(2), dim=2)
    scale = torch.linalg.vector_norm(target.flatten(2), dim=2).clamp_min(1e-12)
    return error / scale


def validation_metrics(model, dataset, pfr_normalizer, wall_normalizer, batch_size, device):
    totals = {name: 0.0 for name in ("relative_l2", "species", "gas", "solid", "bc")}
    samples, cache = 0, InformerCache(device)
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction = model(batch["pfr_x"], batch["wall_conditions"])
            predicted = physical_predictions(prediction, pfr_normalizer, wall_normalizer)
            target = physical_predictions(
                {"pfr": batch["pfr_y"], "wall": batch["wall_y"]},
                pfr_normalizer, wall_normalizer,
            )
            p1 = torch.cat((predicted[0], predicted[1][:, None]), dim=1)
            p2 = torch.cat((target[0], target[1][:, None]), dim=1)
            errors = torch.cat((
                relative_channels(p1, p2),
                relative_channels(predicted[2][:, None], target[2][:, None]),
            ), dim=1).mean(1)
            losses = physics_losses(
                prediction, batch, pfr_normalizer, wall_normalizer, cache
            )
            count = p1.shape[0]
            totals["relative_l2"] += float(errors.sum())
            for name in losses:
                totals[name] += float(losses[name]) * count
            samples += count
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_model(
    config, train_data, valid_data, pfr_normalizer, wall_normalizer, *,
    epochs, device, reporter=None, print_epochs=False,
):
    torch.manual_seed(SEED)
    model = model_from_config(config, pfr_normalizer, wall_normalizer, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loader = DataLoader(
        train_data, batch_size=int(config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    history = {name: [] for name in HISTORY_KEYS}
    best_error, best_state = float("inf"), deepcopy(model.state_dict())
    cache, tic = InformerCache(device), time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {name: 0.0 for name in HISTORY_KEYS[:6]}
        samples = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            prediction = model(batch["pfr_x"], batch["wall_conditions"])
            data = supervised_loss(prediction, batch)
            physics = physics_losses(
                prediction, batch, pfr_normalizer, wall_normalizer, cache
            )
            total = data + sum(
                float(config[key]) * physics[name]
                for key, name in (
                    ("lambda_f", "species"), ("lambda_g", "gas"),
                    ("lambda_s", "solid"), ("lambda_bc", "bc"),
                )
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            count = batch["pfr_x"].shape[0]
            values = {"data_loss": data, "total_loss": total}
            values.update({f"{name}_loss": value for name, value in physics.items()})
            for name, value in values.items():
                sums[name] += float(value.detach()) * count
            samples += count
        for name in HISTORY_KEYS[:6]:
            history[name].append(sums[name] / samples)
        valid = validation_metrics(
            model, valid_data, pfr_normalizer, wall_normalizer,
            int(config["batch_size"]), device,
        )
        history["valid_relative_l2"].append(valid["relative_l2"])
        for name in ("species", "gas", "solid", "bc"):
            history[f"valid_{name}_loss"].append(valid[name])
        if valid["relative_l2"] < best_error:
            best_error, best_state = valid["relative_l2"], deepcopy(model.state_dict())
        if print_epochs:
            pde_value = sum(
                history[name][-1]
                for name in ("species_loss", "gas_loss", "solid_loss")
            )
            print(
                f"Epoch {epoch:03d}/{epochs}: data={history['data_loss'][-1]:.3e}, "
                f"physics={pde_value:.3e}, "
                f"bc={history['bc_loss'][-1]:.3e}, valid={valid['relative_l2']:.3e}"
            )
        if reporter is not None:
            reporter({
                **{name: history[name][-1] for name in HISTORY_KEYS},
                METRIC: best_error, "n_params": count_parameters(model),
            })
    model.load_state_dict(best_state)
    return model.eval(), history, time.perf_counter() - tic


def make_adapter(data_dir, split, pfr_normalizer, wall_normalizer, shape):
    raw = raw_dataset(Path(data_dir), split)
    return raw, CoupledPFRHeatDataset(raw, pfr_normalizer, wall_normalizer, shape)


def ray_trial(config, *, data_dir, pfr_state, wall_state, shape, reporter):
    pfr = normalizer_from_state_dict(pfr_state)
    wall = normalizer_from_state_dict(wall_state)
    train_raw, train = make_adapter(data_dir, "train", pfr, wall, shape)
    valid_raw, valid = make_adapter(data_dir, "valid", pfr, wall, shape)
    try:
        train_model(
            config, train, valid, pfr, wall, epochs=TUNE_EPOCHS,
            device=resolve_device("auto"), reporter=reporter,
        )
    finally:
        train_raw.close()
        valid_raw.close()


def tune_hyperparameters(data_dir, output_dir, pfr, wall, shape):
    """Run the local search space using shared Ray/Optuna mechanics."""
    space = {
        "pfr_modes": tune.choice([8, 12, 16]),
        "wall_modes_z": tune.choice([8, 12, 16]),
        "wall_modes_r": tune.choice([4, 6, 8]),
        "latent_channels": tune.choice([8, 16, 24]),
        "n_layers": tune.choice([3, 4]), "padding": tune.choice([0, 4]),
        "decoder_layers": tune.choice([1, 2]),
        "decoder_layer_size": tune.choice([16, 32]),
        "learning_rate": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-4),
        "batch_size": tune.choice([1, 2, 4]),
        "lambda_f": tune.loguniform(1e-4, 1e-1),
        "lambda_g": tune.loguniform(1e-4, 1e-1),
        "lambda_s": tune.loguniform(1e-4, 1e-1),
        "lambda_bc": tune.loguniform(1e-4, 1e-1),
    }
    storage = (output_dir / "ray_results").resolve()
    pfr_state, wall_state = pfr.state_dict(), wall.state_dict()

    def objective(config, _train, _validation, _context, report):
        return ray_trial(
            config,
            data_dir=str(data_dir.resolve()),
            pfr_state=pfr_state,
            wall_state=wall_state,
            shape=shape,
            reporter=report,
        )

    tuner = Tuner.from_objective(
        objective,
        space,
        config=TuningConfig(
            metric=METRIC,
            mode="min",
            num_samples=TUNE_SAMPLES,
            max_epochs=TUNE_EPOCHS,
            resources_per_trial={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
            grace_period=TUNE_EPOCHS // 3,
            optuna_seed=SEED,
            optuna_startup_trials=3,
            ray_runtime=RayRuntimeConfig(temp_dir=PATHS.ray.resolve()),
        ),
    )
    context = RunContext(
        RunPaths(output_dir), SEED, torch.float32, resolve_device("auto")
    )
    return tuner.fit(
        None,
        None,
        context,
        storage_path=storage,
        experiment_name="pfr_heat_coupled_physicsnemo_fno",
    )


def save_checkpoint(path, model, config, pfr, wall, shape):
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": dict(config), "pfr_normalization": pfr.state_dict(),
        "wall_normalization": wall.state_dict(), "pfr_input_names": PFR_INPUTS,
        "pfr_output_names": PFR_OUTPUTS,
        "wall_input_names": (WALL_GAS,) + WALL_CONDITIONS,
        "wall_output_names": WALL_OUTPUTS, "coordinate_names": ("z", "r"),
        "training_shape": tuple(shape), "physicsnemo_pdes": EXPECTED_PDES,
    }, path)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    expected = {
        "pfr_input_names": PFR_INPUTS, "pfr_output_names": PFR_OUTPUTS,
        "wall_input_names": (WALL_GAS,) + WALL_CONDITIONS,
        "wall_output_names": WALL_OUTPUTS, "coordinate_names": ("z", "r"),
    }
    for key, value in expected.items():
        if tuple(checkpoint.get(key, ())) != value:
            raise ValueError(f"Checkpoint {key} does not match this workflow.")
    if checkpoint.get("physicsnemo_pdes") != EXPECTED_PDES:
        raise ValueError("Checkpoint PDE provenance does not match this workflow.")
    shape = tuple(int(value) for value in checkpoint["training_shape"])
    if len(shape) != 2 or min(shape) < 5:
        raise ValueError("Checkpoint training shape is invalid.")
    pfr = normalizer_from_state_dict(checkpoint["pfr_normalization"])
    wall = normalizer_from_state_dict(checkpoint["wall_normalization"])
    config = dict(checkpoint["model_config"])
    model = model_from_config(config, pfr, wall, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), pfr, wall, shape, config


def train_best_config(config, pfr, wall, shape, device):
    train_raw, train = make_adapter(PATHS.data, "train", pfr, wall, shape)
    valid_raw, valid = make_adapter(PATHS.data, "valid", pfr, wall, shape)
    try:
        model, history, elapsed = train_model(
            config, train, valid, pfr, wall, epochs=FINAL_EPOCHS,
            device=device, print_epochs=True,
        )
    finally:
        train_raw.close()
        valid_raw.close()
    save_checkpoint(PATHS.output / CHECKPOINT, model, config, pfr, wall, shape)
    with (PATHS.output / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    return history, elapsed


def evaluate(model, dataset, pfr, wall, device):
    """Calculate per-channel physical relative errors and physics metrics."""
    totals = {name: 0.0 for name in PFR_OUTPUTS + WALL_OUTPUTS}
    samples = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=EVALUATION_BATCH_SIZE):
            batch = {k: v.to(device) for k, v in batch.items()}
            predicted = physical_predictions(
                model(batch["pfr_x"], batch["wall_conditions"]), pfr, wall
            )
            target = physical_predictions(
                {"pfr": batch["pfr_y"], "wall": batch["wall_y"]}, pfr, wall
            )
            pfr_errors = relative_channels(
                torch.cat((predicted[0], predicted[1][:, None]), 1),
                torch.cat((target[0], target[1][:, None]), 1),
            )
            wall_errors = relative_channels(
                predicted[2][:, None], target[2][:, None]
            )
            for i, name in enumerate(PFR_OUTPUTS):
                totals[name] += float(pfr_errors[:, i].sum())
            totals["T_solid"] += float(wall_errors.sum())
            samples += pfr_errors.shape[0]
    metrics = {f"relative_l2_{k}": v / samples for k, v in totals.items()}
    metrics["relative_l2"] = sum(metrics.values()) / len(totals)
    physics = validation_metrics(
        model, dataset, pfr, wall, EVALUATION_BATCH_SIZE, device
    )
    metrics.update({f"{k}_loss": v for k, v in physics.items() if k != "relative_l2"})
    return metrics


def plot_history(history):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name in ("data_loss", "total_loss", "valid_relative_l2"):
        axes[0].plot(history[name], label=name)
    for name in ("species", "gas", "solid", "bc"):
        axes[1].plot(history[f"valid_{name}_loss"], label=name)
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Epoch")
        axis.legend()
    figure.savefig(PATHS.output / "training_history.png", dpi=180)
    plt.close(figure)


def plot_cases(model, dataset, pfr, wall, count, device):
    for case in range(min(count, len(dataset))):
        sample = dataset[case]
        with torch.no_grad():
            prediction = physical_predictions(model(
                sample["pfr_x"][None].to(device),
                sample["wall_conditions"][None].to(device),
            ), pfr, wall)
            target = physical_predictions({
                "pfr": sample["pfr_y"][None].to(device),
                "wall": sample["wall_y"][None].to(device),
            }, pfr, wall)
        z, r = sample["z"].numpy(), sample["r"].numpy()
        predicted_pfr = torch.cat((prediction[0], prediction[1][:, None]), 1)[0].cpu()
        target_pfr = torch.cat((target[0], target[1][:, None]), 1)[0].cpu()
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        for axis, i, name in zip(axes.flat, range(4), PFR_OUTPUTS, strict=True):
            axis.plot(z, target_pfr[i], label="truth")
            axis.plot(z, predicted_pfr[i], "--", label="FNO")
            axis.set_title(name)
            axis.legend()
        figure.savefig(PATHS.output / f"case_{case:02d}_pfr.png", dpi=180)
        plt.close(figure)
        truth, estimate = target[2][0].cpu().numpy(), prediction[2][0].cpu().numpy()
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
        for axis, field, title in zip(
            axes, (truth, estimate, estimate - truth), ("Truth", "FNO", "Error"),
            strict=True,
        ):
            image = axis.pcolormesh(z, r, field.T, shading="auto")
            axis.set_title(title)
            figure.colorbar(image, ax=axis)
        figure.savefig(PATHS.output / f"case_{case:02d}_wall.png", dpi=180)
        plt.close(figure)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found at {path}.")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def use_saved_model(device, calculate_metrics, cases):
    model, pfr, wall, shape, _ = load_checkpoint(PATHS.output / CHECKPOINT, device)
    raw, dataset = make_adapter(PATHS.data, "test", pfr, wall, shape)
    try:
        metrics = (
            evaluate(model, dataset, pfr, wall, device)
            if calculate_metrics else {}
        )
        plot_cases(model, dataset, pfr, wall, cases, device)
    finally:
        raw.close()
    history_path = PATHS.output / "history.json"
    if history_path.is_file():
        plot_history(load_json(history_path))
    return metrics


def main() -> None:
    args = parse_args()
    PATHS.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    if args.generate:
        generate_missing_data()
    if args.plot and not args.tune and not args.train:
        use_saved_model(device, False, args.plot_cases)
        return
    raw = raw_dataset(PATHS.data, "train")
    try:
        pfr, wall, shape = fit_normalizers(raw)
    finally:
        raw.close()
    print(f"Validated coupled grid {shape[0]}x{shape[1]}.")
    config_path = PATHS.output / "best_config.json"
    config, tuning_seconds = None, 0.0
    if args.tune:
        (PATHS.output / "ray_results").mkdir(parents=True, exist_ok=True)
        PATHS.ray.mkdir(parents=True, exist_ok=True)
        tuning = tune_hyperparameters(PATHS.data, PATHS.output, pfr, wall, shape)
        config = dict(tuning.best_config)
        tuning_seconds = tuning.tuning_seconds
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)
    history, training_seconds = None, 0.0
    if args.train:
        selected = config_path if args.train_config == "best" else Path(args.train_config)
        config = config or load_json(selected)
        history, training_seconds = train_best_config(config, pfr, wall, shape, device)
    if args.plot:
        metrics = use_saved_model(device, args.train, args.plot_cases)
        if history is not None:
            model, _, _, _, _ = load_checkpoint(PATHS.output / CHECKPOINT, device)
            metrics.update({
                "parameters": count_parameters(model),
                "training_seconds": training_seconds,
                "tuning_seconds": tuning_seconds,
                "best_epoch": int(torch.tensor(history["valid_relative_l2"]).argmin()) + 1,
                "best_validation_relative_l2": min(history["valid_relative_l2"]),
                "training_shape": list(shape),
            })
            with (PATHS.output / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(metrics, file, indent=2)
            print(json.dumps(metrics, indent=2))
    print(f"Selected stages completed in {PATHS.output}")


if __name__ == "__main__":
    main()
