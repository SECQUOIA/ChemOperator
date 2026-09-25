"""Shared problem specification and data preparation."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections.abc import Mapping, Sequence
from typing import Any
import torch
from torch.utils.data import Dataset
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.experiments import parse_operator_args, operator_spec
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

TUNE_SAMPLES = 8
TUNE_EPOCHS = 40
FINAL_EPOCHS = 50
EVALUATION_BATCH_SIZE = 4
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = int(torch.cuda.is_available())
MAX_CONCURRENT_TRIALS = 1
HISTORY_KEYS = (
    "data_loss", "species_loss", "gas_loss", "solid_loss", "bc_loss",
    "total_loss", "valid_relative_l2", "valid_species_loss",
    "valid_gas_loss", "valid_solid_loss", "valid_bc_loss",
)

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
        pfr_branch = torch.stack([
            self.pfr_normalizer.normalize(constants[name], name).reshape(())
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
            "pfr_x": pfr_x, "pfr_branch": pfr_branch, "pfr_y": pfr_y,
            "wall_conditions": wall_conditions,
            "wall_y": self.wall_normalizer.normalize(solid, "T_solid").unsqueeze(0),
            "z": coordinate(sample, "z"), "r": coordinate(sample, "r"),
            "physics_constants": torch.stack([
                constants[name].reshape(()) for name in PHYSICS_CONSTANTS
            ]),
        }


def make_adapter(data_dir, split, pfr_normalizer, wall_normalizer, shape):
    raw = raw_dataset(Path(data_dir), split)
    return raw, CoupledPFRHeatDataset(raw, pfr_normalizer, wall_normalizer, shape)

PROBLEM_ID = "pfr_heat"

def experiment_spec(args, model_id):
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FILE_STEM,
        PFR_OUTPUTS + WALL_OUTPUTS, {"F_A": "mol/s", "F_B": "mol/s", "F_C": "mol/s", "T_gas": "K", "T_solid": "K"},
        {"pfr": ["z"], "wall": ["z", "r"]}, channels={"pfr": list(PFR_OUTPUTS), "wall": list(WALL_OUTPUTS)})

def parse_args(parser=None):
    return parse_operator_args(
        PATHS,
        epochs=FINAL_EPOCHS,
        tune_epochs=TUNE_EPOCHS,
        samples=TUNE_SAMPLES,
        parser=parser,
    )

__all__ = ['PATHS', 'FILE_STEM', 'CHECKPOINT', 'SEED', 'METRIC', 'PFR_INPUTS', 'PFR_OUTPUTS', 'WALL_CONDITIONS', 'WALL_GAS', 'WALL_OUTPUTS', 'PHYSICS_CONSTANTS', 'EXPECTED_PDES', 'TUNE_SAMPLES', 'TUNE_EPOCHS', 'FINAL_EPOCHS', 'EVALUATION_BATCH_SIZE', 'CPUS_PER_TRIAL', 'GPUS_PER_TRIAL', 'MAX_CONCURRENT_TRIALS', 'HISTORY_KEYS', 'raw_dataset', 'complete_field', 'coordinate', 'uniform_spacing', 'validate_sample', 'Moments', 'make_normalizer', 'fit_normalizers', 'CoupledPFRHeatDataset', 'make_adapter', 'PROBLEM_ID', 'experiment_spec', 'parse_args']
