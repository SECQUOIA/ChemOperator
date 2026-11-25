"""In-memory pipe-flow tensors shared by the physics-informed DeepONet and FNO scripts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import Dataset, TensorDataset

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.experiments import operator_spec
from experiments.pipe_flow.common import CONSTANTS, FIELDS, PATHS, PROBLEM_ID, UNITS


BRANCH_NAMES = CONSTANTS
ALL_CONSTANTS = BRANCH_NAMES + ("density", "pressure_gradient")


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


def load_physics_split(data_dir: Path, split: str, maximum: int | None) -> PipeData:
    torch.set_default_device("cpu")
    dataset = ChemOperatorDataset(
        data_dir / f"{PROBLEM_ID}_{split}.h5",
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


def fit_physics_normalization(data: PipeData) -> Normalization:
    return Normalization(
        branch_mean=data.branch.mean(dim=0),
        branch_std=data.branch.std(dim=0, unbiased=False).clamp_min(1.0e-8),
        velocity_mean=data.target.mean(),
        velocity_std=data.target.std(unbiased=False).clamp_min(1.0e-8),
    )

class PhysicsPipeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tensors = data.dataset()

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        values = self.tensors[index]
        return dict(zip(("branch", "coordinates", "radius", "viscosity", "pressure_gradient", "y", "reference"), values))

    def close(self):
        """This dataset owns in-memory tensors; its HDF5 reader is already closed."""


def physics_experiment_spec(args, model_id):
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FIELDS, UNITS, ("r",))
