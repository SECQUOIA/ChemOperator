"""Shared problem specification and data preparation."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any, Mapping
from physicsnemo.sym.eq.pde import PDE
import torch
from torch.utils.data import Subset
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import FNOAdapter, FNOChannel, fit_fno_zscore_normalizer
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.reactors.pipe_flow_transient.dataset_generator import TransientHagenPoiseuille
from chem_operator.experiments import parse_operator_args, operator_spec
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


MAX_TRAIN_TRAJECTORIES = None
CONSTANT_NAMES = MODEL_CONSTANT_NAMES
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


def training_pde_name(data_dir: Path) -> str:
    """Resolve the governing PDE from the generated training data."""
    dataset = raw_dataset(data_dir, "train")
    try:
        return dataset_pde_name(dataset, 1)
    finally:
        dataset.close()

PROBLEM_ID = "pipe_flow_transient"

def experiment_spec(args, model_id):
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FILE_STEM,
                         FIELD_NAMES, {"velocity": "m/s"}, ("t", "r"))

def parse_args(*, epochs, tune_epochs, samples, parser=None):
    return parse_operator_args(
        PATHS,
        epochs=epochs,
        tune_epochs=tune_epochs,
        samples=samples,
        parser=parser,
    )

__all__ = ['PATHS', 'FIELD_NAMES', 'MODEL_CONSTANT_NAMES', 'PHYSICS_CONSTANT_NAMES', 'INPUT_CHANNELS', 'OUTPUT_CHANNELS', 'FILE_STEM', 'METRIC', 'SEED', 'PDE_REGISTRY', 'MAX_TRAIN_TRAJECTORIES', 'CONSTANT_NAMES', 'PhysicsFNOAdapter', 'raw_dataset', 'dataset_pde_name', 'make_adapter', 'fit_normalizer', 'normalizer_state', 'normalizer_from_state', 'adapter_radial_spacing', 'training_pde_name', 'PROBLEM_ID', 'experiment_spec', 'parse_args']
