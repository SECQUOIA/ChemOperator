"""Shared problem specification and data preparation."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import Any
import numpy as np
import torch
from neuralop.losses import LpLoss
from torch.utils.data import DataLoader
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import FNOAdapter, FNOChannel, fit_fno_zscore_normalizer
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.experiments import (
    FNOTrainer,
    LossTerm,
    evaluate_fields,
    operator_spec,
    parse_operator_args,
    run_operator,
)
PATHS = ExamplePaths.from_script(__file__, dataset="q2d_cmr")

FILE_STEM = "q2d_cmr"
SEED = 42
METRIC = "best_valid_loss"

TUNE_SAMPLES = 20
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

PROBLEM_ID = "q2d_cmr"

def experiment_spec(args, model_id):
    labels = tuple(channel.label for channel in OUTPUT_CHANNELS)
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FILE_STEM,
                         labels, {channel.label: channel.unit for channel in OUTPUT_CHANNELS}, ("z", "r"))

def parse_args(*, model_name: str = "FNO", benchmark: bool = True):
    parser = argparse.ArgumentParser(description=f"Tune and train a Q2D {model_name} with canonical artifacts.")
    if benchmark:
        parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True,
                            help="Evaluate available mesh-sweep files and save cost and superresolution artifacts.")
    return parse_operator_args(PATHS, epochs=FINAL_EPOCHS, tune_epochs=TUNE_EPOCHS, samples=TUNE_SAMPLES, parser=parser)

ModelFactory = Callable[[Mapping[str, Any], torch.device], torch.nn.Module]


def relative_l2_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=1)
    reference = target.flatten(start_dim=1)
    return torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(RELATIVE_L2_EPS)


def relative_l2_per_field(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=2)
    reference = target.flatten(start_dim=2)
    return torch.linalg.vector_norm(difference, dim=2) / torch.linalg.vector_norm(
        reference, dim=2
    ).clamp_min(RELATIVE_L2_EPS)


def evaluate(
    model: torch.nn.Module,
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
            global_values.append(relative_l2_per_sample(prediction, target).cpu())
            field_values.append(relative_l2_per_field(prediction, target).cpu())
    global_tensor = torch.cat(global_values)
    fields_tensor = torch.cat(field_values)
    metrics = {"normalized_relative_l2": float(global_tensor.mean())}
    for index, spec in enumerate(OUTPUT_CHANNELS):
        metrics[f"normalized_relative_l2_{spec.label}"] = float(
            fields_tensor[:, index].mean()
        )
    return metrics


def make_trainer(
    model_factory: ModelFactory,
    config: Mapping[str, Any],
    context: Any,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    shape: tuple[int, int],
) -> FNOTrainer:
    """Use the same loss, physical metrics, and checkpoint contract for all models."""
    loss = LpLoss(d=2, p=2, reduction="mean")

    def physical(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                normalizer.denormalize(value[:, index], channel.label)
                for index, channel in enumerate(OUTPUT_CHANNELS)
            ],
            dim=1,
        )

    return FNOTrainer(
        lambda cfg: model_factory(cfg, context.device),
        config,
        loss_terms=[LossTerm("data_loss", lambda prediction, target, batch: loss(prediction, target))],
        selection_metric="data_loss",
        metric_adapter=lambda prediction, target, batch: (
            physical(prediction),
            physical(target),
        ),
        checkpoint_metadata={
            "normalizer": normalizer.state_dict(),
            "geometry": list(geometry),
            "training_shape": list(shape),
            "input_channels": [channel.label for channel in INPUT_CHANNELS],
            "output_channels": [channel.label for channel in OUTPUT_CHANNELS],
        },
    )


def run_model(args: Any, model_id: str, model_factory: ModelFactory, search_space: Mapping[str, Any]) -> None:
    """Run Q2D tuning, final training, and test evaluation."""
    normalizer, geometry, shape = fit_normalizer(args.data_dir / f"{FILE_STEM}_train.h5")

    def evaluate_run(trainer: FNOTrainer, data: FNOAdapter, context: Any):
        return evaluate_fields(
            trainer,
            data,
            context,
            labels=tuple(channel.label for channel in OUTPUT_CHANNELS),
            coordinate_names=("z", "r"),
            cases=args.plot_cases,
            extra_metrics=lambda trained, dataset, run_context: evaluate(
                trained.model,
                dataset,
                batch_size=EVALUATION_BATCH_SIZE,
                device=run_context.device,
            ),
        )

    run_operator(
        args,
        PATHS,
        experiment_spec(args, model_id),
        lambda split: make_adapter(
            args.data_dir / f"{FILE_STEM}_{split}.h5", normalizer, geometry
        ),
        lambda config, context: make_trainer(
            model_factory, config, context, normalizer, geometry, shape
        ),
        search_space,
        evaluate_run,
        selection_metric="data_loss",
    )


__all__ = ['PATHS', 'FILE_STEM', 'SEED', 'METRIC', 'TUNE_SAMPLES', 'TUNE_EPOCHS', 'FINAL_EPOCHS', 'EVALUATION_BATCH_SIZE', 'CPUS_PER_TRIAL', 'GPUS_PER_TRIAL', 'MAX_CONCURRENT_TRIALS', 'LATENCY_WARMUPS', 'LATENCY_REPEATS', 'RELATIVE_L2_EPS', 'AMORTIZED_EVALUATION_COUNTS', 'REPRESENTATIVE_RESOLUTION_COUNT', 'MESH_FILE_PATTERN', 'INPUT_CHANNELS', 'OUTPUT_CHANNELS', 'MeshFile', 'raw_dataset', 'adapter_spatial_shape', 'fit_normalizer', 'normalizer_state', 'normalizer_from_state', 'make_adapter', 'discover_mesh_files', 'find_superresolution_pair', 'PROBLEM_ID', 'experiment_spec', 'parse_args', 'ModelFactory', 'relative_l2_per_sample', 'relative_l2_per_field', 'evaluate', 'make_trainer', 'run_model']
