"""Tests for the metadata-driven PhysicsNeMo transient pipe-flow FNO."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.models import fit_fno_zscore_normalizer
from chem_operator.reactors.pipe_flow_transient.dataset_generator import (
    TransientHagenPoiseuillePipeFlowSim,
)
from chem_operator.sampling import Constant, Uniform


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pipe_flow_transient"
    / "transient_nemo_fno.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "pipe_flow_transient_nemo_fno",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
NEMO_FNO = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(NEMO_FNO)


@pytest.fixture(scope="module", name="nemo_data_dir")
def fixture_nemo_data_dir(tmp_path_factory) -> object:
    """Generate small varied splits carrying PhysicsNeMo PDE metadata."""
    output = tmp_path_factory.mktemp("pipe_flow_nemo_fno")
    simulator = TransientHagenPoiseuillePipeFlowSim(
        parameter_space={
            "radius": Uniform(0.8e-3, 1.2e-3),
            "length": Uniform(0.8, 1.2),
            "dynamic_viscosity": Uniform(0.9e-3, 1.1e-3),
            "pressure_drop": Uniform(40.0, 80.0),
            "density": Constant(1000.0),
            "n_time_points": Constant(8),
            "n_radial_points": Constant(8),
            "max_fourier_number": Constant(2.0),
        }
    )
    generator = SimulationDatasetGenerator(simulator, output, seed=19)
    generator.save_splits(generator.generate_splits(n_cases=10))
    return output


@pytest.fixture(name="nemo_normalizer")
def fixture_nemo_normalizer(nemo_data_dir):
    """Fit channel statistics on the generated training split."""
    dataset = NEMO_FNO.raw_dataset(nemo_data_dir, "train")
    try:
        return fit_fno_zscore_normalizer(
            dataset,
            NEMO_FNO.INPUT_CHANNELS,
            NEMO_FNO.OUTPUT_CHANNELS,
        )
    finally:
        dataset.close()


def small_config() -> dict[str, float | int]:
    """Return a minimal CPU training configuration."""
    return {
        "modes": 2,
        "latent_channels": 4,
        "n_layers": 2,
        "padding": 0,
        "decoder_layers": 1,
        "decoder_layer_size": 8,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-8,
        "batch_size": 2,
        "physics_weight": 1.0e-4,
        "constraint_weight": 1.0e-4,
    }


def test_dataset_pde_resolution_uses_metadata() -> None:
    """PDE selection accepts one registered name and rejects bad provenance."""

    class MetadataDataset:
        """Minimal metadata-only dataset used for PDE resolution tests."""

        def __init__(self, names):
            self.names = names

        def __len__(self):
            return len(self.names)

        def __getitem__(self, index):
            name = self.names[index]
            metadata = {} if name is None else {"physicsnemo_pde": name}
            return {"metadata": metadata}

    valid = MetadataDataset(["TransientHagenPoiseuille"] * 2)
    assert NEMO_FNO.dataset_pde_name(valid) == "TransientHagenPoiseuille"
    with pytest.raises(KeyError, match="physicsnemo_pde"):
        NEMO_FNO.dataset_pde_name(MetadataDataset([None]))
    with pytest.raises(ValueError, match="exactly one"):
        NEMO_FNO.dataset_pde_name(
            MetadataDataset(["TransientHagenPoiseuille", "AnotherPDE"])
        )
    with pytest.raises(ValueError, match="Unsupported"):
        NEMO_FNO.dataset_pde_name(MetadataDataset(["AnotherPDE"]))


def test_adapter_and_exact_solution_physics_loss(
    nemo_data_dir,
    nemo_normalizer,
) -> None:
    """Generated truth has correct tensors and a small differentiable residual."""
    raw, adapter = NEMO_FNO.make_adapter(
        nemo_data_dir, "train", nemo_normalizer, maximum=2
    )
    try:
        sample = adapter[0]
        assert sample["x"].shape == (4, 8, 8)
        assert sample["y"].shape == (1, 8, 8)
        assert sample["physics_constants"].shape == (5,)
        assert sample["physicsnemo_pde"] == "TransientHagenPoiseuille"

        spacing = NEMO_FNO.adapter_radial_spacing(adapter)
        informer = NEMO_FNO.make_physics_informer(
            sample["physicsnemo_pde"],
            radial_spacing=spacing,
            device=torch.device("cpu"),
        )
        batch = next(iter(DataLoader(adapter, batch_size=2)))
        prediction = batch["y"].detach().clone().requires_grad_(True)
        physics_loss, constraint_loss = NEMO_FNO.physics_losses(
            prediction,
            batch,
            nemo_normalizer,
            informer,
            pde_name=sample["physicsnemo_pde"],
            radial_spacing=spacing,
        )
        assert math.isfinite(float(physics_loss.detach()))
        assert float(physics_loss.detach()) < 0.1
        assert float(constraint_loss.detach()) < 1.0e-6
        (physics_loss + constraint_loss).backward()
        assert prediction.grad is not None
        assert torch.isfinite(prediction.grad).all()
    finally:
        raw.close()


def test_training_evaluation_and_checkpoint_smoke(  # pylint: disable=too-many-locals
    nemo_data_dir,
    nemo_normalizer,
    tmp_path,
) -> None:
    """One CPU epoch trains, evaluates, and round-trips a PhysicsNeMo FNO."""
    train_raw, train_data = NEMO_FNO.make_adapter(
        nemo_data_dir, "train", nemo_normalizer, maximum=4
    )
    valid_raw, valid_data = NEMO_FNO.make_adapter(
        nemo_data_dir, "valid", nemo_normalizer, maximum=1
    )
    config = small_config()
    pde_name = "TransientHagenPoiseuille"
    try:
        model, history, _ = NEMO_FNO.train_model(
            config,
            train_data,
            valid_data,
            nemo_normalizer,
            pde_name=pde_name,
            epochs=1,
            device=torch.device("cpu"),
        )
        assert set(history) == set(NEMO_FNO.HISTORY_FIELDS)
        for values in history.values():
            assert len(values) == 1
            assert math.isfinite(values[0])

        spacing = NEMO_FNO.adapter_radial_spacing(train_data)
        metrics = NEMO_FNO.evaluate(
            model,
            valid_data,
            nemo_normalizer,
            pde_name=pde_name,
            radial_spacing=spacing,
            batch_size=1,
            device=torch.device("cpu"),
        )
        assert all(math.isfinite(value) for value in metrics.values())

        checkpoint = tmp_path / "physicsnemo_fno.pt"
        NEMO_FNO.save_checkpoint(
            checkpoint,
            model,
            config,
            nemo_normalizer,
            pde_name,
            spacing,
        )
        loaded, loaded_normalizer, loaded_pde, loaded_spacing = (
            NEMO_FNO.load_checkpoint(checkpoint, torch.device("cpu"))
        )
        assert loaded_pde == pde_name
        assert loaded_spacing == pytest.approx(spacing)
        sample = valid_data[0]
        loaded_data = NEMO_FNO.PhysicsFNOAdapter(
            valid_raw,
            loaded_normalizer,
            input_channels=NEMO_FNO.INPUT_CHANNELS,
            output_channels=NEMO_FNO.OUTPUT_CHANNELS,
            coordinate_names=("t", "r"),
            max_trajectories=1,
        )
        loaded_sample = loaded_data[0]
        with torch.no_grad():
            expected = model(sample["x"].unsqueeze(0))
            actual = loaded(loaded_sample["x"].unsqueeze(0))
        torch.testing.assert_close(actual, expected)
    finally:
        train_raw.close()
        valid_raw.close()


def test_normalizer_can_fit_a_subset(nemo_data_dir) -> None:
    """The normalizer path accepts bounded lazy training subsets."""
    raw = NEMO_FNO.raw_dataset(nemo_data_dir, "train")
    try:
        normalizer = fit_fno_zscore_normalizer(
            Subset(raw, range(2)),
            NEMO_FNO.INPUT_CHANNELS,
            NEMO_FNO.OUTPUT_CHANNELS,
        )
        assert set(normalizer.means) == {
            *NEMO_FNO.MODEL_CONSTANT_NAMES,
            *NEMO_FNO.FIELD_NAMES,
        }
    finally:
        raw.close()
