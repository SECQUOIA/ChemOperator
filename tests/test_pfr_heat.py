"""PhysicsNeMo checks for the coupled PFR/cylindrical-wall simulator."""

# PhysicsNeMo must see the Warp cache setting before it is imported.
# pylint: disable=wrong-import-position,too-many-locals

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp")

import torch
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.reactors.pfr_heat.dataset_generator import (
    CylindricalWall,
    PFRHeatSim,
    PlugFlowReactor,
)
SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "pfr_heat" / "fno.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location("pfr_heat_fno", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
FNO_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(FNO_SCRIPT)
CoupledPFRHeatDataset = FNO_SCRIPT.CoupledPFRHeatDataset
InformerCache = FNO_SCRIPT.InformerCache
PFR_OUTPUTS = FNO_SCRIPT.PFR_OUTPUTS
fit_normalizers = FNO_SCRIPT.fit_normalizers
load_checkpoint = FNO_SCRIPT.load_checkpoint
model_from_config = FNO_SCRIPT.model_from_config
physics_losses = FNO_SCRIPT.physics_losses
raw_dataset = FNO_SCRIPT.raw_dataset
save_checkpoint = FNO_SCRIPT.save_checkpoint


def _small_case(simulator: PFRHeatSim):
    """Return a cheap coupled solution suitable for FNO adapter tests."""
    params = {
        "inlet_flow_a": 80.0,
        "inlet_concentration_a": 0.075,
        "inlet_temperature": 390.0,
        "outer_temperature": 320.0,
        "volumetric_heat_transfer": 4000.0,
        "wall_aspect_ratio_sq": 0.2,
        "interface_biot": 1.5,
        "inner_radius": 1.0,
        "outer_radius": 2.0,
        "n_axial_points": 9,
        "n_radial_points": 7,
        "tolerance": 1.0e-4,
        "max_nodes": 5000,
    }
    record = simulator.run_case(simulator.make_case(params))
    record.metadata["params"] = params
    return record


def _small_config() -> dict[str, float | int]:
    return {
        "pfr_modes": 3,
        "wall_modes_z": 3,
        "wall_modes_r": 2,
        "latent_channels": 4,
        "n_layers": 2,
        "padding": 0,
        "decoder_layers": 1,
        "decoder_layer_size": 8,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "batch_size": 1,
        "lambda_f": 0.0,
        "lambda_g": 0.0,
        "lambda_s": 0.0,
        "lambda_bc": 0.0,
    }


def test_simulated_solution_has_near_zero_physics_loss() -> None:
    """The BVP solution satisfies both PhysicsNeMo PDEs and their BCs."""
    params = {
        "inlet_flow_a": 100.0,
        "inlet_concentration_a": 0.1,
        "inlet_temperature": 393.15,
        "outer_temperature": 323.15,
        "volumetric_heat_transfer": 4000.0,
        "wall_aspect_ratio_sq": 0.2,
        "interface_biot": 1.5,
        "inner_radius": 1.0,
        "outer_radius": 2.0,
        "n_axial_points": 401,
        "n_radial_points": 21,
        "tolerance": 1.0e-6,
        "max_nodes": 10_000,
    }
    simulator = PFRHeatSim()
    record = simulator.run_case(simulator.make_case(params))
    constants = simulator.constants
    z, r = record.coordinates["z"], record.coordinates["r"]
    dz, dr = float(z[1] - z[0]), float(r[1] - r[0])

    def field(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64)[None, None]

    flows = record.fields["F"] / constants.flow_scale
    gas_temperature = field(
        record.fields["T_gas"] / constants.temperature_scale
    )
    solid_temperature = field(
        record.fields["T_solid"] / constants.temperature_scale
    )
    gas_residuals = PhysicsInformer(
        ["flow_a", "flow_b", "flow_c", "gas_energy"],
        PlugFlowReactor(
            constants,
            params["inlet_concentration_a"],
            params["inlet_temperature"],
            params["volumetric_heat_transfer"],
        ),
        "finite_difference",
        fd_dx=dz,
        device="cpu",
    ).forward(
        {
            "f_a": field(flows[:, 0]),
            "f_b": field(flows[:, 1]),
            "f_c": field(flows[:, 2]),
            "t_gas": gas_temperature,
            "t_wall": solid_temperature[..., 0],
        }
    )
    radial_coordinate = field(r)[..., None, :].expand_as(solid_temperature)
    solid_residual = PhysicsInformer(
        ["solid_heat"],
        CylindricalWall(
            params["wall_aspect_ratio_sq"], params["interface_biot"]
        ),
        "finite_difference",
        fd_dx=[dz, dr],
        device="cpu",
    ).forward({"t_solid": solid_temperature, "y": radial_coordinate})[
        "solid_heat"
    ]

    interior_loss = sum(
        residual[..., 2:-2].square().mean()
        for residual in gas_residuals.values()
    ) + solid_residual[..., 2:-2, 2:-2].square().mean()
    interface_gradient = (
        -3.0 * solid_temperature[..., 0]
        + 4.0 * solid_temperature[..., 1]
        - solid_temperature[..., 2]
    ) / (2.0 * dr)
    axial_gradient = torch.gradient(
        solid_temperature[0, 0], spacing=(dz, dr), edge_order=2
    )[0]
    boundary_loss = (
        (
            field(flows[0])
            - field(
                [params["inlet_flow_a"] / constants.flow_scale, 0.0, 0.0]
            )
        )
        .square()
        .mean()
        + (
            gas_temperature[..., 0]
            - params["inlet_temperature"] / constants.temperature_scale
        ).square().mean()
        + (
            solid_temperature[..., -1]
            - params["outer_temperature"] / constants.temperature_scale
        ).square().mean()
        + axial_gradient[[0, -1]].square().mean()
        + (
            interface_gradient
            - params["interface_biot"]
            * (solid_temperature[..., 0] - gas_temperature)
        ).square().mean()
    )
    physics_loss = interior_loss + boundary_loss

    assert record.metadata["physicsnemo_pdes"] == {
        "reactor": "PlugFlowReactor",
        "solid": "CylindricalWall",
    }
    assert physics_loss.item() < 1.0e-3


def test_coupled_fno_adapter_shapes_and_wall_gradient(tmp_path) -> None:
    """The wall consumes predicted gas temperature without target leakage."""
    simulator = PFRHeatSim()
    generator = SimulationDatasetGenerator(simulator, tmp_path)
    generator.save_split("train", [_small_case(simulator)])
    raw = raw_dataset(tmp_path, "train")
    try:
        pfr_normalizer, wall_normalizer, shape = fit_normalizers(raw)
        dataset = CoupledPFRHeatDataset(
            raw, pfr_normalizer, wall_normalizer, shape
        )
        sample = dataset[0]
        assert sample["pfr_x"].shape == (9, 9)
        assert sample["pfr_y"].shape == (4, 9)
        assert sample["wall_conditions"].shape == (5, 9, 7)
        assert sample["wall_y"].shape == (1, 9, 7)
        assert PFR_OUTPUTS == ("F_A", "F_B", "F_C", "T_gas")
        batch = {
            name: value.unsqueeze(0) for name, value in sample.items()
        }
        exact_losses = physics_losses(
            {"pfr": batch["pfr_y"], "wall": batch["wall_y"]},
            batch,
            pfr_normalizer,
            wall_normalizer,
            InformerCache(torch.device("cpu")),
        )
        # The deliberately coarse 9x7 smoke grid has appreciable wall-FD error.
        assert max(float(value) for value in exact_losses.values()) < 1.0e-1

        model = model_from_config(
            _small_config(), pfr_normalizer, wall_normalizer, torch.device("cpu")
        )
        prediction = model(
            sample["pfr_x"].unsqueeze(0),
            sample["wall_conditions"].unsqueeze(0),
        )
        assert prediction["pfr"].shape == (1, 4, 9)
        assert prediction["wall"].shape == (1, 1, 9, 7)
        prediction["wall"].square().mean().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.pfr.parameters()
            if parameter.grad is not None
        )
        assert gradient > 0.0
    finally:
        raw.close()


def test_coupled_fno_checkpoint_round_trip(tmp_path) -> None:
    """One checkpoint restores both branches and their normalization state."""
    simulator = PFRHeatSim()
    data_path = tmp_path / "data"
    generator = SimulationDatasetGenerator(simulator, data_path)
    generator.save_split("train", [_small_case(simulator)])
    raw = raw_dataset(data_path, "train")
    try:
        pfr_normalizer, wall_normalizer, shape = fit_normalizers(raw)
        dataset = CoupledPFRHeatDataset(
            raw, pfr_normalizer, wall_normalizer, shape
        )
        sample = dataset[0]
        config = _small_config()
        model = model_from_config(
            config, pfr_normalizer, wall_normalizer, torch.device("cpu")
        ).eval()
        checkpoint = tmp_path / "coupled.pt"
        save_checkpoint(
            checkpoint,
            model,
            config,
            pfr_normalizer,
            wall_normalizer,
            shape,
        )
        restored, _, _, restored_shape, _ = load_checkpoint(
            checkpoint, torch.device("cpu")
        )
        with torch.no_grad():
            expected = model(
                sample["pfr_x"].unsqueeze(0),
                sample["wall_conditions"].unsqueeze(0),
            )
            actual = restored(
                sample["pfr_x"].unsqueeze(0),
                sample["wall_conditions"].unsqueeze(0),
            )
        assert restored_shape == shape
        assert torch.equal(actual["pfr"], expected["pfr"])
        assert torch.equal(actual["wall"], expected["wall"])
    finally:
        raw.close()
