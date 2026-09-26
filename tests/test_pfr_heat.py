"""PhysicsNeMo checks for the coupled PFR/cylindrical-wall simulator."""

# PhysicsNeMo must see the Warp cache setting before it is imported.
# pylint: disable=wrong-import-position,too-many-locals

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp")

import torch
import pytest
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
HYBRID_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pfr_heat"
    / "deeponet_fno.py"
)
HYBRID_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "pfr_heat_deeponet_fno", HYBRID_SCRIPT_PATH
)
assert HYBRID_SCRIPT_SPEC is not None and HYBRID_SCRIPT_SPEC.loader is not None
HYBRID_SCRIPT = importlib.util.module_from_spec(HYBRID_SCRIPT_SPEC)
HYBRID_SCRIPT_SPEC.loader.exec_module(HYBRID_SCRIPT)
CoupledPFRHeatDataset = FNO_SCRIPT.CoupledPFRHeatDataset
InformerCache = FNO_SCRIPT.InformerCache
PFR_OUTPUTS = FNO_SCRIPT.PFR_OUTPUTS
fit_normalizers = FNO_SCRIPT.fit_normalizers
model_from_config = FNO_SCRIPT.model_from_config
physics_losses = FNO_SCRIPT.physics_losses
raw_dataset = FNO_SCRIPT.raw_dataset


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


def _small_hybrid_config() -> dict[str, float | int | str]:
    return {
        "branch_width": 8,
        "trunk_width": 8,
        "depth": 2,
        "latent_width": 4,
        "activation": "silu",
        "wall_modes_z": 3,
        "wall_modes_r": 2,
        "wall_latent_channels": 4,
        "wall_n_layers": 2,
        "wall_padding": 0,
        "wall_decoder_layers": 1,
        "wall_decoder_layer_size": 8,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "batch_size": 1,
        "lambda_f": 0.0,
        "lambda_g": 0.0,
        "lambda_s": 0.0,
        "lambda_bc": 0.0,
    }


def test_variant_search_spaces_zero_all_data_weights() -> None:
    """Both PFR-heat architectures expose fixed-zero data-only spaces."""
    for module in (FNO_SCRIPT, HYBRID_SCRIPT):
        data_space = module.search_space("data")
        assert all(
            data_space[key] == 0.0 for key in FNO_SCRIPT.PHYSICS_WEIGHT_KEYS
        )
        physics_space = module.search_space("physics")
        assert all(
            physics_space[key] != 0.0 for key in FNO_SCRIPT.PHYSICS_WEIGHT_KEYS
        )


def test_pfr_heat_variant_clis_default_to_physics(monkeypatch) -> None:
    """Both PFR-heat entry points expose the same validated variant CLI."""
    for module in (FNO_SCRIPT, HYBRID_SCRIPT):
        monkeypatch.setattr(module.sys, "argv", [str(SCRIPT_PATH)])
        assert module.parse_cli_args().variant == "physics"
        for variant in ("both", "data", "physics"):
            monkeypatch.setattr(
                module.sys,
                "argv",
                [str(SCRIPT_PATH), "--variant", variant],
            )
            assert module.parse_cli_args().variant == variant
        monkeypatch.setattr(
            module.sys,
            "argv",
            [str(SCRIPT_PATH), "--variant", "invalid"],
        )
        with pytest.raises(SystemExit):
            module.parse_cli_args()


def test_data_trainers_force_zero_weights_and_skip_residuals(monkeypatch) -> None:
    """Explicit configs cannot enable residual work in data-only training."""

    class Normalizer:
        def state_dict(self):
            return {}

    class TrainingModel:
        training = True

        def __call__(self, *inputs):
            del inputs
            return {"pfr": torch.ones(1), "wall": torch.ones(1)}

    context = SimpleNamespace(device=torch.device("cpu"))
    normalizer = Normalizer()
    for module, config in (
        (FNO_SCRIPT, _small_config()),
        (HYBRID_SCRIPT, _small_hybrid_config()),
    ):
        config.update({key: 0.1 for key in FNO_SCRIPT.PHYSICS_WEIGHT_KEYS})
        monkeypatch.setattr(
            module,
            "physics_losses",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("data training computed physics losses")
            ),
        )
        trainer = module.make_trainer(
            config,
            context,
            normalizer,
            normalizer,
            (9, 7),
            variant="data",
        )
        assert all(
            trainer.config[key] == 0.0 for key in FNO_SCRIPT.PHYSICS_WEIGHT_KEYS
        )
        batch = {}
        trainer.forward_adapter(TrainingModel(), (), batch)
        assert all(value == 0.0 for value in batch["physics_losses"].values())


def test_both_variant_uses_explicit_pfr_heat_model_ids(monkeypatch, tmp_path) -> None:
    """Each PFR-heat runner launches independent, explicitly named variants."""

    class Raw:
        def close(self):
            return None

    class Normalizer:
        pass

    expected = {
        FNO_SCRIPT: ("fno_data", "fno_physics"),
        HYBRID_SCRIPT: ("deeponet_fno_data", "deeponet_fno_physics"),
    }
    for module, model_ids in expected.items():
        args = SimpleNamespace(
            variant="both", data_dir=tmp_path, plot_cases=2
        )
        monkeypatch.setattr(module, "parse_cli_args", lambda: args)
        monkeypatch.setattr(module, "raw_dataset", lambda *args: Raw())
        monkeypatch.setattr(
            module,
            "fit_normalizers",
            lambda raw: (Normalizer(), Normalizer(), (9, 7)),
        )
        monkeypatch.setattr(
            module,
            "experiment_spec",
            lambda _args, model_id: SimpleNamespace(model_id=model_id),
        )
        calls = []
        monkeypatch.setattr(
            module,
            "run_operator",
            lambda _args, _paths, spec, _data, _trainer, space, _evaluator: calls.append(
                (spec.model_id, space)
            ),
        )

        module.main()

        assert tuple(model_id for model_id, _ in calls) == model_ids
        assert all(
            calls[0][1][key] == 0.0 for key in FNO_SCRIPT.PHYSICS_WEIGHT_KEYS
        )


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
        assert sample["pfr_branch"].shape == (9,)
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


def test_coupled_deeponet_fno_shapes_physics_and_wall_gradient(tmp_path) -> None:
    """The wall FNO consumes reactor DeepONet predictions end to end."""
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
        model = HYBRID_SCRIPT.model_from_config(
            _small_hybrid_config(),
            pfr_normalizer,
            wall_normalizer,
            torch.device("cpu"),
        )
        prediction = model(
            sample["pfr_branch"].unsqueeze(0),
            sample["z"].unsqueeze(0),
            sample["wall_conditions"].unsqueeze(0),
        )
        assert prediction["pfr"].shape == (1, 4, 9)
        assert prediction["wall"].shape == (1, 1, 9, 7)
        assert model.pfr_bias.shape == (4,)

        batch = {name: value.unsqueeze(0) for name, value in sample.items()}
        losses = physics_losses(
            prediction,
            batch,
            pfr_normalizer,
            wall_normalizer,
            InformerCache(torch.device("cpu")),
        )
        assert set(losses) == {"species", "gas", "solid", "bc"}
        assert all(torch.isfinite(value) for value in losses.values())

        prediction["wall"].square().mean().backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.branch.parameters()
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
        from chem_operator.experiments import RunContext
        context = RunContext.create(tmp_path,problem="pfr_heat",model="fno",run_id="test",seed=42)
        config.update(epochs=1,batch_size=1,learning_rate=1e-3,weight_decay=0,
                      lambda_f=1e-4,lambda_g=1e-4,lambda_s=1e-4,lambda_bc=1e-4)
        trainer = FNO_SCRIPT.make_trainer(config,context,pfr_normalizer,wall_normalizer,shape)
        training = trainer.fit(dataset,dataset,context)
        model = trainer.model
        restored_trainer = FNO_SCRIPT.make_trainer(config,context,pfr_normalizer,wall_normalizer,shape)
        restored_trainer.load_checkpoint(training.checkpoint,context)
        restored = restored_trainer.model
        restored_shape = tuple(restored_trainer.checkpoint_metadata["shape"])
        outcome = FNO_SCRIPT.evaluate_run(trainer,dataset,context,pfr_normalizer,wall_normalizer,cases=1)
        assert outcome.reconstructions["wall_prediction"].shape == (1,1,9,7)
        assert "solid_loss" in outcome.metrics
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


def test_coupled_deeponet_fno_checkpoint_round_trip(tmp_path) -> None:
    """The hybrid checkpoint restores both components and evaluation metadata."""
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
        config = _small_hybrid_config()
        config["epochs"] = 1
        from chem_operator.experiments import RunContext

        context = RunContext.create(
            tmp_path,
            problem="pfr_heat",
            model="deeponet_fno",
            run_id="test",
            seed=42,
        )
        trainer = HYBRID_SCRIPT.make_trainer(
            config, context, pfr_normalizer, wall_normalizer, shape
        )
        training = trainer.fit(dataset, dataset, context)
        restored_trainer = HYBRID_SCRIPT.make_trainer(
            config, context, pfr_normalizer, wall_normalizer, shape
        )
        restored_trainer.load_checkpoint(training.checkpoint, context)
        assert restored_trainer.checkpoint_metadata["components"] == {
            "reactor": "PhysicsNeMo FullyConnected DeepONet",
            "wall": "PhysicsNeMo FNO",
        }
        outcome = HYBRID_SCRIPT.evaluate_run(
            trainer,
            dataset,
            context,
            pfr_normalizer,
            wall_normalizer,
            cases=1,
        )
        assert outcome.reconstructions["pfr_prediction"].shape == (1, 4, 9)
        assert outcome.reconstructions["wall_prediction"].shape == (1, 1, 9, 7)
        assert {
            "data_loss",
            "species_loss",
            "gas_loss",
            "solid_loss",
            "bc_loss",
        }.issubset(outcome.metrics)
        with torch.no_grad():
            inputs = (
                sample["pfr_branch"].unsqueeze(0),
                sample["z"].unsqueeze(0),
                sample["wall_conditions"].unsqueeze(0),
            )
            expected = trainer.model(*inputs)
            actual = restored_trainer.model(*inputs)
        assert torch.equal(actual["pfr"], expected["pfr"])
        assert torch.equal(actual["wall"], expected["wall"])
    finally:
        raw.close()
