"""Tests for the analytical circular-pipe dataset generator."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from chem_operator.datasets import ChemOperatorDataset, SimulationDatasetGenerator
from chem_operator.reactors.pipe_flow.dataset_generator import (
    HagenPoiseuille,
    HagenPoiseuillePipeFlowSim,
    hagen_poiseuille_velocity,
)
from chem_operator.sampling import Constant


def _params(**overrides: float | int) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "radius": 1.0e-3,
        "length": 1.0,
        "dynamic_viscosity": 1.0e-3,
        "density": 1000.0,
        "pressure_drop": 80.0,
        "n_radial_points": 128,
    }
    params.update(overrides)
    return params


def test_analytical_profile_and_flow_rate() -> None:
    """The exact profile satisfies its boundary and integral properties."""
    params = _params(n_radial_points=10_001)
    radius = float(params["radius"])
    length = float(params["length"])
    viscosity = float(params["dynamic_viscosity"])
    pressure_drop = float(params["pressure_drop"])
    pressure_gradient = -pressure_drop / length

    radial_coordinate = torch.linspace(
        0.0, radius, int(params["n_radial_points"]), dtype=torch.float64
    )
    velocity = hagen_poiseuille_velocity(
        radial_coordinate,
        radius=radius,
        dynamic_viscosity=viscosity,
        pressure_gradient=pressure_gradient,
    )

    expected_max_velocity = pressure_drop * radius**2 / (4.0 * viscosity * length)
    expected_flow_rate = (
        np.pi * radius**4 * pressure_drop / (8.0 * viscosity * length)
    )
    numerical_flow_rate = torch.trapezoid(
        2.0 * torch.pi * radial_coordinate * velocity,
        radial_coordinate,
    )

    assert velocity[0].item() == pytest.approx(expected_max_velocity)
    assert velocity[-1].item() == pytest.approx(0.0, abs=1.0e-15)
    assert torch.all(torch.diff(velocity) <= 0.0)
    assert numerical_flow_rate.item() == pytest.approx(
        expected_flow_rate, rel=1.0e-7
    )


def test_simulator_record_and_laminar_validation() -> None:
    """The simulator emits the schema and rejects non-laminar cases."""
    simulator = HagenPoiseuillePipeFlowSim()
    case = simulator.make_case(_params())
    record = simulator.run_case(case)

    assert record.coordinates["r"].shape == (128,)
    assert record.fields["velocity"].shape == (128,)
    assert record.coordinates["r"][0] == pytest.approx(0.0)
    assert record.coordinates["r"][-1] == pytest.approx(1.0e-3)
    assert record.constants["pressure_gradient"] == pytest.approx(-80.0)
    assert record.metadata["reynolds_number"] < 2300.0
    assert record.metadata["units"]["velocity"] == "m/s"

    turbulent_case = simulator.make_case(
        _params(radius=1.0e-2, length=0.1, pressure_drop=1000.0)
    )
    with pytest.raises(ValueError, match="must remain laminar"):
        simulator.run_case(turbulent_case)


def test_hdf5_round_trip(tmp_path) -> None:
    """Generated records remain usable through ChemOperator's HDF5 reader."""
    values = _params(n_radial_points=16)
    simulator = HagenPoiseuillePipeFlowSim(
        parameter_space={name: Constant(value) for name, value in values.items()}
    )
    generator = SimulationDatasetGenerator(simulator, tmp_path, seed=7)
    splits = generator.generate_splits(n_cases=3)
    generator.save_splits(splits)

    dataset = ChemOperatorDataset(
        tmp_path / "hagen_poiseuille_pipe_flow_train.h5",
        task="field_map",
        coordinate_name="r",
        input_fields=("velocity",),
        output_fields=("velocity",),
        constant_inputs=(
            "radius",
            "length",
            "dynamic_viscosity",
            "density",
            "pressure_drop",
            "pressure_gradient",
        ),
        dtype=torch.float64,
    )
    try:
        sample = dataset[0]
        assert sample["input_fields"]["velocity"].shape == (1,)
        assert sample["output_fields"]["velocity"].shape == (15,)
        assert sample["output_coordinates"]["r"].shape == (15,)
        assert sample["constant_inputs"]["radius"].ndim == 0
        assert sample["metadata"]["equation"] == "Hagen-Poiseuille"
    finally:
        dataset.close()


def test_physics_informer_residual_for_exact_profile() -> None:
    """PhysicsInformer evaluates the exact cylindrical residual as zero."""
    radius = 1.0e-3
    dynamic_viscosity_value = 1.0e-3
    pressure_gradient_value = -80.0

    radial_coordinate = torch.linspace(
        0.0, radius, 64, dtype=torch.float64
    ).reshape(-1, 1)
    radial_coordinate.requires_grad_(True)
    velocity = hagen_poiseuille_velocity(
        radial_coordinate,
        radius=radius,
        dynamic_viscosity=dynamic_viscosity_value,
        pressure_gradient=pressure_gradient_value,
    )
    dynamic_viscosity = torch.full_like(
        radial_coordinate, dynamic_viscosity_value
    )
    pressure_gradient = torch.full_like(
        radial_coordinate, pressure_gradient_value
    )

    physics = PhysicsInformer(
        required_outputs=["momentum"],
        equations=HagenPoiseuille(),
        grad_method="autodiff",
        device="cpu",
    )
    residual = physics.forward(
        {
            "coordinates": radial_coordinate,
            "x": radial_coordinate,
            "velocity": velocity,
            "dynamic_viscosity": dynamic_viscosity,
            "pressure_gradient": pressure_gradient,
        }
    )["momentum"]

    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_physics_deeponet_canonical_run(tmp_path):
    """Autodiff residuals work in shared training, validation, and saved plots."""
    from argparse import Namespace
    from scripts.pipe_flow import physics_deeponet as model_script
    from scripts.pipe_flow.common import load_physics_split, fit_physics_normalization, PhysicsPipeDataset, physics_experiment_spec
    from chem_operator.experiments import ExperimentRunner, RunContext, load_run
    from chem_operator.plotting import plot_operator_runs
    generator = SimulationDatasetGenerator(HagenPoiseuillePipeFlowSim(), tmp_path / "data")
    records = [generator.simulator.run_case(generator.simulator.make_case(_params(pressure_drop=p,n_radial_points=8)))
               for p in (40.,60.,80.)]
    for split in ("train","valid","test"):
        generator.save_split(split,records)
    statistics = fit_physics_normalization(load_physics_split(tmp_path / "data","train",None))
    data = PhysicsPipeDataset(load_physics_split(tmp_path / "data","train",None).normalized(statistics))
    config = dict(width=4,depth=1,latent_width=4,activation="tanh",learning_rate=1e-3,
                  weight_decay=0.,batch_size=2,epochs=1,physics_weight=1e-4)
    context = RunContext.create(tmp_path / "runs",problem="pipe_flow",model="physics_deeponet",run_id="smoke",seed=42)
    args = Namespace(data_dir=tmp_path / "data",samples=1,tune_epochs=1,plot_cases=2,max_cases=None)
    trainer = model_script.make_trainer(config,context,statistics)
    result = ExperimentRunner(context,physics_experiment_spec(args,"physics_deeponet")).run(
        trainer,data,data,data,config=config,evaluator=lambda t,d,c: model_script.evaluate_run(t,d,c,statistics,cases=2))
    assert result.evaluation.metrics["physics_loss"] >= 0
    assert result.training.metadata["checkpoint_reload_verified"]
    assert load_run(context.paths.run_dir).manifest["status"] == "completed"
    assert all(path.is_file() for path in plot_operator_runs([context.paths.run_dir],tmp_path / "plots"))
