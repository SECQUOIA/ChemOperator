"""Tests for analytical startup-flow dataset generation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import jn_zeros  # pylint: disable=no-name-in-module
from sympy import Derivative, Symbol

from chem_operator.datasets import ChemOperatorDataset, SimulationDatasetGenerator
from chem_operator.reactors.pipe_flow_transient.dataset_generator import (
    TransientHagenPoiseuille,
    TransientHagenPoiseuillePipeFlowSim,
    startup_hagen_poiseuille_dimensionless_velocity,
)
from chem_operator.sampling import Constant


def _params(**overrides: float | int) -> dict[str, float | int]:
    params: dict[str, float | int] = {
        "radius": 1.0e-3,
        "length": 1.0,
        "dynamic_viscosity": 1.0e-3,
        "density": 1000.0,
        "pressure_drop": 80.0,
        "n_time_points": 32,
        "n_radial_points": 32,
        "max_fourier_number": 2.0,
    }
    params.update(overrides)
    return params


@pytest.fixture(name="simulator")
def fixture_simulator() -> TransientHagenPoiseuillePipeFlowSim:
    """Return a deterministic analytical simulator."""
    values = _params()
    return TransientHagenPoiseuillePipeFlowSim(
        parameter_space={name: Constant(value) for name, value in values.items()},
        series_terms=128,
    )


def test_generated_profile_is_exact_startup_solution(
    simulator: TransientHagenPoiseuillePipeFlowSim,
) -> None:
    """The generated profile satisfies exact startup-flow properties."""
    case = simulator.make_case(_params())
    record = simulator.run_case(case)

    radius = float(case.geometry["radius"])
    viscosity = float(case.physical_parameters["dynamic_viscosity"])
    density = float(case.physical_parameters["density"])
    time_values = record.coordinates["t"]
    radial_values = record.coordinates["r"]
    fourier_number = (viscosity / density) * time_values / radius**2
    radius_fraction = radial_values / radius
    normalized_velocity = (
        record.fields["velocity"] / record.metadata["steady_max_velocity"]
    )

    np.testing.assert_allclose(normalized_velocity[0], 0.0, atol=0.0)
    np.testing.assert_allclose(normalized_velocity[:, -1], 0.0, atol=0.0)
    assert np.all(np.diff(normalized_velocity[1:], axis=1) <= 1.0e-12)

    steady_profile = 1.0 - radius_fraction**2
    np.testing.assert_allclose(
        normalized_velocity[-1],
        steady_profile,
        rtol=2.0e-5,
        atol=2.0e-5,
    )

    roots = jn_zeros(0, 256)
    exact_flow_ratio = 1.0 - 32.0 * np.sum(
        np.exp(-fourier_number[:, None] * roots[None, :] ** 2)
        / roots[None, :] ** 4,
        axis=1,
    )
    exact_flow_ratio[0] = 0.0
    np.testing.assert_allclose(
        record.fields["flow_rate"],
        record.metadata["steady_flow_rate"] * exact_flow_ratio,
        rtol=2.0e-3,
        atol=1.0e-16,
    )
    assert record.metadata["solver"] == "analytical Fourier-Bessel series"
    assert record.metadata["analytical_solution"]["series_terms"] == 128
    assert "pinn" not in record.metadata


def test_analytical_helper_validates_and_enforces_edges() -> None:
    """The analytical evaluator handles the exact initial and wall values."""
    velocity = startup_hagen_poiseuille_dimensionless_velocity(
        np.array([0.0, 0.5]),
        np.array([0.0, 0.5, 1.0]),
    )
    assert velocity.shape == (2, 3)
    np.testing.assert_array_equal(velocity[0], 0.0)
    np.testing.assert_array_equal(velocity[:, -1], 0.0)

    with pytest.raises(ValueError, match="positive integer"):
        startup_hagen_poiseuille_dimensionless_velocity(
            np.array([0.0]),
            np.array([0.0]),
            n_terms=0,
        )


def test_transient_pde_contains_time_derivative() -> None:
    """The PhysicsNeMo equation exposes the expected residual."""
    equation = TransientHagenPoiseuille()
    assert set(equation.equations) == {"momentum"}
    derivatives = equation.equations["momentum"].atoms(Derivative)
    assert any(
        derivative.variables == (Symbol("t"),)
        for derivative in derivatives
    )


def test_equal_coordinate_lengths_round_trip(
    tmp_path,
    simulator: TransientHagenPoiseuillePipeFlowSim,
) -> None:
    """The time axis is sliced while the independent radial axis stays whole."""
    generator = SimulationDatasetGenerator(simulator, tmp_path, seed=7)
    splits = generator.generate_splits(n_cases=3)
    generator.save_splits(splits)

    dataset = ChemOperatorDataset(
        tmp_path / "pipe_flow_transient_train.h5",
        task="next_step",
        coordinate_name="t",
        input_fields=("velocity", "flow_rate"),
        output_fields=("velocity", "flow_rate"),
        constant_inputs=(
            "radius",
            "length",
            "dynamic_viscosity",
            "density",
            "pressure_drop",
            "pressure_gradient",
        ),
        n_steps_input=2,
        n_steps_output=1,
        dtype=torch.float64,
    )
    try:
        sample = dataset[0]
        assert sample["input_fields"]["velocity"].shape == (2, 32)
        assert sample["output_fields"]["velocity"].shape == (1, 32)
        assert sample["input_fields"]["flow_rate"].shape == (2,)
        assert sample["input_coordinates"]["t"].shape == (2,)
        assert sample["output_coordinates"]["t"].shape == (1,)
        assert sample["input_coordinates"]["r"].shape == (32,)
        assert sample["output_coordinates"]["r"].shape == (32,)
        assert set(sample["constant_inputs"]) == {
            "radius",
            "length",
            "dynamic_viscosity",
            "density",
            "pressure_drop",
            "pressure_gradient",
        }
        assert (
            sample["metadata"]["solver"]
            == "analytical Fourier-Bessel series"
        )
        assert sample["metadata"]["units"]["velocity"] == "m/s"
        assert sample["metadata"]["solver_parameters"]["n_time_points"] == 32
    finally:
        dataset.close()


def test_non_laminar_case_is_rejected() -> None:
    """Cases outside the governing assumptions fail before evaluation."""
    simulator = TransientHagenPoiseuillePipeFlowSim()
    case = simulator.make_case(
        _params(radius=1.0e-2, length=0.1, pressure_drop=1000.0)
    )
    with pytest.raises(ValueError, match="must remain laminar"):
        simulator.run_case(case)
