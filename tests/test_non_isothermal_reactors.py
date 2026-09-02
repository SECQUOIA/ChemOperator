from __future__ import annotations

import cantera as ct
import numpy as np
import pytest

from chem_operator.datasets import CaseParameters
from chem_operator.reactors.cstr.dataset_generator import (
    CSTRSim,
    NonIsothermalCSTRSim,
)
from chem_operator.reactors.pfr.dataset_generator import (
    PFRChainOfReactorsSim,
    PFRNonIsothermalChainOfReactorsSim,
)
from chem_operator.sampling import Constant


def _valid_cstr_params() -> dict[str, object]:
    return {
        "phi": 1.0,
        "reactive_fraction": 0.05,
        "T0": 925.0,
        "P0": ct.one_atm,
        "reactor_volume": 1.0e-3,
        "residence_time": 1.0,
        "pressure_controller_K": 1.0e-6,
        "solve_energy": True,
        "ambient_temperature": 300.0,
        "wall_area": 1.0e-2,
        "heat_transfer_coefficient": 10.0,
        "t_final": 0.1,
        "dt": 0.05,
        "adaptive": False,
    }


def _valid_cstr_parameter_space() -> dict[str, Constant]:
    return {
        name: Constant(value)
        for name, value in _valid_cstr_params().items()
    }


def test_non_isothermal_dataset_prefixes():
    assert CSTRSim.name == "cstr"
    assert NonIsothermalCSTRSim.name == "cstr_non_isothermal"
    assert NonIsothermalCSTRSim.make_case is CSTRSim.make_case
    assert NonIsothermalCSTRSim.run_case is CSTRSim.run_case
    assert (
        PFRNonIsothermalChainOfReactorsSim.name
        == "pfr_non_isothermal_chain_of_reactors"
    )


def test_non_isothermal_cstr_requires_a_parameter_space():
    with pytest.raises(ValueError, match="requires a parameter space"):
        NonIsothermalCSTRSim()


@pytest.mark.parametrize(
    "missing_name",
    [
        "solve_energy",
        "ambient_temperature",
        "wall_area",
        "heat_transfer_coefficient",
    ],
)
def test_non_isothermal_cstr_requires_thermal_parameters(missing_name):
    parameter_space = _valid_cstr_parameter_space()
    parameter_space.pop(missing_name)

    with pytest.raises(ValueError, match=missing_name):
        NonIsothermalCSTRSim(parameter_space)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("solve_energy", False),
        ("solve_energy", "True"),
        ("ambient_temperature", 0.0),
        ("ambient_temperature", -1.0),
        ("wall_area", 0.0),
        ("wall_area", -1.0),
        ("heat_transfer_coefficient", 0.0),
        ("heat_transfer_coefficient", -1.0),
    ],
)
def test_non_isothermal_cstr_rejects_invalid_thermal_values(name, value):
    params = _valid_cstr_params()
    params[name] = value
    simulator = NonIsothermalCSTRSim(_valid_cstr_parameter_space())

    with pytest.raises(ValueError, match=name):
        simulator.make_case(params)


def test_base_cstr_allows_cases_without_thermal_controls():
    params = _valid_cstr_params()
    for name in (
        "solve_energy",
        "ambient_temperature",
        "wall_area",
        "heat_transfer_coefficient",
    ):
        params.pop(name)

    case = CSTRSim().make_case(params)

    assert "wall_area" not in case.geometry
    assert "solve_energy" not in case.controls
    assert "ambient_temperature" not in case.controls
    assert "heat_transfer_coefficient" not in case.controls


def test_non_isothermal_run_case_revalidates_direct_case():
    simulator = NonIsothermalCSTRSim(_valid_cstr_parameter_space())
    case = simulator.make_case(_valid_cstr_params())
    case.controls["solve_energy"] = False

    with pytest.raises(ValueError, match="solve_energy"):
        simulator.run_case(case)


def test_cstr_wall_cools_an_inert_reactor():
    case = CaseParameters(
        initial_conditions={
            "gas/T": 1000.0,
            "gas/P": ct.one_atm,
            "gas/X": {"HE": 1.0},
        },
        geometry={"reactor_volume": 1.0e-3, "wall_area": 1.0e-2},
        controls={
            "residence_time": 1.0,
            "pressure_controller_K": 1.0e-6,
            "solve_energy": True,
            "ambient_temperature": 300.0,
            "heat_transfer_coefficient": 10.0,
        },
        solver_parameters={"t_final": 0.1, "dt": 0.05, "adaptive": False},
        mechanism_parameters={},
    )

    simulator = NonIsothermalCSTRSim(_valid_cstr_parameter_space())
    record = simulator.run_case(case)

    np.testing.assert_allclose(record.fields["T"][0], 1000.0)
    assert record.fields["T"][-1] < record.fields["T"][0]


def test_pfr_wall_cools_an_inert_reactor():
    case = CaseParameters(
        initial_conditions={
            "gas/T": 1000.0,
            "gas/P": ct.one_atm,
            "gas/X": {"AR": 1.0},
        },
        geometry={
            "length": 0.03,
            "area": 1.0e-3,
            "wall_area_per_volume": 100.0,
        },
        controls={
            "inlet_velocity": 0.1,
            "pressure_controller_K": 1.0e-12,
            "solve_energy": True,
            "ambient_temperature": 300.0,
            "heat_transfer_coefficient": 100.0,
        },
        solver_parameters={"n_steps": 3, "max_time_step": 1.0e4},
        mechanism_parameters={},
    )

    record = PFRChainOfReactorsSim().run_case(case)

    assert np.all(np.diff(record.fields["T"]) < 0.0)
    assert record.fields["T"][-1] < case.initial_conditions["gas/T"]


def test_pfr_without_new_controls_keeps_energy_enabled():
    case = CaseParameters(
        initial_conditions={
            "gas/T": 1000.0,
            "gas/P": ct.one_atm,
            "gas/X": {"AR": 1.0},
        },
        geometry={"length": 0.02, "area": 1.0e-3},
        controls={
            "inlet_velocity": 0.1,
            "pressure_controller_K": 1.0e-12,
        },
        solver_parameters={"n_steps": 2, "max_time_step": 1.0e4},
        mechanism_parameters={},
    )

    record = PFRChainOfReactorsSim().run_case(case)

    np.testing.assert_allclose(record.fields["T"], 1000.0, rtol=1.0e-12)
