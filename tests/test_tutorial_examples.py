from __future__ import annotations

import cantera as ct
import numpy as np

from chem_operator.datasets import CaseParameters
from chem_operator.utils import get_mechanism_file
from chem_operator.reactors.cstr.dataset_generator import CSTRSim
from chem_operator.reactors.packed_bed_1D.dataset_generator import PackedBed1DSimulator

def _run_cstr_tutorial_reference(case: CaseParameters) -> tuple[float, float, np.ndarray]:
    # Cantera tutorial:
    # https://cantera.org/dev/examples/python/reactors/continuous_reactor.html
    gas = ct.Solution(get_mechanism_file("n-heptane-NUIG-2016.yaml"))
    gas.TPX = (
        case.initial_conditions["gas/T"],
        case.initial_conditions["gas/P"],
        case.initial_conditions["gas/X"],
    )

    fuel_air_mixture_tank = ct.Reservoir(gas, clone=False)
    exhaust = ct.Reservoir(gas, clone=False)
    stirred_reactor = ct.IdealGasMoleReactor(
        gas,
        energy="off",
        volume=case.geometry["reactor_volume"],
        clone=False,
    )
    mass_flow_controller = ct.MassFlowController(
        upstream=fuel_air_mixture_tank,
        downstream=stirred_reactor,
        mdot=lambda t: stirred_reactor.mass / case.controls["residence_time"],
    )
    ct.PressureController(
        upstream=stirred_reactor,
        downstream=exhaust,
        primary=mass_flow_controller,
        K=case.controls["pressure_controller_K"],
    )

    reactor_network = ct.ReactorNet([stirred_reactor])
    t = 0.0
    while t < case.solver_parameters["t_final"]:
        t = reactor_network.step()

    phase = stirred_reactor.phase
    return phase.T, phase.P, phase.X.copy()


def test_cstr_matches_continuous_reactor_tutorial_case():
    tutorial_case = CaseParameters(
        initial_conditions={
            "gas/T": 925,
            "gas/P": 1.046138 * ct.one_atm,
            "gas/X": {"NC7H16": 0.005, "O2": 0.0275, "HE": 0.9675},
        },
        geometry={
            "reactor_volume": 30.5 * (1e-2) ** 3,
        },
        controls={
            "residence_time": 2,
            "pressure_controller_K": 1e-6,
        },
        solver_parameters={
            "t_final": 50.0,
            "dt": None,
            "adaptive": True,
        },
        mechanism_parameters={},
    )

    tutorial_T, tutorial_P, tutorial_X = _run_cstr_tutorial_reference(tutorial_case)
    record = CSTRSim().run_case(tutorial_case)

    assert record.coordinates["t"][-1] >= tutorial_case.solver_parameters["t_final"]
    np.testing.assert_allclose(record.fields["T"][-1], tutorial_T, rtol=1e-10)
    np.testing.assert_allclose(record.fields["P"][-1], tutorial_P, rtol=1e-10)
    np.testing.assert_allclose(record.fields["X"][-1], tutorial_X, rtol=1e-8, atol=1e-14)


def test_packed_bed_matches_1d_packed_bed_tutorial_final_state():
    # Cantera tutorial:
    # https://cantera.org/dev/examples/python/reactors/1D_packed_bed.html
    expected_z = 0.05023234878932287
    expected_final_state = np.array(
        [
            1.07321086e-03,
            4.99997967e05,
            6.85046090e02,
            8.79748457e-02,
            4.78742610e-01,
            4.09667771e-01,
            2.31456973e-02,
            3.26029248e-03,
            9.93143157e-01,
            2.67793201e-03,
            8.56449959e-04,
            3.65351166e-08,
            6.21317364e-05,
        ]
    )

    tutorial_case = CaseParameters(
        initial_conditions={
            "gas/T": 673.0,
            "gas/P": 5e5,
            "gas/X": {"NH3": 0.99, "AR": 0.01},
        },
        geometry={
            "length": 5e-2,
            "radius": 5e-3,
            "porosity": 0.5,
            "tortuosity": 2.0,
            "particle_diameter": 3.37e-4,
            "specific_surface_area": 3.5e6,
        },
        controls={
            "inlet_velocity": 0.001,
            "wall_temperature": 723.0,
            "heat_transfer_coefficient": 1e2,
            "solve_energy": True,
            "membrane_present": True,
            "membrane_permeability": 1e-15,
            "membrane_thickness": 3e-6,
            "membrane_species": "H2",
            "sweep_pressure": 1e5,
        },
        solver_parameters={
            "first_step_size": 1e-15,
            "atol": 1e-14,
            "rtol": 1e-6,
            "max_steps": 8000,
        },
        mechanism_parameters={
            "inlet_nh3_mole_fraction": 0.99,
            "diluent_species": "AR",
        },
    )
    simulator = PackedBed1DSimulator()
    record = simulator.run_case(tutorial_case)
    final_state = np.concatenate(
        [
            [record.fields["rhou"][-1], record.fields["P"][-1], record.fields["T"][-1]],
            record.fields["Y"][-1],
            record.fields["Z"][-1],
        ]
    )

    np.testing.assert_allclose(record.coordinates["z"][-1], expected_z, rtol=1e-4)
    np.testing.assert_allclose(final_state, expected_final_state, rtol=1e-5, atol=1e-5)
