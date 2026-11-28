import os
from pathlib import Path
import time
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import cantera as ct
import numpy as np

from chem_operator.datasets import (
    CaseParameters,
    CaseSimulator,
    SimulationRecord,
    SimulationDatasetGenerator,
)
from chem_operator.sampling import ParameterSpec, Constant, Uniform
from chem_operator.utils import datasets_path

def _inlet_composition(
    mechanism_file_name: str,
    T0: float,
    P0: float,
    phi: float,
    ar_o2_ratio: float,
) -> dict[str, float]:
    gas = ct.Solution(mechanism_file_name)
    gas.TP = T0, P0
    gas.set_equivalence_ratio(
        phi,
        fuel="H2",
        oxidizer={"O2": 1.0, "AR": ar_o2_ratio},
    )

    return {
        species: float(mole_fraction)
        for species, mole_fraction in zip(gas.species_names, gas.X)
        if mole_fraction > 0.0
    }


def _make_pfr_case(
    params: Mapping[str, Any],
    mechanism_file_name: str,
    include_chain_controls: bool = False,
) -> CaseParameters:
    controls = {
        "inlet_velocity": params["u0"],
    }
    if "solve_energy" in params:
        controls["solve_energy"] = params["solve_energy"]

    solver_parameters = {
        "n_steps": params["n_steps"],
    }
    geometry = {
        "length": params["length"],
        "area": params["area"],
    }

    if include_chain_controls:
        controls["pressure_controller_K"] = params["pressure_controller_K"]
        solver_parameters["max_time_step"] = params["max_time_step"]
        if "wall_area_per_volume" in params:
            geometry["wall_area_per_volume"] = params["wall_area_per_volume"]
        for name in ("ambient_temperature", "heat_transfer_coefficient"):
            if name in params:
                controls[name] = params[name]

    return CaseParameters(
        initial_conditions={
            "gas/T": params["T0"],
            "gas/P": params["P0"],
            "gas/X": _inlet_composition(
                mechanism_file_name=mechanism_file_name,
                T0=params["T0"],
                P0=params["P0"],
                phi=params["phi"],
                ar_o2_ratio=params["ar_o2_ratio"],
            ),
        },
        geometry=geometry,
        controls=controls,
        solver_parameters=solver_parameters,
        mechanism_parameters={
            "phi": params["phi"],
            "ar_o2_ratio": params["ar_o2_ratio"],
        },
    )

def _initial_gas(
    case: CaseParameters,
    mechanism_file_name: str,
) -> ct.Solution:
    gas = ct.Solution(mechanism_file_name)
    gas.TPX = (
        case.initial_conditions["gas/T"],
        case.initial_conditions["gas/P"],
        case.initial_conditions["gas/X"],
    )
    return gas

class PFRLagrangianParticleSim(CaseSimulator):
    name = "pfr_lagrangian_particle"

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        mechanism_file_name: str = "h2o2.yaml",
    ):
        if parameter_space is None:
            self._parameter_space = {}
        else:
            self._parameter_space = dict(parameter_space)

        self.mechanism_file_name = mechanism_file_name

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    def make_case(
        self,
        params: Mapping[str, Any],
    ) -> CaseParameters:
        return _make_pfr_case(params, self.mechanism_file_name)

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        gas = _initial_gas(case, self.mechanism_file_name)

        area = case.geometry["area"]
        length = case.geometry["length"]
        inlet_velocity = case.controls["inlet_velocity"]
        n_steps = case.solver_parameters["n_steps"]

        mass_flow_rate = inlet_velocity * gas.density * area

        solve_energy = bool(case.controls.get("solve_energy", True))
        if float(case.controls.get("heat_transfer_coefficient", 0.0)) > 0.0:
            raise ValueError(
                "Wall heat transfer is only supported by "
                "PFRChainOfReactorsSim."
            )

        reactor = ct.IdealGasConstPressureReactor(
            gas,
            energy="on" if solve_energy else "off",
            clone=True,
        )
        reactor_network = ct.ReactorNet([reactor])

        t_total = length / inlet_velocity
        dt = t_total / n_steps

        times = (np.arange(n_steps) + 1) * dt
        z = np.zeros_like(times)
        velocity = np.zeros_like(times)
        states = ct.SolutionArray(reactor.phase)

        tic = time.perf_counter()

        for i, t_i in enumerate(times):
            reactor_network.advance(t_i)

            velocity[i] = mass_flow_rate / (area * reactor.phase.density)
            z[i] = z[i - 1] + velocity[i] * dt

            states.append(reactor.phase.state)

        toc = time.perf_counter()

        return SimulationRecord(
            coordinates={"t": times},
            fields={
                "z": z,
                "T": states.T,
                "P": states.P,
                "X": states.X,
                "velocity": velocity,
            },
            constants=case.geometry | case.controls | case.mechanism_parameters,
            metadata={
                "wall_time": toc - tic,
                "mechanism": self.mechanism_file_name,
                "solver_parameters": deepcopy(case.solver_parameters),
            },
        )


class PFRChainOfReactorsSim(CaseSimulator):
    name = "pfr_chain"

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        mechanism_file_name: str = "h2o2.yaml",
    ):
        if parameter_space is None:
            self._parameter_space = {}
        else:
            self._parameter_space = dict(parameter_space)

        self.mechanism_file_name = mechanism_file_name

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    def make_case(
        self,
        params: Mapping[str, Any],
    ) -> CaseParameters:
        return _make_pfr_case(
            params,
            self.mechanism_file_name,
            include_chain_controls=True,
        )

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        gas = _initial_gas(case, self.mechanism_file_name)

        area = case.geometry["area"]
        length = case.geometry["length"]
        inlet_velocity = case.controls["inlet_velocity"]
        pressure_controller_K = case.controls["pressure_controller_K"]
        n_steps = case.solver_parameters["n_steps"]
        max_time_step = case.solver_parameters["max_time_step"]

        solve_energy = bool(case.controls.get("solve_energy", True))
        heat_transfer_coefficient = float(
            case.controls.get("heat_transfer_coefficient", 0.0)
        )
        wall_area_per_volume = float(
            case.geometry.get("wall_area_per_volume", 0.0)
        )

        if heat_transfer_coefficient < 0.0:
            raise ValueError("heat_transfer_coefficient must be non-negative.")
        if wall_area_per_volume < 0.0:
            raise ValueError("wall_area_per_volume must be non-negative.")
        if heat_transfer_coefficient > 0.0 and wall_area_per_volume == 0.0:
            raise ValueError(
                "wall_area_per_volume must be positive when heat transfer is enabled."
            )
        if heat_transfer_coefficient > 0.0 and not solve_energy:
            raise ValueError(
                "solve_energy must be True when heat transfer is enabled."
            )

        mass_flow_rate = inlet_velocity * gas.density * area
        dz = length / n_steps
        reactor_volume = area * dz

        reactor = ct.IdealGasReactor(
            gas,
            energy="on" if solve_energy else "off",
            clone=True,
        )
        reactor.volume = reactor_volume

        upstream = ct.Reservoir(gas, name="upstream", clone=True)
        downstream = ct.Reservoir(gas, name="downstream", clone=True)

        mass_flow_controller = ct.MassFlowController(
            upstream,
            reactor,
            mdot=mass_flow_rate,
        )
        pressure_controller = ct.PressureController(
            reactor,
            downstream,
            primary=mass_flow_controller,
            K=pressure_controller_K,
        )

        if heat_transfer_coefficient > 0.0:
            ambient_temperature = float(case.controls["ambient_temperature"])
            if ambient_temperature <= 0.0:
                raise ValueError("ambient_temperature must be positive.")

            ambient_gas = ct.Solution(self.mechanism_file_name)
            ambient_gas.TPX = (
                ambient_temperature,
                case.initial_conditions["gas/P"],
                case.initial_conditions["gas/X"],
            )
            environment = ct.Reservoir(
                ambient_gas,
                name="environment",
                clone=False,
            )
            thermal_wall = ct.Wall(
                reactor,
                environment,
                A=wall_area_per_volume * reactor_volume,
                U=heat_transfer_coefficient,
            )

        reactor_network = ct.ReactorNet([reactor])
        reactor_network.max_time_step = max_time_step

        # Define time, space, and other information vectors before solving.
        z = (np.arange(n_steps) + 1) * dz
        residence_time = np.zeros_like(z)
        velocity = np.zeros_like(z)
        times = np.zeros_like(z)
        states = ct.SolutionArray(gas)

        tic = time.perf_counter()

        for i in range(n_steps):
            upstream.phase.TDY = reactor.phase.TDY
            upstream.syncState()

            reactor_network.reinitialize()
            reactor_network.solve_steady()

            velocity[i] = mass_flow_rate / (area * reactor.phase.density)
            residence_time[i] = reactor.mass / mass_flow_rate
            times[i] = np.sum(residence_time)

            states.append(reactor.phase.state)

        toc = time.perf_counter()

        return SimulationRecord(
            coordinates={"z": z},
            fields={
                "t": times,
                "T": states.T,
                "P": states.P,
                "X": states.X,
                "velocity": velocity,
                "residence_time": residence_time,
            },
            constants=case.geometry | case.controls | case.mechanism_parameters,
            metadata={
                "wall_time": toc - tic,
                "mechanism": self.mechanism_file_name,
                "solver_parameters": deepcopy(case.solver_parameters),
            },
        )


class PFRNonIsothermalChainOfReactorsSim(PFRChainOfReactorsSim):
    """PFR chain using a separate non-isothermal dataset prefix."""

    name = "pfr_non_isothermal_chain"
