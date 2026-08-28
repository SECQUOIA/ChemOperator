import os
from pathlib import Path
import time
from copy import deepcopy
from collections.abc import Iterable, Sequence, Callable, Mapping
from typing import Any, Literal, Protocol

import cantera as ct
import numpy as np

from chem_operator.utils import get_mechanism_file, datasets_path
from chem_operator.datasets import CaseParameters, CaseSimulator, SimulationRecord, SimulationDatasetGenerator
from chem_operator.sampling import ParameterSpec, Constant, Grid, Uniform

class CSTRCaseSimulator(CaseSimulator):
    name = "cstr"
    _requires_heat_transfer = False

    _required_thermal_parameters = frozenset(
        {
            "solve_energy",
            "ambient_temperature",
            "wall_area",
            "heat_transfer_coefficient",
        }
    )

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        mechanism_file_name: str = "n-heptane-NUIG-2016.yaml",
    ):
        if self._requires_heat_transfer:
            if parameter_space is None:
                raise ValueError(
                    f"{type(self).__name__} requires a parameter space with "
                    "thermal controls."
                )

            missing = self._required_thermal_parameters.difference(parameter_space)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    "Non-isothermal parameter space is missing required "
                    f"parameters: {missing_names}."
                )

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
        params: Mapping[str, Any]
    ) -> CaseParameters:
        phi = params["phi"]
        reactive_fraction = params["reactive_fraction"]

        # X_fuel / X_O2 = phi / 11
        ratio = phi / 11.0

        x_o2 = reactive_fraction / (1.0 + ratio)
        x_fuel = ratio * x_o2
        x_he = 1.0 - reactive_fraction

        geometry = {
            "reactor_volume": params["reactor_volume"],
        }
        controls = {
            "residence_time": params["residence_time"],
            "pressure_controller_K": params["pressure_controller_K"],
        }

        # Only add thermal settings when the parameter space provides them.
        # This keeps cases produced by the original parameter space unchanged.
        if "wall_area" in params:
            geometry["wall_area"] = params["wall_area"]
        for name in (
            "solve_energy",
            "ambient_temperature",
            "heat_transfer_coefficient",
        ):
            if name in params:
                controls[name] = params[name]

        case = CaseParameters(
            initial_conditions={
                "gas/T": params["T0"],
                "gas/P": params["P0"],
                "gas/X": {"NC7H16": x_fuel, "O2": x_o2,"HE": x_he}
            },
            geometry=geometry,
            controls=controls,
            solver_parameters={
                "t_final": params["t_final"],
                "dt": params["dt"],
                "adaptive": params["adaptive"],
            },
            mechanism_parameters={
                "phi": phi,
                "reactive_fraction": reactive_fraction,
            }
        )
        self._validate_thermal_case(case)
        return case

    def _validate_thermal_case(
        self,
        case: CaseParameters,
    ) -> tuple[bool, float, float, float | None]:
        controls = case.controls or {}
        geometry = case.geometry or {}

        if self._requires_heat_transfer:
            missing = self._required_thermal_parameters.difference(
                controls.keys() | geometry.keys()
            )
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(
                    "Non-isothermal case is missing required thermal values: "
                    f"{missing_names}."
                )

        solve_energy_value = controls.get("solve_energy", False)
        if not isinstance(solve_energy_value, (bool, np.bool_)):
            raise ValueError("solve_energy must be a Boolean value.")
        solve_energy = bool(solve_energy_value)

        heat_transfer_coefficient = controls.get("heat_transfer_coefficient", 0.0)
        wall_area = geometry.get("wall_area", 0.0)
        ambient_temperature = controls.get("ambient_temperature")

        if heat_transfer_coefficient < 0.0:
            raise ValueError("heat_transfer_coefficient must be non-negative.")
        if wall_area < 0.0:
            raise ValueError("wall_area must be non-negative.")
        if ambient_temperature is not None and ambient_temperature <= 0.0:
            raise ValueError("ambient_temperature must be positive.")
        if self._requires_heat_transfer and heat_transfer_coefficient == 0.0:
            raise ValueError(
                "heat_transfer_coefficient must be positive for a non-isothermal CSTR."
            )

        if heat_transfer_coefficient > 0.0:
            if wall_area == 0.0:
                raise ValueError(
                    "wall_area must be positive when heat transfer is enabled."
                )
            if not solve_energy:
                raise ValueError(
                    "solve_energy must be True when heat transfer is enabled."
                )
            if ambient_temperature is None:
                raise ValueError(
                    "ambient_temperature is required when heat transfer is enabled."
                )

        return (
            solve_energy,
            heat_transfer_coefficient,
            wall_area,
            ambient_temperature,
        )

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        (
            solve_energy,
            heat_transfer_coefficient,
            wall_area,
            ambient_temperature,
        ) = self._validate_thermal_case(case)

        gas = ct.Solution(get_mechanism_file(self.mechanism_file_name))
        gas.TPX = case.initial_conditions["gas/T"], case.initial_conditions["gas/P"], case.initial_conditions["gas/X"]

        fuel_air_mixture_tank = ct.Reservoir(gas, clone=False)
        exhaust = ct.Reservoir(gas, clone=False)

        stirred_reactor = ct.IdealGasMoleReactor(
            gas,
            energy="on" if solve_energy else "off",
            volume=case.geometry["reactor_volume"],
            clone=False,
        )
        mass_flow_controller = ct.MassFlowController(
            upstream=fuel_air_mixture_tank,
            downstream=stirred_reactor,
            mdot=lambda t: stirred_reactor.mass / case.controls["residence_time"],
        )

        pressure_regulator = ct.PressureController(
            upstream=stirred_reactor,
            downstream=exhaust,
            primary=mass_flow_controller,
            K=case.controls["pressure_controller_K"],
        )

        if heat_transfer_coefficient > 0.0:
            assert ambient_temperature is not None

            ambient_gas = ct.Solution(get_mechanism_file(self.mechanism_file_name))
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
                stirred_reactor,
                environment,
                A=wall_area,
                U=heat_transfer_coefficient,
            )

        reactor_network = ct.ReactorNet([stirred_reactor])

        states = ct.SolutionArray(gas, extra=["t"])

        tic = time.perf_counter()

        if case.solver_parameters["adaptive"]:
            t = 0.0
            states.append(stirred_reactor.phase.state, t=t)
            while t < case.solver_parameters["t_final"]:
                t = reactor_network.step()
                states.append(stirred_reactor.phase.state, t=t)
        else:
            n_steps = int(np.round(case.solver_parameters["t_final"] / case.solver_parameters["dt"]))
            times = np.linspace(0.0, n_steps * case.solver_parameters["dt"], n_steps + 1)
            for t in times:
                reactor_network.advance(t)
                states.append(stirred_reactor.phase.state, t=t)

        toc = time.perf_counter()

        return SimulationRecord(
            coordinates={"t": states.t},
            fields={"T": states.T, "P": states.P, "X": states.X},
            constants=case.geometry | case.controls | case.mechanism_parameters,
            metadata={
                "wall_time": toc - tic,
                "mechanism": self.mechanism_file_name,
                "solver_parameters": deepcopy(case.solver_parameters)
            }
        )


class NonIsothermalCSTRCaseSimulator(CSTRCaseSimulator):
    """CSTR simulator using a separate non-isothermal dataset prefix."""

    name = "cstr_non_isothermal"
    _requires_heat_transfer = True

if __name__ == "__main__":
    if False:
        cstr_simulator = CSTRCaseSimulator(
            parameter_space={
                # sampled physical parameters
                "phi": Uniform(0.5, 2.0),
                "reactive_fraction": Uniform(0.02, 0.08),
                # constants
                "T0": Constant(925.0), # Kelvin
                "P0": Constant(1.046138 * ct.one_atm), # in atm. This equals 1.06 bars
                "reactor_volume": Constant(30.5 * (1e-2) ** 3), # m^3
                "residence_time": Constant(2.0), # s
                "pressure_controller_K": Constant(1e-6),
                # solver controls
                "t_final": Constant(50.0), # s
                "dt": Constant(2e-2), # Not used when adaptive = True
                "adaptive": Constant(False),
            }
        )
        cstr_dataset_generator = SimulationDatasetGenerator(
            cstr_simulator, datasets_path / "cstr",
        )
        records_splits = cstr_dataset_generator.generate_splits(n_cases=50)
        cstr_dataset_generator.save_splits(records_splits, overwrite=True)

    cstr_non_isothermal_param_space: dict[str, ParameterSpec] = {
        # sampled physical parameters
        "phi": Uniform(0.5, 2.0),
        "reactive_fraction": Uniform(0.02, 0.08),
        # constants
        "T0": Constant(925.0),
        "P0": Constant(1.046138 * ct.one_atm),
        "reactor_volume": Constant(30.5 * (1e-2) ** 3),
        "residence_time": Constant(2.0),
        "pressure_controller_K": Constant(1e-6),
        # thermal controls
        "solve_energy": Constant(True), # controls isothermal or non-isothermal
        "ambient_temperature": Constant(600.0),
        "wall_area": Constant(1.0e-2),
        "heat_transfer_coefficient": Uniform(0.001, 2.0),
        # solver controls
        "t_final": Constant(50.0),
        "dt": Constant(2e-2),
        "adaptive": Constant(False),
    }

    if True:
        cstr_non_isothermal_simulator = NonIsothermalCSTRCaseSimulator(
            cstr_non_isothermal_param_space
        )
        cstr_non_isothermal_dataset_generator = SimulationDatasetGenerator(
            cstr_non_isothermal_simulator,
            datasets_path / "cstr",
        )
        records_splits = cstr_non_isothermal_dataset_generator.generate_splits(
            n_cases=50
        )
        cstr_non_isothermal_dataset_generator.save_splits(
            records_splits,
            overwrite=True,
        )
