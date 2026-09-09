"""Datasets for the three-species PFR with a conducting cylindrical wall."""

# The single simulator intentionally keeps the coupled BVP state together.
# pylint: disable=too-many-instance-attributes,too-many-locals

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import time
from typing import Any

import numpy as np
from physicsnemo.sym.eq.pde import PDE
from scipy.integrate import solve_bvp
from sympy import Function, Number, Symbol, exp

from chem_operator.datasets import CaseParameters, CaseSimulator, SimulationRecord
from chem_operator.sampling import ParameterSpec


@dataclass(frozen=True)
class ModelConstants:
    """Chemistry and scales shared with the PhysicsNeMo example."""

    flow_scale: float = 200.0
    concentration_scale: float = 0.1
    temperature_scale: float = 800.0
    volume_scale: float = 1.0
    heat_capacity_scale: float = 200.0
    reference_temperature: float = 300.0
    reaction_1_pre_exponential: float = 10.0
    reaction_1_activation_temperature: float = 4000.0
    reaction_1_heat: float = -20_000.0
    reaction_2_pre_exponential: float = 0.5
    reaction_2_activation_temperature: float = 9000.0
    reaction_2_heat: float = -60_000.0
    heat_capacities: tuple[float, float, float] = (90.0, 90.0, 180.0)


class PlugFlowReactor(PDE):
    """PhysicsNeMo molar-flow and gas-energy equations along ``z``."""

    def __init__(
        self,
        constants: ModelConstants,
        inlet_concentration_a: float,
        inlet_temperature: float,
        volumetric_heat_transfer: float,
    ) -> None:
        self.dim = 1
        z = Symbol("x")
        f_a = Function("f_a")(z)  # pylint: disable=not-callable
        f_b = Function("f_b")(z)  # pylint: disable=not-callable
        f_c = Function("f_c")(z)  # pylint: disable=not-callable
        t_gas = Function("t_gas")(z)  # pylint: disable=not-callable
        t_wall = Function("t_wall")(z)  # pylint: disable=not-callable

        total_flow = f_a + f_b + f_c
        c_a = (
            Number(inlet_concentration_a / constants.concentration_scale)
            * f_a
            / total_flow
            * Number(inlet_temperature / constants.temperature_scale)
            / t_gas
        )
        inverse_temperature = (
            Number(constants.temperature_scale / constants.reference_temperature)
            - 1 / t_gas
        )
        q1 = (
            Number(constants.reaction_1_pre_exponential)
            * exp(
                Number(
                    constants.reaction_1_activation_temperature
                    / constants.temperature_scale
                )
                * inverse_temperature
            )
            * c_a
            * Number(constants.concentration_scale)
        )
        q2 = (
            Number(constants.reaction_2_pre_exponential)
            * exp(
                Number(
                    constants.reaction_2_activation_temperature
                    / constants.temperature_scale
                )
                * inverse_temperature
            )
            * (c_a * Number(constants.concentration_scale)) ** 2
        )
        flow_factor = Number(constants.volume_scale / constants.flow_scale)
        heat_capacity_flow = sum(
            flow * Number(cp / constants.heat_capacity_scale)
            for flow, cp in zip(
                (f_a, f_b, f_c), constants.heat_capacities, strict=True
            )
        )
        wall_heat = Number(
            volumetric_heat_transfer
            * constants.volume_scale
            / (constants.flow_scale * constants.heat_capacity_scale)
        ) * (t_wall - t_gas)
        reaction_heat = sum(
            Number(
                -heat
                * constants.volume_scale
                / (
                    constants.flow_scale
                    * constants.temperature_scale
                    * constants.heat_capacity_scale
                )
            )
            * rate
            for heat, rate in (
                (constants.reaction_1_heat, q1),
                (constants.reaction_2_heat, q2),
            )
        )

        self.equations = {
            "flow_a": f_a.diff(z) + flow_factor * (q1 + q2),
            "flow_b": f_b.diff(z) - flow_factor * q1,
            "flow_c": f_c.diff(z) - Number(1.5) * flow_factor * q2,
            "gas_energy": (
                t_gas.diff(z) - (wall_heat + reaction_heat) / heat_capacity_flow
            ),
        }


class CylindricalWall(PDE):
    """PhysicsNeMo solid conduction and gas/solid interface equations."""

    def __init__(self, wall_aspect_ratio_sq: float, interface_biot: float) -> None:
        self.dim = 2
        z, r = Symbol("x"), Symbol("y")
        t_solid = Function("t_solid")(z, r)  # pylint: disable=not-callable
        t_gas = Function("t_gas")(z, r)  # pylint: disable=not-callable
        self.equations = {
            "solid_heat": (
                Number(wall_aspect_ratio_sq) * t_solid.diff(z, 2)
                + t_solid.diff(r, 2)
                + t_solid.diff(r) / r
            ),
            "interface": (
                t_solid.diff(r)
                - Number(interface_biot) * (t_solid - t_gas)
            ),
            "axial_flux": t_solid.diff(z),
        }


class PFRHeatSim(CaseSimulator):
    """Solve the coupled axial PFR/radial-wall boundary-value problem."""

    name = "pfr_cylindrical_heat"
    species = ("A", "B", "C")

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        constants: ModelConstants = ModelConstants(),
    ) -> None:
        self._parameter_space = {} if parameter_space is None else dict(parameter_space)
        self.constants = constants

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    @staticmethod
    def _positive(params: Mapping[str, Any], name: str) -> float:
        value = float(params[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite.")
        return value

    @staticmethod
    def _points(params: Mapping[str, Any], name: str) -> int:
        raw = params[name]
        value = int(raw)
        if value != raw or value < 3:
            raise ValueError(f"{name} must be an integer of at least 3.")
        return value

    def make_case(self, params: Mapping[str, Any]) -> CaseParameters:
        inner_radius = self._positive(params, "inner_radius")
        outer_radius = self._positive(params, "outer_radius")
        if outer_radius <= inner_radius:
            raise ValueError("outer_radius must exceed inner_radius.")
        n_axial_points = self._points(params, "n_axial_points")
        max_nodes = self._points(params, "max_nodes")
        if max_nodes < n_axial_points:
            raise ValueError("max_nodes must be at least n_axial_points.")

        return CaseParameters(
            initial_conditions={
                "inlet_flow_a": self._positive(params, "inlet_flow_a"),
                "inlet_concentration_a": self._positive(
                    params, "inlet_concentration_a"
                ),
                "inlet_temperature": self._positive(params, "inlet_temperature"),
            },
            boundary_conditions={
                "outer_temperature": self._positive(params, "outer_temperature"),
                "solid_axial_flux_inlet": 0.0,
                "solid_axial_flux_outlet": 0.0,
            },
            geometry={"inner_radius": inner_radius, "outer_radius": outer_radius},
            controls={
                "volumetric_heat_transfer": self._positive(
                    params, "volumetric_heat_transfer"
                ),
                "wall_aspect_ratio_sq": self._positive(
                    params, "wall_aspect_ratio_sq"
                ),
                "interface_biot": self._positive(params, "interface_biot"),
            },
            physical_parameters=asdict(self.constants),
            solver_parameters={
                "n_axial_points": n_axial_points,
                "n_radial_points": self._points(params, "n_radial_points"),
                "tolerance": self._positive(params, "tolerance"),
                "max_nodes": max_nodes,
            },
            mechanism_parameters={"species": self.species},
        )

    def _rhs(self, case: CaseParameters, n_wall: int, dr: float):
        constants = self.constants
        initial = case.initial_conditions
        boundary = case.boundary_conditions
        controls = case.controls
        geometry = case.geometry
        assert initial and boundary and controls and geometry

        inlet_c = initial["inlet_concentration_a"] / constants.concentration_scale
        inlet_t = initial["inlet_temperature"] / constants.temperature_scale
        outer_t = boundary["outer_temperature"] / constants.temperature_scale
        reference_t = constants.reference_temperature / constants.temperature_scale
        radii = np.linspace(
            geometry["inner_radius"], geometry["outer_radius"], n_wall + 1
        )
        cp = np.asarray(constants.heat_capacities) / constants.heat_capacity_scale
        flow_factor = constants.volume_scale / constants.flow_scale
        wall_factor = (
            controls["volumetric_heat_transfer"]
            * constants.volume_scale
            / (constants.flow_scale * constants.heat_capacity_scale)
        )
        heat_factors = -np.array(
            [constants.reaction_1_heat, constants.reaction_2_heat]
        ) * constants.volume_scale / (
            constants.flow_scale
            * constants.temperature_scale
            * constants.heat_capacity_scale
        )

        def rhs(_z: np.ndarray, state: np.ndarray) -> np.ndarray:
            flows = np.maximum(state[:3], 1.0e-10)
            gas_t = np.maximum(state[3], 1.0e-8)
            wall_t = state[4 : 4 + n_wall]
            wall_z = state[4 + n_wall :]

            total_flow = np.maximum(flows.sum(axis=0), 1.0e-10)
            c_a = inlet_c * flows[0] / total_flow * inlet_t / gas_t
            inverse_t = 1.0 / reference_t - 1.0 / gas_t
            q1 = constants.reaction_1_pre_exponential * np.exp(
                np.clip(
                    constants.reaction_1_activation_temperature
                    / constants.temperature_scale
                    * inverse_t,
                    -60.0,
                    60.0,
                )
            ) * (c_a * constants.concentration_scale)
            q2 = constants.reaction_2_pre_exponential * np.exp(
                np.clip(
                    constants.reaction_2_activation_temperature
                    / constants.temperature_scale
                    * inverse_t,
                    -60.0,
                    60.0,
                )
            ) * (c_a * constants.concentration_scale) ** 2

            derivatives = np.empty_like(state)
            derivatives[0] = -flow_factor * (q1 + q2)
            derivatives[1] = flow_factor * q1
            derivatives[2] = 1.5 * flow_factor * q2
            heat_capacity_flow = np.maximum(cp @ flows, 1.0e-10)
            wall_heat = wall_factor * (wall_t[0] - gas_t)
            derivatives[3] = (
                wall_heat + heat_factors[0] * q1 + heat_factors[1] * q2
            ) / heat_capacity_flow
            derivatives[4 : 4 + n_wall] = wall_z

            radial_laplacian = np.empty_like(wall_t)
            interface_jump = wall_t[0] - gas_t
            ghost = wall_t[1] - 2.0 * dr * controls["interface_biot"] * interface_jump
            radial_laplacian[0] = (
                (wall_t[1] - 2.0 * wall_t[0] + ghost) / dr**2
                + (wall_t[1] - ghost) / (2.0 * dr * radii[0])
            )
            radial_laplacian[1:-1] = (
                (wall_t[2:] - 2.0 * wall_t[1:-1] + wall_t[:-2]) / dr**2
                + (wall_t[2:] - wall_t[:-2])
                / (2.0 * dr * radii[1 : n_wall - 1, None])
            )
            radial_laplacian[-1] = (
                (outer_t - 2.0 * wall_t[-1] + wall_t[-2]) / dr**2
                + (outer_t - wall_t[-2]) / (2.0 * dr * radii[-2])
            )
            derivatives[4 + n_wall :] = (
                -radial_laplacian / controls["wall_aspect_ratio_sq"]
            )
            return derivatives

        return rhs

    def run_case(self, case: CaseParameters) -> SimulationRecord:
        initial = case.initial_conditions
        boundary = case.boundary_conditions
        geometry = case.geometry
        controls = case.controls
        physical_parameters = case.physical_parameters
        solver_parameters = case.solver_parameters
        assert (
            initial
            and boundary
            and geometry
            and controls
            and physical_parameters
            and solver_parameters
        )

        n_z = int(solver_parameters["n_axial_points"])
        n_r = int(solver_parameters["n_radial_points"])
        n_wall = n_r - 1
        z = np.linspace(0.0, 1.0, n_z)
        r = np.linspace(geometry["inner_radius"], geometry["outer_radius"], n_r)
        dr = r[1] - r[0]

        inlet_flow = initial["inlet_flow_a"] / self.constants.flow_scale
        inlet_t = initial["inlet_temperature"] / self.constants.temperature_scale
        outer_t = boundary["outer_temperature"] / self.constants.temperature_scale
        guess = np.zeros((4 + 2 * n_wall, n_z))
        guess[0] = inlet_flow * (1.0 - 0.1 * z)
        guess[1] = 0.07 * inlet_flow * z
        guess[2] = 0.04 * inlet_flow * z
        guess[3] = inlet_t + 0.2 * (outer_t - inlet_t) * z
        log_fraction = np.log(geometry["outer_radius"] / r[:-1]) / np.log(
            geometry["outer_radius"] / geometry["inner_radius"]
        )
        guess[4 : 4 + n_wall] = outer_t + np.outer(
            log_fraction, guess[3] - outer_t
        )

        q_slice = slice(4 + n_wall, 4 + 2 * n_wall)

        def boundary_residual(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.concatenate(
                (
                    left[:4] - [inlet_flow, 0.0, 0.0, inlet_t],
                    left[q_slice],
                    right[q_slice],
                )
            )

        tic = time.perf_counter()
        solution = solve_bvp(
            self._rhs(case, n_wall, dr),
            boundary_residual,
            z,
            guess,
            tol=solver_parameters["tolerance"],
            max_nodes=int(solver_parameters["max_nodes"]),
        )
        if not solution.success:
            raise RuntimeError(f"Coupled PFR solve failed: {solution.message}")
        state = solution.sol(z)
        toc = time.perf_counter()

        flows = state[:3].T
        gas_t = state[3]
        total_flow = np.maximum(flows.sum(axis=1, keepdims=True), 1.0e-10)
        concentrations = (
            initial["inlet_concentration_a"]
            * flows
            / total_flow
            * initial["inlet_temperature"]
            / (gas_t[:, None] * self.constants.temperature_scale)
        )
        solid_t = np.column_stack(
            (state[4 : 4 + n_wall].T, np.full(n_z, outer_t))
        )

        return SimulationRecord(
            coordinates={"z": z, "r": r},
            fields={
                "F": flows * self.constants.flow_scale,
                "C": concentrations,
                "T_gas": gas_t * self.constants.temperature_scale,
                "T_solid": solid_t * self.constants.temperature_scale,
            },
            constants=(
                initial
                | boundary
                | geometry
                | controls
                | physical_parameters
            ),
            metadata={
                "wall_time": toc - tic,
                "equation": "three-species PFR with cylindrical wall conduction",
                "solver": "scipy.solve_bvp with radial finite differences",
                "species": self.species,
                "solver_iterations": solution.niter,
                "solver_nodes": solution.x.size,
                "physicsnemo_pdes": {
                    "reactor": PlugFlowReactor.__name__,
                    "solid": CylindricalWall.__name__,
                },
                "solver_parameters": deepcopy(solver_parameters),
                "units": {
                    "z": "z/L",
                    "r": "r/Ri",
                    "F": "mol/s",
                    "C": "mol/L",
                    "T_gas": "K",
                    "T_solid": "K",
                },
            },
        )
