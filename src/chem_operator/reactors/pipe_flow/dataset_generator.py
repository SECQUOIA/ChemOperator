"""Analytical Hagen-Poiseuille dataset generation for circular pipes."""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from physicsnemo.sym.eq.pde import PDE
from sympy import Function, Symbol

from chem_operator.datasets import (
    CaseParameters,
    CaseSimulator,
    SimulationDatasetGenerator,
    SimulationRecord,
)
from chem_operator.sampling import Constant, ParameterSpec, Uniform

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LAMINAR_REYNOLDS_LIMIT = 2300.0

class HagenPoiseuille(PDE):
    """PhysicsNeMo form of steady, fully developed circular-pipe flow."""

    def __init__(self) -> None:
        self.dim = 1

        # PhysicsInformer recognizes x as its first spatial coordinate. Here x is
        # the physical radial coordinate r.
        x = Symbol("x")
        velocity = Function("velocity")(x)  # pylint: disable=not-callable
        dynamic_viscosity = Symbol("dynamic_viscosity")
        pressure_gradient = Symbol("pressure_gradient")

        # Multiplication by r removes the cylindrical 1/r singularity at the
        # centerline while retaining the symmetry condition du/dr = 0 there.
        self.equations = {
            "momentum": (
                dynamic_viscosity
                * (x * velocity.diff(x, 2) + velocity.diff(x))
                - pressure_gradient * x
            )
        }


def hagen_poiseuille_velocity(
    radial_coordinate: torch.Tensor,
    *,
    radius: float | torch.Tensor,
    dynamic_viscosity: float | torch.Tensor,
    pressure_gradient: float | torch.Tensor,
) -> torch.Tensor:
    """Return the analytical axial velocity profile in a circular pipe."""
    return (
        -pressure_gradient
        * (radius**2 - radial_coordinate**2)
        / (4.0 * dynamic_viscosity)
    )


def _flow_diagnostics(
    *,
    radius: float,
    length: float,
    dynamic_viscosity: float,
    density: float,
    pressure_drop: float,
) -> dict[str, float]:
    flow_rate = (
        np.pi
        * radius**4
        * pressure_drop
        / (8.0 * dynamic_viscosity * length)
    )
    mean_velocity = flow_rate / (np.pi * radius**2)
    return {
        "flow_rate": float(flow_rate),
        "mean_velocity": float(mean_velocity),
        "max_velocity": float(2.0 * mean_velocity),
        "reynolds_number": float(
            density * mean_velocity * (2.0 * radius) / dynamic_viscosity
        ),
    }


class HagenPoiseuillePipeFlowSim(CaseSimulator):
    """Generate analytical Hagen-Poiseuille radial velocity profiles."""

    name = "hagen_poiseuille_pipe_flow"

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
    ) -> None:
        self._parameter_space = (
            {} if parameter_space is None else dict(parameter_space)
        )

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    @staticmethod
    def _positive(params: Mapping[str, Any], name: str) -> float:
        value = float(params[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
        return value

    def make_case(self, params: Mapping[str, Any]) -> CaseParameters:
        radius = self._positive(params, "radius")
        length = self._positive(params, "length")
        dynamic_viscosity = self._positive(params, "dynamic_viscosity")
        density = self._positive(params, "density")
        pressure_drop = self._positive(params, "pressure_drop")

        n_radial_points_value = params["n_radial_points"]
        n_radial_points = int(n_radial_points_value)
        if n_radial_points != n_radial_points_value or n_radial_points < 2:
            raise ValueError("n_radial_points must be an integer of at least 2.")

        return CaseParameters(
            boundary_conditions={
                "centerline/velocity_gradient": 0.0,
                "wall/velocity": 0.0,
            },
            geometry={
                "radius": radius,
                "length": length,
            },
            controls={
                "pressure_drop": pressure_drop,
            },
            physical_parameters={
                "dynamic_viscosity": dynamic_viscosity,
                "density": density,
            },
            solver_parameters={
                "n_radial_points": n_radial_points,
            },
        )

    def run_case(self, case: CaseParameters) -> SimulationRecord:
        radius = float(case.geometry["radius"])
        length = float(case.geometry["length"])
        pressure_drop = float(case.controls["pressure_drop"])
        dynamic_viscosity = float(
            case.physical_parameters["dynamic_viscosity"]
        )
        density = float(case.physical_parameters["density"])
        n_radial_points = int(case.solver_parameters["n_radial_points"])

        pressure_gradient = -pressure_drop / length

        tic = time.perf_counter()
        radial_coordinate = torch.linspace(
            0.0,
            radius,
            n_radial_points,
            dtype=torch.float64,
        )
        velocity = hagen_poiseuille_velocity(
            radial_coordinate,
            radius=radius,
            dynamic_viscosity=dynamic_viscosity,
            pressure_gradient=pressure_gradient,
        )
        toc = time.perf_counter()

        diagnostics = _flow_diagnostics(
            radius=radius,
            length=length,
            dynamic_viscosity=dynamic_viscosity,
            density=density,
            pressure_drop=pressure_drop,
        )

        if diagnostics["reynolds_number"] >= LAMINAR_REYNOLDS_LIMIT:
            raise ValueError(
                "Hagen-Poiseuille flow must remain laminar: "
                f"Re={diagnostics['reynolds_number']:.6g} is not below "
                f"{LAMINAR_REYNOLDS_LIMIT:.0f}."
            )

        return SimulationRecord(
            coordinates={
                "r": radial_coordinate.detach().cpu().numpy(),
            },
            fields={
                "velocity": velocity.detach().cpu().numpy(),
            },
            constants=(
                case.geometry
                | case.controls
                | case.physical_parameters
                | {"pressure_gradient": pressure_gradient}
            ),
            metadata={
                "wall_time": toc - tic,
                "equation": "Hagen-Poiseuille",
                **diagnostics,
                "solver_parameters": deepcopy(case.solver_parameters),
                "units": {
                    "r": "m",
                    "velocity": "m/s",
                    "radius": "m",
                    "length": "m",
                    "dynamic_viscosity": "Pa s",
                    "density": "kg/m^3",
                    "pressure_drop": "Pa",
                    "pressure_gradient": "Pa/m",
                    "flow_rate": "m^3/s",
                },
            },
        )
