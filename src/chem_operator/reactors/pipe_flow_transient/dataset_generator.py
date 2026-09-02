"""Analytical dataset generation for startup flow in a circular pipe."""

from __future__ import annotations

import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from physicsnemo.sym.eq.pde import PDE
from scipy.special import j0, j1, jn_zeros  # pylint: disable=no-name-in-module
from sympy import Function, Number, Symbol

from chem_operator.datasets import (
    CaseParameters,
    CaseSimulator,
    SimulationDatasetGenerator,
    SimulationRecord,
)
from chem_operator.sampling import Constant, ParameterSpec, Uniform


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LAMINAR_REYNOLDS_LIMIT = 2300.0
DEFAULT_BESSEL_SERIES_TERMS = 128
STARTUP_REFERENCE = (
    "https://en.wikipedia.org/wiki/"
    "Hagen%E2%80%93Poiseuille_equation#Startup_of_Poiseuille_flow_in_a_pipe"
)


class TransientHagenPoiseuille(PDE):
    """Nondimensional startup-flow equation for PhysicsInformer.

    With ``x = r / R``, ``t = nu * time / R**2``, and velocity normalized by
    its steady centerline value, the equation is

    ``x * velocity__t - x * velocity__x__x - velocity__x - 4 * x = 0``.

    Multiplication by ``x`` removes the cylindrical ``1 / x`` singularity.
    PhysicsInformer computes the spatial derivatives; callers provide
    ``velocity__t`` explicitly.
    """

    def __init__(self) -> None:
        self.dim = 1
        x = Symbol("x")
        t = Symbol("t")
        velocity = Function("velocity")(x, t)  # pylint: disable=not-callable
        self.equations = {
            "momentum": (
                x * velocity.diff(t)
                - x * velocity.diff(x, 2)
                - velocity.diff(x)
                - Number(4.0) * x
            )
        }


def startup_hagen_poiseuille_dimensionless_velocity(
    fourier_number: np.ndarray,
    radius_fraction: np.ndarray,
    *,
    n_terms: int = DEFAULT_BESSEL_SERIES_TERMS,
) -> np.ndarray:
    """Evaluate the exact Fourier--Bessel startup velocity solution.

    The returned velocity is normalized by the steady centerline velocity.
    ``fourier_number`` is ``nu * time / R**2`` and ``radius_fraction`` is
    ``r / R``. The infinite analytical series is truncated to ``n_terms`` for
    numerical evaluation.
    """
    fourier_number = np.asarray(fourier_number, dtype=np.float64)
    radius_fraction = np.asarray(radius_fraction, dtype=np.float64)

    if fourier_number.ndim != 1 or radius_fraction.ndim != 1:
        raise ValueError(
            "fourier_number and radius_fraction must be one-dimensional."
        )
    if not np.all(np.isfinite(fourier_number)) or np.any(fourier_number < 0.0):
        raise ValueError("fourier_number must contain finite, nonnegative values.")
    if (
        not np.all(np.isfinite(radius_fraction))
        or np.any(radius_fraction < 0.0)
        or np.any(radius_fraction > 1.0)
    ):
        raise ValueError("radius_fraction must lie between zero and one.")
    if int(n_terms) != n_terms or n_terms < 1:
        raise ValueError("n_terms must be a positive integer.")

    roots = jn_zeros(0, int(n_terms))
    radial_basis = j0(roots[:, None] * radius_fraction[None, :])
    weights = 1.0 / (roots**3 * j1(roots))
    decay = np.exp(-fourier_number[:, None] * roots[None, :] ** 2)
    velocity = (
        (1.0 - radius_fraction**2)[None, :]
        - 8.0 * np.einsum("tn,n,nr->tr", decay, weights, radial_basis)
    )

    # These conditions are exact but converge only in the infinite-series
    # limit. Set them explicitly to avoid finite-series roundoff at the edges.
    velocity[np.isclose(fourier_number, 0.0, rtol=0.0, atol=0.0), :] = 0.0
    velocity[
        :,
        np.isclose(radius_fraction, 1.0, rtol=0.0, atol=1.0e-14),
    ] = 0.0
    return velocity


def startup_hagen_poiseuille_dimensionless_flow_rate(
    fourier_number: np.ndarray,
    *,
    n_terms: int = DEFAULT_BESSEL_SERIES_TERMS,
) -> np.ndarray:
    """Evaluate the exact startup flow rate normalized by its steady value."""
    fourier_number = np.asarray(fourier_number, dtype=np.float64)
    if fourier_number.ndim != 1:
        raise ValueError("fourier_number must be one-dimensional.")
    if not np.all(np.isfinite(fourier_number)) or np.any(fourier_number < 0.0):
        raise ValueError("fourier_number must contain finite, nonnegative values.")
    if int(n_terms) != n_terms or n_terms < 1:
        raise ValueError("n_terms must be a positive integer.")

    roots = jn_zeros(0, int(n_terms))
    flow_rate = 1.0 - 32.0 * np.sum(
        np.exp(-fourier_number[:, None] * roots[None, :] ** 2)
        / roots[None, :] ** 4,
        axis=1,
    )
    flow_rate[
        np.isclose(fourier_number, 0.0, rtol=0.0, atol=0.0)
    ] = 0.0
    return flow_rate


def _steady_diagnostics(
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
        "steady_flow_rate": float(flow_rate),
        "steady_mean_velocity": float(mean_velocity),
        "steady_max_velocity": float(2.0 * mean_velocity),
        "steady_reynolds_number": float(
            density * mean_velocity * (2.0 * radius) / dynamic_viscosity
        ),
    }


class TransientHagenPoiseuillePipeFlowSim(CaseSimulator):
    """Generate exact analytical startup-flow trajectories."""

    name = "transient_hagen_poiseuille_pipe_flow"

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        *,
        series_terms: int = DEFAULT_BESSEL_SERIES_TERMS,
    ) -> None:
        if int(series_terms) != series_terms or series_terms < 1:
            raise ValueError("series_terms must be a positive integer.")
        self._parameter_space = (
            {} if parameter_space is None else dict(parameter_space)
        )
        self.series_terms = int(series_terms)
        self._profile_cache: dict[tuple[int, int, float], np.ndarray] = {}

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    @staticmethod
    def _positive(params: Mapping[str, Any], name: str) -> float:
        value = float(params[name])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")
        return value

    @staticmethod
    def _positive_integer(params: Mapping[str, Any], name: str) -> int:
        raw_value = params[name]
        value = int(raw_value)
        if value != raw_value or value < 2:
            raise ValueError(f"{name} must be an integer of at least 2.")
        return value

    def make_case(self, params: Mapping[str, Any]) -> CaseParameters:
        return CaseParameters(
            initial_conditions={"velocity": 0.0},
            boundary_conditions={
                "centerline/velocity_gradient": 0.0,
                "wall/velocity": 0.0,
            },
            geometry={
                "radius": self._positive(params, "radius"),
                "length": self._positive(params, "length"),
            },
            controls={
                "pressure_drop": self._positive(params, "pressure_drop"),
            },
            physical_parameters={
                "dynamic_viscosity": self._positive(
                    params, "dynamic_viscosity"
                ),
                "density": self._positive(params, "density"),
            },
            solver_parameters={
                "n_time_points": self._positive_integer(
                    params, "n_time_points"
                ),
                "n_radial_points": self._positive_integer(
                    params, "n_radial_points"
                ),
                "max_fourier_number": self._positive(
                    params, "max_fourier_number"
                ),
            },
        )

    def dimensionless_velocity(
        self,
        *,
        n_time_points: int,
        n_radial_points: int,
        max_fourier_number: float,
    ) -> np.ndarray:
        """Return the analytical solution on a regular dimensionless grid."""
        key = (n_time_points, n_radial_points, float(max_fourier_number))
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached.copy()

        radius_fraction = np.linspace(0.0, 1.0, n_radial_points)
        fourier_number = np.linspace(
            0.0,
            max_fourier_number,
            n_time_points,
        )
        velocity = startup_hagen_poiseuille_dimensionless_velocity(
            fourier_number,
            radius_fraction,
            n_terms=self.series_terms,
        )
        self._profile_cache[key] = velocity
        return velocity.copy()

    def run_case(  # pylint: disable=too-many-locals
        self, case: CaseParameters
    ) -> SimulationRecord:
        radius = float(case.geometry["radius"])
        length = float(case.geometry["length"])
        pressure_drop = float(case.controls["pressure_drop"])
        dynamic_viscosity = float(
            case.physical_parameters["dynamic_viscosity"]
        )
        density = float(case.physical_parameters["density"])
        n_time_points = int(case.solver_parameters["n_time_points"])
        n_radial_points = int(case.solver_parameters["n_radial_points"])
        max_fourier_number = float(
            case.solver_parameters["max_fourier_number"]
        )

        diagnostics = _steady_diagnostics(
            radius=radius,
            length=length,
            dynamic_viscosity=dynamic_viscosity,
            density=density,
            pressure_drop=pressure_drop,
        )
        if diagnostics["steady_reynolds_number"] >= LAMINAR_REYNOLDS_LIMIT:
            raise ValueError(
                "Hagen-Poiseuille flow must remain laminar: "
                f"Re={diagnostics['steady_reynolds_number']:.6g} is not below "
                f"{LAMINAR_REYNOLDS_LIMIT:.0f}."
            )

        kinematic_viscosity = dynamic_viscosity / density
        viscous_time_scale = radius**2 / kinematic_viscosity
        pressure_gradient = -pressure_drop / length

        tic = time.perf_counter()
        normalized_velocity = self.dimensionless_velocity(
            n_time_points=n_time_points,
            n_radial_points=n_radial_points,
            max_fourier_number=max_fourier_number,
        )
        fourier_number = np.linspace(
            0.0,
            max_fourier_number,
            n_time_points,
        )
        radial_coordinate = np.linspace(0.0, radius, n_radial_points)
        times = fourier_number * viscous_time_scale
        velocity = normalized_velocity * diagnostics["steady_max_velocity"]
        normalized_flow_rate = (
            startup_hagen_poiseuille_dimensionless_flow_rate(
                fourier_number,
                n_terms=self.series_terms,
            )
        )
        flow_rate = normalized_flow_rate * diagnostics["steady_flow_rate"]
        toc = time.perf_counter()

        return SimulationRecord(
            coordinates={
                "t": times,
                "r": radial_coordinate,
            },
            fields={
                "velocity": velocity,
                "flow_rate": flow_rate,
            },
            constants=(
                case.geometry
                | case.controls
                | case.physical_parameters
                | {
                    "pressure_gradient": pressure_gradient,
                    "kinematic_viscosity": kinematic_viscosity,
                    "viscous_time_scale": viscous_time_scale,
                }
            ),
            metadata={
                "wall_time": toc - tic,
                "equation": "transient Hagen-Poiseuille startup flow",
                "solver": "analytical Fourier-Bessel series",
                "reference": STARTUP_REFERENCE,
                **diagnostics,
                "analytical_solution": {
                    "series_terms": self.series_terms,
                },
                "physicsnemo_pde": TransientHagenPoiseuille.__name__,
                "solver_parameters": deepcopy(case.solver_parameters),
                "units": {
                    "t": "s",
                    "r": "m",
                    "velocity": "m/s",
                    "flow_rate": "m^3/s",
                    "radius": "m",
                    "length": "m",
                    "dynamic_viscosity": "Pa s",
                    "kinematic_viscosity": "m^2/s",
                    "density": "kg/m^3",
                    "pressure_drop": "Pa",
                    "pressure_gradient": "Pa/m",
                    "viscous_time_scale": "s",
                },
            },
        )
