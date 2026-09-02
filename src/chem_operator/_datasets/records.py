"""Simulation records and simulator protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import cantera as ct
import numpy as np

from chem_operator.sampling import ParameterSpec
from chem_operator.utils import get_mechanism_file


@dataclass
class CaseParameters:
    initial_conditions: dict | None = None
    boundary_conditions: dict | None = None
    controls: dict | None = None
    geometry: dict | None = None
    physical_parameters: dict | None = None
    solver_parameters: dict | None = None
    mechanism_parameters: dict | None = None

@dataclass
class SimulationRecord:
    # Examples: time, z, x/y grid
    coordinates: dict[str, np.ndarray]

    # Arrays sharing one or more coordinate dimensions
    fields: dict[str, np.ndarray]
    
    # Values constant over the record
    constants: dict[str, float | np.ndarray]

    # Names, units, phase information, provenance, ...
    metadata: dict

    def to_SolutionArray(self) -> ct.SolutionArray:
        mechanism = self.metadata.get("mechanism")
        if mechanism is None:
            raise ValueError("metadata must contain 'mechanism'.")

        gas = ct.Solution(get_mechanism_file(mechanism))

        times = self.coordinates.get("t")
        if times is None:
            raise ValueError("coordinates must contain time 't'.")

        T = self.fields.get("T")
        P = self.fields.get("P")
        X = self.fields.get("X")
        Y = self.fields.get("Y")

        if T is None or P is None:
            raise ValueError("fields must contain 'T' and 'P'.")

        if X is None and Y is None:
            raise ValueError("fields must contain either 'X' or 'Y'.")

        states = ct.SolutionArray(gas, extra=["t"])

        for i, t in enumerate(times):
            if X is not None:
                gas.TPX = T[i], P[i], X[i]
            else:
                gas.TPY = T[i], P[i], Y[i]

            states.append(gas.state, t=t)

        return states

class CaseSimulator(Protocol):
    name: str

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec] | None:
        ...

    def make_case(
        self,
        params: Mapping[str, Any],
    ) -> CaseParameters:
        ...

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        ...
