from pathlib import Path
import os
import time
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import cantera as ct
import numpy as np
from scikits.odes import dae

from chem_operator.utils import get_mechanism_file, datasets_path
from chem_operator.datasets import (
    CaseParameters,
    CaseSimulator,
    SimulationRecord
)
from chem_operator.sampling import ParameterSpec

DEFAULT_MECHANISM_FILE_NAME = "ammonia-Ru-Ba-YSZ-CSM-2019.yaml"
DEFAULT_SURFACE_PHASE_NAME = "Ru_surface"

def _inlet_composition(
    nh3_mole_fraction: float,
    diluent_species: str = "AR",
) -> dict[str, float]:
    return {
        "NH3": nh3_mole_fraction,
        diluent_species: 1.0 - nh3_mole_fraction,
    }


def make_tutorial_case() -> CaseParameters:
    """Direct CaseParameters version of the Cantera packed-bed tutorial."""
    return 


class _PackedBedModel:
    def __init__(
        self,
        case: CaseParameters,
        mechanism_file_name: str,
        surface_phase_name: str,
    ):
        self.case = case
        self.mechanism_file_name = mechanism_file_name
        self.surface_phase_name = surface_phase_name

        mechfile = get_mechanism_file(mechanism_file_name)
        self.surf = ct.Interface(mechfile, surface_phase_name)
        self.gas = self.surf.adjacent["gas"]

        self.n_gas = self.gas.n_species
        self.n_surf = self.surf.n_species
        self.n_gas_reactions = self.gas.n_reactions

        self.offset_rhou = 0
        self.offset_p = 1
        self.offset_T = 2
        self.offset_Y = 3
        self.offset_Z = self.offset_Y + self.n_gas
        self.n_var = self.offset_Z + self.n_surf

        self.length = case.geometry["length"]
        radius = case.geometry["radius"]
        porosity = case.geometry["porosity"]
        tortuosity = case.geometry["tortuosity"]
        particle_diameter = case.geometry["particle_diameter"]

        vol_ratio = porosity / (1 - porosity)
        eff_factor = porosity / tortuosity
        self.B_g = vol_ratio**2 * particle_diameter**2 * eff_factor / 72
        self.area2vol = 2 / radius
        self.D_h = 2 * radius

        self.W_g = self.gas.molecular_weights
        self.As = case.geometry["specific_surface_area"]
        self.phi = porosity
        self.solve_energy = case.controls["solve_energy"]
        self.T_wall = case.controls["wall_temperature"]
        self.h_coeff = case.controls["heat_transfer_coefficient"]
        self.radius = radius

        self.membrane_present = case.controls["membrane_present"]
        self.membrane_sp_name = case.controls["membrane_species"]
        self.membrane_sp_ind = self.gas.species_index(self.membrane_sp_name)
        self.p_sweep = case.controls["sweep_pressure"]
        self.permeance = (
            case.controls["membrane_permeability"]
            / case.controls["membrane_thickness"]
        )

    def initial_state(self) -> tuple[np.ndarray, np.ndarray]:
        T0 = self.case.initial_conditions["gas/T"]
        P0 = self.case.initial_conditions["gas/P"]
        X0 = self.case.initial_conditions["gas/X"]
        inlet_velocity = self.case.controls["inlet_velocity"]

        self.gas.TPX = T0, P0, X0
        self.surf.TP = T0, P0

        Yk_0 = self.gas.Y
        rhou0 = self.gas.density * inlet_velocity

        self.surf.advance_coverages_to_steady_state()
        Zk_0 = self.surf.coverages

        y0 = np.hstack((rhou0, P0, T0, Yk_0, Zk_0))
        yprime0 = np.zeros(self.n_var)
        res = np.zeros(self.n_var)
        self.residual(0, y0, yprime0, res)

        return y0, -res

    def residual(self, z, y, yPrime, res):
        rhou = y[self.offset_rhou]
        p = y[self.offset_p]
        T = y[self.offset_T]
        Y = y[self.offset_Y : self.offset_Y + self.n_gas]
        Z = y[self.offset_Z : self.offset_Z + self.n_surf]

        drhoudz = yPrime[self.offset_rhou]
        dpdz = yPrime[self.offset_p]
        dTdz = yPrime[self.offset_T]
        dYdz = yPrime[self.offset_Y : self.offset_Y + self.n_gas]

        self.gas.set_unnormalized_mass_fractions(Y)
        self.gas.TP = T, p
        self.surf.set_unnormalized_coverages(Z)
        self.surf.TP = T, p

        coverages = self.surf.coverages
        sdot_g = self.surf.get_net_production_rates("gas")
        sdot_s = self.surf.get_net_production_rates(self.surface_phase_name)
        wdot_g = np.zeros(self.n_gas)
        cp = self.gas.cp_mass
        hk_g = self.gas.partial_molar_enthalpies

        if self.n_gas_reactions > 0:
            wdot_g = self.gas.net_production_rates

        mu = self.gas.viscosity
        rho = self.gas.density

        memsp_pres = p * self.gas.X[self.membrane_sp_ind]
        membrane_flux = (
            -self.permeance
            * (memsp_pres - self.p_sweep)
            * self.W_g[self.membrane_sp_ind]
        )

        sum_continuity = self.As * np.sum(sdot_g * self.W_g) + self.phi * np.sum(
            wdot_g * self.W_g
        )
        res[self.offset_rhou] = (
            drhoudz
            - sum_continuity
            - self.area2vol * membrane_flux * self.membrane_present
        )

        res[self.offset_Y : self.offset_Y + self.n_gas] = (
            dYdz
            + (
                Y * sum_continuity
                - self.phi * np.multiply(wdot_g, self.W_g)
                - self.As * np.multiply(sdot_g, self.W_g)
            )
            / rhou
        )
        res[self.offset_Y + self.membrane_sp_ind] -= (
            self.area2vol * membrane_flux * self.membrane_present
        )

        res[self.offset_Z : self.offset_Z + self.n_surf] = sdot_s
        ind_large = np.argmax(coverages)
        res[self.offset_Z + ind_large] = 1 - np.sum(coverages)

        u = rhou / rho
        res[self.offset_p] = dpdz + self.phi * mu * u / self.B_g

        res[self.offset_T] = dTdz
        if self.solve_energy:
            conv_term = (
                (4 / self.D_h)
                * self.h_coeff
                * (self.T_wall - T)
                * (2 * np.pi * self.radius)
            )
            chem_term = np.sum(hk_g * (self.phi * wdot_g + self.As * sdot_g))
            res[self.offset_T] -= (conv_term - chem_term) / (rhou * cp)

    def solve(self) -> tuple[np.ndarray, np.ndarray]:
        y0, yprime0 = self.initial_state()

        solver = dae(
            "ida",
            self.residual,
            first_step_size=self.case.solver_parameters["first_step_size"],
            atol=self.case.solver_parameters["atol"],
            rtol=self.case.solver_parameters["rtol"],
            algebraic_vars_idx=list(range(self.offset_Z, self.offset_Z + self.n_surf)),
            max_steps=self.case.solver_parameters["max_steps"],
            one_step_compute=True,
            old_api=False,
        )

        distance = []
        solution = []
        state = solver.init_step(0.0, y0, yprime0)

        while state.values.t < self.length:
            distance.append(state.values.t)
            solution.append(state.values.y.copy())
            state = solver.step(self.length)

        distance.append(state.values.t)
        solution.append(state.values.y.copy())

        return np.array(distance), np.array(solution)


class PackedBed1DSimulator(CaseSimulator):
    name = "packed_bed_1d"

    def __init__(
        self,
        parameter_space: Mapping[str, ParameterSpec] | None = None,
        mechanism_file_name: str = DEFAULT_MECHANISM_FILE_NAME,
        surface_phase_name: str = DEFAULT_SURFACE_PHASE_NAME,
    ):
        if parameter_space is None:
            self._parameter_space = {}
        else:
            self._parameter_space = dict(parameter_space)

        self.mechanism_file_name = mechanism_file_name
        self.surface_phase_name = surface_phase_name

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    def make_case(
        self,
        params: Mapping[str, Any],
    ) -> CaseParameters:
        return CaseParameters(
            initial_conditions={
                "gas/T": params["T0"],
                "gas/P": params["P0"],
                "gas/X": _inlet_composition(
                    params["inlet_nh3_mole_fraction"],
                    params["diluent_species"],
                ),
            },
            geometry={
                "length": params["length"],
                "radius": params["radius"],
                "porosity": params["porosity"],
                "tortuosity": params["tortuosity"],
                "particle_diameter": params["particle_diameter"],
                "specific_surface_area": params["specific_surface_area"],
            },
            controls={
                "inlet_velocity": params["inlet_velocity"],
                "wall_temperature": params["wall_temperature"],
                "heat_transfer_coefficient": params["heat_transfer_coefficient"],
                "solve_energy": params["solve_energy"],
                "membrane_present": params["membrane_present"],
                "membrane_permeability": params["membrane_permeability"],
                "membrane_thickness": params["membrane_thickness"],
                "membrane_species": params["membrane_species"],
                "sweep_pressure": params["sweep_pressure"],
            },
            solver_parameters={
                "first_step_size": params["first_step_size"],
                "atol": params["atol"],
                "rtol": params["rtol"],
                "max_steps": params["max_steps"],
            },
            mechanism_parameters={
                "inlet_nh3_mole_fraction": params["inlet_nh3_mole_fraction"],
                "diluent_species": params["diluent_species"],
            },
        )

    def run_case(
        self,
        case: CaseParameters,
    ) -> SimulationRecord:
        model = _PackedBedModel(
            case,
            mechanism_file_name=self.mechanism_file_name,
            surface_phase_name=self.surface_phase_name,
        )

        tic = time.perf_counter()
        z, solution = model.solve()
        toc = time.perf_counter()

        rhou = solution[:, model.offset_rhou]
        P = solution[:, model.offset_p]
        T = solution[:, model.offset_T]
        Y = solution[:, model.offset_Y : model.offset_Y + model.n_gas]
        Z = solution[:, model.offset_Z : model.offset_Z + model.n_surf]

        X = np.zeros_like(Y)
        velocity = np.zeros_like(z)
        for i in range(len(z)):
            model.gas.TPY = T[i], P[i], Y[i]
            X[i] = model.gas.X
            velocity[i] = rhou[i] / model.gas.density

        return SimulationRecord(
            coordinates={"z": z},
            fields={
                "rhou": rhou,
                "P": P,
                "T": T,
                "Y": Y,
                "X": X,
                "Z": Z,
                "velocity": velocity,
            },
            constants=(
                case.geometry
                | case.controls
                | case.mechanism_parameters
            ),
            metadata={
                "wall_time": toc - tic,
                "mechanism": self.mechanism_file_name,
                "surface_phase": self.surface_phase_name,
                "gas_species": model.gas.species_names,
                "surface_species": model.surf.species_names,
                "solver_parameters": deepcopy(case.solver_parameters),
            },
        )