"""Coupled non-isothermal PFR and cylindrical-wall PhysicsNeMo PINN.

The gas uses the three-species parallel mechanism ``A -> B`` and
``2 A -> 3 C`` from the PINNSE non-isothermal PFR example:
https://github.com/hverma99/pinnse/tree/main/examples/nonisopfr

Three molar-flow balances and a gas energy balance are solved along ``z``.
Species concentrations follow from an ideal-gas algebraic closure.  The solid
wall conducts heat in ``(z, r)`` and is coupled to the gas through both the gas
energy balance and a Robin condition at the inner wall.

All coordinates and states are dimensionless.  PhysicsNeMo names Cartesian
symbols ``x`` and ``y`` internally; here they represent axial position ``z``
and cylindrical radius ``r``, respectively.

Run a quick smoke example with::

    python nemo-examples/pfr_cylindrical_heat_pinn.py --steps 50 --no-plot

The default run trains longer and writes a profile/temperature-field figure to
``nemo-examples/Figures``.
"""

# A single-file tutorial keeps the coupled loss terms together intentionally.
# pylint: disable=too-many-instance-attributes,too-many-locals
# Environment variables are set before plotting/PhysicsNeMo imports.
# pylint: disable=wrong-import-position,not-callable

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from sympy import Function, Number, Symbol, exp
import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional

from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.eq.phy_informer import PhysicsInformer


@dataclass(frozen=True)
class Parameters:
    """Reference chemistry, nondimensionalization, and wall parameters."""

    # Dimensional characteristic scales used by the reference PFR.
    flow_scale: float = 200.0  # mol/s
    concentration_scale: float = 0.1  # mol/L
    temperature_scale: float = 800.0  # K
    volume_scale: float = 1.0
    heat_capacity_scale: float = 200.0  # J/(mol K)

    # Nominal inlet and cooling-wall conditions.
    inlet_flow_a_dimensional: float = 100.0  # mol/s
    inlet_concentration_a_dimensional: float = 0.1  # mol/L
    inlet_temperature_dimensional: float = 393.15  # K
    outer_temperature_dimensional: float = 303.15  # K
    reference_temperature_dimensional: float = 300.0  # K

    # A -> B and 2 A -> 3 C Arrhenius/thermochemical data.
    reaction_1_pre_exponential: float = 10.0
    reaction_1_activation_temperature: float = 4000.0  # E/R, K
    reaction_1_heat: float = -20_000.0  # J/mol A reacted
    reaction_2_pre_exponential: float = 0.5
    reaction_2_activation_temperature: float = 9000.0  # E/R, K
    reaction_2_heat: float = -60_000.0  # J/mol A reacted
    heat_capacities: tuple[float, float, float] = (90.0, 90.0, 180.0)
    volumetric_heat_transfer: float = 4000.0

    # The wall is retained from the original conjugate-heat example.
    wall_aspect_ratio_sq: float = 0.20
    interface_biot: float = 1.5
    inner_radius: float = 1.0
    outer_radius: float = 2.0
    minimum_temperature: float = 0.25
    maximum_temperature: float = 0.75

    @property
    def inlet_flow_a(self) -> float:
        """Nondimensional inlet flow of A."""
        return self.inlet_flow_a_dimensional / self.flow_scale

    @property
    def inlet_concentration_a(self) -> float:
        """Nondimensional inlet concentration and total concentration."""
        return self.inlet_concentration_a_dimensional / self.concentration_scale

    @property
    def inlet_temperature(self) -> float:
        """Nondimensional gas inlet temperature."""
        return self.inlet_temperature_dimensional / self.temperature_scale

    @property
    def outer_temperature(self) -> float:
        """Nondimensional fixed temperature at the wall exterior."""
        return self.outer_temperature_dimensional / self.temperature_scale

    @property
    def reference_temperature(self) -> float:
        """Nondimensional Arrhenius reference temperature."""
        return self.reference_temperature_dimensional / self.temperature_scale

    @property
    def flow_rhs_scale(self) -> float:
        """Scale taking dimensional reaction rates to dF/d(z/L)."""
        return self.volume_scale / self.flow_scale

    @property
    def wall_energy_scale(self) -> float:
        """Nondimensional gas/solid heat-transfer coefficient."""
        return (
            self.volumetric_heat_transfer
            * self.volume_scale
            / (self.flow_scale * self.heat_capacity_scale)
        )

    def reaction_energy_scale(self, heat_of_reaction: float) -> float:
        """Return the positive temperature source scale for an exothermic rate."""
        return (
            -heat_of_reaction
            * self.volume_scale
            / (
                self.flow_scale
                * self.temperature_scale
                * self.heat_capacity_scale
            )
        )


class PlugFlowReactor(PDE):
    """Three-species parallel-reaction PFR coupled to the wall temperature."""

    def __init__(self, parameters: Parameters) -> None:
        self.dim = 1
        z = Symbol("x")
        f_a = Function("f_a")(z)
        f_b = Function("f_b")(z)
        f_c = Function("f_c")(z)
        t_gas = Function("t_gas")(z)
        t_wall = Function("t_wall")(z)

        total_flow = f_a + f_b + f_c
        c_a = (
            Number(parameters.inlet_concentration_a)
            * f_a
            / total_flow
            * Number(parameters.inlet_temperature)
            / t_gas
        )
        arrhenius_1 = Number(parameters.reaction_1_pre_exponential) * exp(
            Number(
                parameters.reaction_1_activation_temperature
                / parameters.temperature_scale
            )
            * (Number(1.0 / parameters.reference_temperature) - 1 / t_gas)
        )
        arrhenius_2 = Number(parameters.reaction_2_pre_exponential) * exp(
            Number(
                parameters.reaction_2_activation_temperature
                / parameters.temperature_scale
            )
            * (Number(1.0 / parameters.reference_temperature) - 1 / t_gas)
        )
        progress_1 = arrhenius_1 * (
            c_a * Number(parameters.concentration_scale)
        )
        progress_2 = arrhenius_2 * (
            c_a * Number(parameters.concentration_scale)
        ) ** 2

        flow_scale = Number(parameters.flow_rhs_scale)
        heat_capacity_flow = (
            f_a * Number(parameters.heat_capacities[0] / parameters.heat_capacity_scale)
            + f_b
            * Number(parameters.heat_capacities[1] / parameters.heat_capacity_scale)
            + f_c
            * Number(parameters.heat_capacities[2] / parameters.heat_capacity_scale)
        )
        wall_heat = Number(parameters.wall_energy_scale) * (t_wall - t_gas)
        reaction_heat = (
            Number(parameters.reaction_energy_scale(parameters.reaction_1_heat))
            * progress_1
            + Number(parameters.reaction_energy_scale(parameters.reaction_2_heat))
            * progress_2
        )

        self.equations = {
            "flow_a": f_a.diff(z) + flow_scale * (progress_1 + progress_2),
            "flow_b": f_b.diff(z) - flow_scale * progress_1,
            "flow_c": f_c.diff(z) - Number(1.5) * flow_scale * progress_2,
            "gas_energy": (
                t_gas.diff(z) - (wall_heat + reaction_heat) / heat_capacity_flow
            ),
        }


class CylindricalWall(PDE):
    """Steady axisymmetric wall conduction and gas/solid interface flux."""

    def __init__(self, parameters: Parameters) -> None:
        self.dim = 2
        z, r = Symbol("x"), Symbol("y")
        t_solid = Function("t_solid")(z, r)
        t_gas = Function("t_gas")(z, r)

        self.equations = {
            "solid_heat": (
                Number(parameters.wall_aspect_ratio_sq) * t_solid.diff(z, 2)
                + t_solid.diff(r, 2)
                + t_solid.diff(r) / r
            ),
            "interface": (
                t_solid.diff(r)
                - Number(parameters.interface_biot) * (t_solid - t_gas)
            ),
            "axial_flux": t_solid.diff(z),
        }


def bounded_temperature(raw: Tensor, parameters: Parameters) -> Tensor:
    """Keep nondimensional temperatures in a stable physical interval."""
    span = parameters.maximum_temperature - parameters.minimum_temperature
    return parameters.minimum_temperature + span * torch.sigmoid(raw)


def gas_state(
    raw: Tensor,
    parameters: Parameters,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return positive nondimensional molar flows and gas temperature."""
    f_a = torch_functional.softplus(raw[:, 0:1])
    f_b = torch_functional.softplus(raw[:, 1:2])
    f_c = torch_functional.softplus(raw[:, 2:3])
    t_gas = bounded_temperature(raw[:, 3:4], parameters)
    return f_a, f_b, f_c, t_gas


def concentrations_from_state(
    f_a: Tensor,
    f_b: Tensor,
    f_c: Tensor,
    t_gas: Tensor,
    parameters: Parameters,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute ideal-gas concentrations from flows and gas temperature."""
    total_flow = (f_a + f_b + f_c).clamp_min(1.0e-8)
    concentration_factor = (
        parameters.inlet_concentration_a
        * parameters.inlet_temperature
        / t_gas.clamp_min(1.0e-8)
    )
    return tuple(
        concentration_factor * flow / total_flow for flow in (f_a, f_b, f_c)
    )


def reaction_progress_rates(
    c_a: Tensor,
    t_gas: Tensor,
    parameters: Parameters,
) -> tuple[Tensor, Tensor]:
    """Evaluate the two reference Arrhenius progress rates in Torch."""
    inverse_temperature_difference = (
        1.0 / parameters.reference_temperature
        - 1.0 / t_gas.clamp_min(1.0e-8)
    )
    exponent_1 = (
        parameters.reaction_1_activation_temperature
        / parameters.temperature_scale
        * inverse_temperature_difference
    ).clamp(-60.0, 60.0)
    exponent_2 = (
        parameters.reaction_2_activation_temperature
        / parameters.temperature_scale
        * inverse_temperature_difference
    ).clamp(-60.0, 60.0)
    dimensional_c_a = c_a.clamp_min(0.0) * parameters.concentration_scale
    progress_1 = (
        parameters.reaction_1_pre_exponential
        * torch.exp(exponent_1)
        * dimensional_c_a
    )
    progress_2 = (
        parameters.reaction_2_pre_exponential
        * torch.exp(exponent_2)
        * dimensional_c_a.square()
    )
    return progress_1, progress_2


def solid_temperature(raw: Tensor, parameters: Parameters) -> Tensor:
    """Map the solid-network output to the same bounded temperature interval."""
    return bounded_temperature(raw, parameters)


def make_informers(
    parameters: Parameters,
    device: torch.device,
) -> tuple[PhysicsInformer, PhysicsInformer, PhysicsInformer, PhysicsInformer]:
    """Create residual evaluators for the reactor, wall, and wall boundaries."""
    reactor = PlugFlowReactor(parameters)
    wall = CylindricalWall(parameters)
    reactor_informer = PhysicsInformer(
        ["flow_a", "flow_b", "flow_c", "gas_energy"],
        reactor,
        "autodiff",
        device=device,
    )
    wall_informer = PhysicsInformer(
        ["solid_heat"], wall, "autodiff", device=device
    )
    interface_informer = PhysicsInformer(
        ["interface"], wall, "autodiff", device=device
    )
    end_informer = PhysicsInformer(
        ["axial_flux"], wall, "autodiff", device=device
    )
    return reactor_informer, wall_informer, interface_informer, end_informer


def squared_mean(values: Tensor) -> Tensor:
    """Return a scalar mean-square residual."""
    return values.square().mean()


def train(
    parameters: Parameters,
    *,
    steps: int,
    points: int,
    device: torch.device,
    report_every: int = 250,
) -> tuple[nn.Module, nn.Module]:
    """Train the coupled gas and solid PINNs."""
    gas_net = FullyConnected(
        in_features=1,
        out_features=4,
        num_layers=4,
        layer_size=64,
        activation_fn="tanh",
    ).to(device)
    solid_net = FullyConnected(
        in_features=2,
        out_features=1,
        num_layers=4,
        layer_size=64,
        activation_fn="tanh",
    ).to(device)
    informers = make_informers(parameters, device)
    reactor_informer, wall_informer, interface_informer, end_informer = informers
    optimizer = torch.optim.Adam(
        [*gas_net.parameters(), *solid_net.parameters()], lr=1.0e-3
    )

    for step in range(1, steps + 1):
        # PFR residuals.  The solid value at r=Ri supplies the local wall field.
        z = torch.rand(points, 1, device=device, requires_grad=True)
        f_a, f_b, f_c, t_gas = gas_state(gas_net(z), parameters)
        inner_coordinates = torch.cat(
            [z, torch.full_like(z, parameters.inner_radius)], dim=1
        )
        t_wall = solid_temperature(solid_net(inner_coordinates), parameters)
        reactor_residuals = reactor_informer.forward(
            {
                "coordinates": z,
                "f_a": f_a,
                "f_b": f_b,
                "f_c": f_c,
                "t_gas": t_gas,
                "t_wall": t_wall,
            }
        )
        reactor_loss = sum(squared_mean(value) for value in reactor_residuals.values())

        # Interior cylindrical conduction in (z, r).
        wall_coordinates = torch.rand(points, 2, device=device)
        wall_coordinates[:, 1] = (
            parameters.inner_radius
            + (parameters.outer_radius - parameters.inner_radius)
            * wall_coordinates[:, 1]
        )
        wall_coordinates.requires_grad_()
        t_solid = solid_temperature(solid_net(wall_coordinates), parameters)
        wall_residual = wall_informer.forward(
            {
                "coordinates": wall_coordinates,
                "y": wall_coordinates[:, 1:2],
                "t_solid": t_solid,
            }
        )["solid_heat"]

        # Inner Robin condition: conduction into the gas equals convection.
        z_interface = torch.rand(points, 1, device=device, requires_grad=True)
        interface_coordinates = torch.cat(
            [
                z_interface,
                torch.full_like(z_interface, parameters.inner_radius),
            ],
            dim=1,
        )
        interface_temperature = solid_temperature(
            solid_net(interface_coordinates), parameters
        )
        _, _, _, interface_gas_temperature = gas_state(
            gas_net(z_interface), parameters
        )
        interface_residual = interface_informer.forward(
            {
                "coordinates": interface_coordinates,
                "t_solid": interface_temperature,
                "t_gas": interface_gas_temperature,
            }
        )["interface"]

        # Fixed cooling-wall temperature at the outer radius.
        z_outer = torch.rand(points, 1, device=device)
        outer_coordinates = torch.cat(
            [z_outer, torch.full_like(z_outer, parameters.outer_radius)], dim=1
        )
        outer_temperature = solid_temperature(
            solid_net(outer_coordinates), parameters
        )
        outer_loss = squared_mean(outer_temperature - parameters.outer_temperature)

        # Adiabatic solid ends at z=0 and z=1.
        radius = parameters.inner_radius + (
            parameters.outer_radius - parameters.inner_radius
        ) * torch.rand(points, 1, device=device)
        end_coordinates = torch.cat(
            [
                torch.cat([torch.zeros_like(radius), radius], dim=1),
                torch.cat([torch.ones_like(radius), radius], dim=1),
            ],
            dim=0,
        ).requires_grad_()
        end_temperature = solid_temperature(solid_net(end_coordinates), parameters)
        end_residual = end_informer.forward(
            {"coordinates": end_coordinates, "t_solid": end_temperature}
        )["axial_flux"]

        # Feed conditions for A -> B and 2 A -> 3 C.
        inlet = torch.zeros(1, 1, device=device)
        inlet_f_a, inlet_f_b, inlet_f_c, inlet_temperature = gas_state(
            gas_net(inlet), parameters
        )
        inlet_loss = (
            squared_mean(inlet_f_a - parameters.inlet_flow_a)
            + squared_mean(inlet_f_b)
            + squared_mean(inlet_f_c)
            + squared_mean(inlet_temperature - parameters.inlet_temperature)
        )

        loss = (
            reactor_loss
            + squared_mean(wall_residual)
            + 5.0 * squared_mean(interface_residual)
            + 10.0 * outer_loss
            + 2.0 * squared_mean(end_residual)
            + 20.0 * inlet_loss
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % report_every == 0 or step == steps:
            print(
                f"step {step:5d} | loss {loss.item():.3e} | "
                f"reactor {reactor_loss.item():.3e} | "
                f"wall {squared_mean(wall_residual).item():.3e}"
            )

    return gas_net, solid_net


def save_solution(
    gas_net: nn.Module,
    solid_net: nn.Module,
    parameters: Parameters,
    output: Path,
    device: torch.device,
) -> Path:
    """Plot dimensional flow, concentration, and temperature solutions."""
    n_z, n_r = 201, 101
    z = torch.linspace(0.0, 1.0, n_z, device=device)[:, None]
    r = torch.linspace(
        parameters.inner_radius, parameters.outer_radius, n_r, device=device
    )[:, None]

    with torch.no_grad():
        f_a, f_b, f_c, t_gas = gas_state(gas_net(z), parameters)
        c_a, c_b, c_c = concentrations_from_state(
            f_a, f_b, f_c, t_gas, parameters
        )
        z_grid, r_grid = torch.meshgrid(z[:, 0], r[:, 0], indexing="ij")
        coordinates = torch.stack([z_grid, r_grid], dim=-1).reshape(-1, 2)
        t_solid = solid_temperature(
            solid_net(coordinates), parameters
        ).reshape(n_z, n_r)

    z_numpy = z[:, 0].cpu().numpy()
    r_numpy = r[:, 0].cpu().numpy()
    solid_kelvin = t_solid.cpu().numpy() * parameters.temperature_scale

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    flow_axis, concentration_axis, temperature_axis, field_axis = axes.flat

    for flow, label in ((f_a, "A"), (f_b, "B"), (f_c, "C")):
        flow_axis.plot(
            z_numpy,
            flow[:, 0].cpu() * parameters.flow_scale,
            label=rf"$F_{label}$",
        )
    flow_axis.set(xlabel="z/L", ylabel="molar flow [mol/s]", title="PFR flows")
    flow_axis.legend()

    for concentration, label in ((c_a, "A"), (c_b, "B"), (c_c, "C")):
        concentration_axis.plot(
            z_numpy,
            concentration[:, 0].cpu() * parameters.concentration_scale,
            label=rf"$C_{label}$",
        )
    concentration_axis.set(
        xlabel="z/L",
        ylabel="concentration [mol/L]",
        title="Ideal-gas concentrations",
    )
    concentration_axis.legend()

    temperature_axis.plot(
        z_numpy,
        t_gas[:, 0].cpu() * parameters.temperature_scale,
        label=r"$T_g$",
    )
    temperature_axis.plot(z_numpy, solid_kelvin[:, 0], label=r"$T_s(z,R_i)$")
    temperature_axis.set(
        xlabel="z/L", ylabel="temperature [K]", title="Coupled temperatures"
    )
    temperature_axis.legend()

    field = field_axis.pcolormesh(
        z_numpy, r_numpy, solid_kelvin.T, shading="auto", cmap="inferno"
    )
    field_axis.set(xlabel="z/L", ylabel=r"r/$R_i$", title=r"solid $T_s(z,r)$")
    figure.colorbar(field, ax=field_axis, label="temperature [K]")
    for axis in (flow_axis, concentration_axis, temperature_axis):
        axis.grid(alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def print_outlet_summary(
    gas_net: nn.Module,
    parameters: Parameters,
    device: torch.device,
) -> None:
    """Print conversion, outlet temperature, and reaction-rate diagnostics."""
    outlet = torch.ones(1, 1, device=device)
    with torch.no_grad():
        f_a, f_b, f_c, t_gas = gas_state(gas_net(outlet), parameters)
        c_a, _, _ = concentrations_from_state(f_a, f_b, f_c, t_gas, parameters)
        progress_1, progress_2 = reaction_progress_rates(c_a, t_gas, parameters)

    conversion = 1.0 - float(f_a.item() / parameters.inlet_flow_a)
    print(
        f"Outlet: conversion={conversion:.3%}, "
        f"T={float(t_gas.item() * parameters.temperature_scale):.2f} K, "
        f"q1={float(progress_1.item()):.3e}, q2={float(progress_2.item()):.3e}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the small training example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--points", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("Figures")
        / "pfr_cylindrical_heat_pinn.png",
    )
    return parser.parse_args()


def main() -> None:
    """Train the coupled example and optionally save its solution figure."""
    args = parse_args()
    if args.steps < 1 or args.points < 2:
        raise ValueError("--steps must be positive and --points must be at least 2")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    parameters = Parameters()
    print(f"Training coupled PFR/cylindrical-wall PINN on {device}.")
    gas_net, solid_net = train(
        parameters,
        steps=args.steps,
        points=args.points,
        device=device,
    )
    print_outlet_summary(gas_net, parameters, device)
    if not args.no_plot:
        path = save_solution(gas_net, solid_net, parameters, args.output, device)
        print(f"Saved solution to {path}")


if __name__ == "__main__":
    main()
