"""PhysicsNeMo checks for the coupled PFR/cylindrical-wall simulator."""

# PhysicsNeMo must see the Warp cache setting before it is imported.
# pylint: disable=wrong-import-position,too-many-locals

from __future__ import annotations

import os

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/warp")

import torch
from physicsnemo.sym.eq.phy_informer import PhysicsInformer

from chem_operator.reactors.pfr_heat.dataset_generator import (
    CylindricalWall,
    PFRHeatSim,
    PlugFlowReactor,
)


def test_simulated_solution_has_near_zero_physics_loss() -> None:
    """The BVP solution satisfies both PhysicsNeMo PDEs and their BCs."""
    params = {
        "inlet_flow_a": 100.0,
        "inlet_concentration_a": 0.1,
        "inlet_temperature": 393.15,
        "outer_temperature": 323.15,
        "volumetric_heat_transfer": 4000.0,
        "wall_aspect_ratio_sq": 0.2,
        "interface_biot": 1.5,
        "inner_radius": 1.0,
        "outer_radius": 2.0,
        "n_axial_points": 401,
        "n_radial_points": 21,
        "tolerance": 1.0e-6,
        "max_nodes": 10_000,
    }
    simulator = PFRHeatSim()
    record = simulator.run_case(simulator.make_case(params))
    constants = simulator.constants
    z, r = record.coordinates["z"], record.coordinates["r"]
    dz, dr = float(z[1] - z[0]), float(r[1] - r[0])

    def field(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float64)[None, None]

    flows = record.fields["F"] / constants.flow_scale
    gas_temperature = field(
        record.fields["T_gas"] / constants.temperature_scale
    )
    solid_temperature = field(
        record.fields["T_solid"] / constants.temperature_scale
    )
    gas_residuals = PhysicsInformer(
        ["flow_a", "flow_b", "flow_c", "gas_energy"],
        PlugFlowReactor(
            constants,
            params["inlet_concentration_a"],
            params["inlet_temperature"],
            params["volumetric_heat_transfer"],
        ),
        "finite_difference",
        fd_dx=dz,
        device="cpu",
    ).forward(
        {
            "f_a": field(flows[:, 0]),
            "f_b": field(flows[:, 1]),
            "f_c": field(flows[:, 2]),
            "t_gas": gas_temperature,
            "t_wall": solid_temperature[..., 0],
        }
    )
    radial_coordinate = field(r)[..., None, :].expand_as(solid_temperature)
    solid_residual = PhysicsInformer(
        ["solid_heat"],
        CylindricalWall(
            params["wall_aspect_ratio_sq"], params["interface_biot"]
        ),
        "finite_difference",
        fd_dx=[dz, dr],
        device="cpu",
    ).forward({"t_solid": solid_temperature, "y": radial_coordinate})[
        "solid_heat"
    ]

    interior_loss = sum(
        residual[..., 2:-2].square().mean()
        for residual in gas_residuals.values()
    ) + solid_residual[..., 2:-2, 2:-2].square().mean()
    interface_gradient = (
        -3.0 * solid_temperature[..., 0]
        + 4.0 * solid_temperature[..., 1]
        - solid_temperature[..., 2]
    ) / (2.0 * dr)
    axial_gradient = torch.gradient(
        solid_temperature[0, 0], spacing=(dz, dr), edge_order=2
    )[0]
    boundary_loss = (
        (
            field(flows[0])
            - field(
                [params["inlet_flow_a"] / constants.flow_scale, 0.0, 0.0]
            )
        )
        .square()
        .mean()
        + (
            gas_temperature[..., 0]
            - params["inlet_temperature"] / constants.temperature_scale
        ).square().mean()
        + (
            solid_temperature[..., -1]
            - params["outer_temperature"] / constants.temperature_scale
        ).square().mean()
        + axial_gradient[[0, -1]].square().mean()
        + (
            interface_gradient
            - params["interface_biot"]
            * (solid_temperature[..., 0] - gas_temperature)
        ).square().mean()
    )
    physics_loss = interior_loss + boundary_loss

    assert record.metadata["physicsnemo_pdes"] == {
        "reactor": "PlugFlowReactor",
        "solid": "CylindricalWall",
    }
    assert physics_loss.item() < 1.0e-3
