"""Model definition, scientific losses, and canonical experiment entry point."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any, Mapping
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ray import tune
import torch
from torch.nn import functional as torch_functional
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.experiments import CompositeLossTrainer, LossTerm
from chem_operator.experiments import run_operator, evaluate_fields
from scripts.pipe_flow_transient.common import (
    PATHS,
    FIELD_NAMES,
    MODEL_CONSTANT_NAMES,
    PHYSICS_CONSTANT_NAMES,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    FILE_STEM,
    METRIC,
    SEED,
    PDE_REGISTRY,
    MAX_TRAIN_TRAJECTORIES,
    CONSTANT_NAMES,
    PhysicsFNOAdapter,
    raw_dataset,
    dataset_pde_name,
    make_adapter,
    fit_normalizer,
    normalizer_state,
    normalizer_from_state,
    adapter_radial_spacing,
    training_pde_name,
    PROBLEM_ID,
    experiment_spec,
    parse_args,
)

def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured two-dimensional PhysicsNeMo FNO."""
    modes = int(config["modes"])
    return FNO(
        in_channels=len(MODEL_CONSTANT_NAMES),
        out_channels=len(FIELD_NAMES),
        dimension=2,
        latent_channels=int(config["latent_channels"]),
        num_fno_layers=int(config["n_layers"]),
        num_fno_modes=[modes, modes],
        padding=int(config.get("padding", 4)),
        decoder_layers=int(config.get("decoder_layers", 1)),
        decoder_layer_size=int(config.get("decoder_layer_size", 32)),
        coord_features=True,
    ).to(device)


def make_physics_informer(
    pde_name: str,
    *,
    radial_spacing: float,
    device: torch.device,
) -> PhysicsInformer:
    """Construct a metadata-selected cylindrical momentum evaluator."""
    try:
        equation = PDE_REGISTRY[pde_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported PhysicsNeMo PDE {pde_name!r}.") from exc
    return PhysicsInformer(
        required_outputs=["momentum"],
        equations=equation,
        grad_method="meshless_finite_difference",
        fd_dx=radial_spacing,
        device=str(device),
    )


def _validate_batch_pde(batch: Mapping[str, Any], expected: str) -> None:
    names = batch["physicsnemo_pde"]
    if isinstance(names, str):
        names = [names]
    observed = {str(name) for name in names}
    if observed != {expected}:
        raise ValueError(
            f"Batch PDE metadata {sorted(observed)} does not match {expected!r}."
        )


def physics_losses(  # pylint: disable=too-many-arguments,too-many-locals
    prediction_normalized: torch.Tensor,
    batch: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    informer: PhysicsInformer,
    *,
    pde_name: str,
    radial_spacing: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nondimensional PDE-residual and condition losses.

    ``TransientHagenPoiseuille`` uses ``x=r/R`` and
    ``tau=nu*t/R**2``. PhysicsInformer computes the radial derivatives from
    meshless centered stencils; the centered time derivative is supplied as
    ``velocity__t`` because the PDE intentionally declares time as an external
    derivative.
    """
    _validate_batch_pde(batch, pde_name)
    if prediction_normalized.ndim != 4 or prediction_normalized.shape[1] != 1:
        raise ValueError(
            "Expected normalized velocity with shape (batch, 1, time, radius); "
            f"received {tuple(prediction_normalized.shape)}."
        )
    if prediction_normalized.shape[-2] < 3 or prediction_normalized.shape[-1] < 3:
        raise ValueError("Physics loss requires at least three points per axis.")

    velocity = normalizer.denormalize(
        prediction_normalized[:, 0], "velocity"
    )
    constants = batch["physics_constants"].to(
        device=velocity.device,
        dtype=velocity.dtype,
    )
    radius, length, viscosity, pressure_drop, density = constants.unbind(dim=1)
    steady_max_velocity = pressure_drop * radius.square() / (
        4.0 * viscosity * length
    )
    dimensionless_velocity = velocity / steady_max_velocity[:, None, None]

    time_values = batch["t"].to(device=velocity.device, dtype=velocity.dtype)
    radial_values = batch["r"].to(device=velocity.device, dtype=velocity.dtype)
    kinematic_viscosity = viscosity / density
    tau = kinematic_viscosity[:, None] * time_values / radius[:, None].square()
    x = radial_values / radius[:, None]

    radial_differences = x[:, 1:] - x[:, :-1]
    expected_dx = torch.as_tensor(
        radial_spacing,
        device=velocity.device,
        dtype=velocity.dtype,
    )
    if not torch.allclose(
        radial_differences,
        expected_dx.expand_as(radial_differences),
        rtol=1.0e-4,
        atol=1.0e-6,
    ):
        raise ValueError("Batch does not match the PhysicsInformer radial spacing.")
    tau_step = tau[:, 2:] - tau[:, :-2]
    if torch.any(tau_step <= 0.0):
        raise ValueError("Dimensionless time coordinates must be increasing.")

    velocity_t = (
        dimensionless_velocity[:, 2:, 1:-1]
        - dimensionless_velocity[:, :-2, 1:-1]
    ) / tau_step[:, :, None]
    center = dimensionless_velocity[:, 1:-1, 1:-1]
    radial_coordinate = x[:, None, 1:-1].expand_as(center)
    residual = informer.forward(
        {
            "velocity": center,
            "velocity>>x::1": dimensionless_velocity[:, 1:-1, 2:],
            "velocity>>x::-1": dimensionless_velocity[:, 1:-1, :-2],
            "velocity__t": velocity_t,
            "x": radial_coordinate,
        }
    )["momentum"]
    physics_loss = residual.square().mean()

    initial_loss = dimensionless_velocity[:, 0, :].square().mean()
    wall_loss = dimensionless_velocity[:, :, -1].square().mean()
    center_gradient = (
        -3.0 * dimensionless_velocity[:, :, 0]
        + 4.0 * dimensionless_velocity[:, :, 1]
        - dimensionless_velocity[:, :, 2]
    ) / (2.0 * expected_dx)
    constraint_loss = (
        initial_loss + wall_loss + center_gradient.square().mean()
    ) / 3.0
    return physics_loss, constraint_loss


def relative_l2_batch(  # pylint: disable=not-callable
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Return the mean per-case relative L2 error."""
    error = torch.linalg.vector_norm(
        (prediction - target).flatten(start_dim=1), dim=1
    )
    scale = torch.linalg.vector_norm(
        target.flatten(start_dim=1), dim=1
    ).clamp_min(1.0e-12)
    return (error / scale).mean()


PHYSICS_WEIGHT_KEYS = ("physics_weight", "constraint_weight")


def search_space(variant="physics"):
    space = {'modes': tune.choice([6, 8, 10]), 'latent_channels': tune.choice([8, 16]), 'n_layers': tune.choice([3, 4]), 'padding': tune.choice([0, 4, 8]), 'decoder_layers': tune.choice([1, 2]), 'decoder_layer_size': tune.choice([8, 16, 32]), 'learning_rate': tune.loguniform(0.0001, 0.003), 'weight_decay': tune.loguniform(1e-08, 0.0001), 'batch_size': tune.choice([2, 4, 8])}
    if variant == "data":
        space.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant == "physics":
        space.update({
            "physics_weight": tune.loguniform(0.0001, 0.1),
            "constraint_weight": tune.loguniform(0.0001, 0.1),
        })
    else:
        raise ValueError(f"Unknown variant {variant!r}.")
    return space

MODEL_IDS = {
    "data": "transient_nemo_fno_data",
    "physics": "transient_nemo_fno_physics",
}
MODEL_ID = MODEL_IDS["physics"]

def make_trainer(config, context, normalizer, *, pde_name, radial_spacing, variant="physics"):
    config = dict(config)
    if variant == "data":
        config.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant != "physics":
        raise ValueError(f"Unknown variant {variant!r}.")
    physics_active = any(
        float(config.get(key, 0.0)) > 0.0 for key in PHYSICS_WEIGHT_KEYS
    )
    informer = make_physics_informer(
        pde_name, radial_spacing=radial_spacing, device=context.device
    )

    def forward(model, inputs, batch):
        prediction = model(inputs)
        if physics_active or not model.training:
            batch["physics_loss"], batch["constraint_loss"] = physics_losses(
                prediction,
                batch,
                normalizer,
                informer,
                pde_name=pde_name,
                radial_spacing=radial_spacing,
            )
        else:
            zero = prediction.new_zeros(())
            batch["physics_loss"], batch["constraint_loss"] = zero, zero
        return prediction
    return CompositeLossTrainer(
        lambda cfg: model_from_config(cfg, context.device), config,
        forward_adapter=forward,
        loss_terms=[LossTerm("data_loss", lambda p,t,b: torch_functional.mse_loss(p,t)),
                    LossTerm("physics_loss", lambda p,t,b: b["physics_loss"], float(config["physics_weight"])),
                    LossTerm("constraint_loss", lambda p,t,b: b["constraint_loss"], float(config["constraint_weight"]))],
        metric_adapter=lambda p,t,b: (normalizer.denormalize(p, "velocity"), normalizer.denormalize(t, "velocity")),
        checkpoint_metadata={"normalizer": normalizer.state_dict(), "pde_name": pde_name,
                             "radial_spacing": radial_spacing, "input_channels": list(MODEL_CONSTANT_NAMES),
                             "output_channels": list(FIELD_NAMES)},
    )

def physics_metrics(trainer, data, context):
    losses = trainer.evaluate_losses(data, context)
    return {name: losses[name] for name in ("physics_loss", "constraint_loss")}


def parse_cli_args():
    """Parse the shared workflow arguments plus the loss variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("both", "data", "physics"), default="physics"
    )
    return parse_args(epochs=15, tune_epochs=10, samples=3, parser=parser)


def main():
    args = parse_cli_args()
    normalizer = fit_normalizer(args.data_dir)
    raw, data = make_adapter(args.data_dir, "train", normalizer, args.max_cases)
    try:
        pde_name, spacing = dataset_pde_name(raw, args.max_cases), adapter_radial_spacing(data)
    finally:
        raw.close()
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        model_id = MODEL_IDS[variant]
        run_operator(
            args, PATHS, experiment_spec(args, model_id),
            lambda split: make_adapter(args.data_dir, split, normalizer, args.max_cases),
            lambda config, context, selected=variant: make_trainer(
                config, context, normalizer, pde_name=pde_name,
                radial_spacing=spacing, variant=selected
            ),
            search_space(variant),
            lambda trainer, data, context: evaluate_fields(
                trainer, data, context, labels=FIELD_NAMES,
                coordinate_names=("t", "r"), cases=args.plot_cases,
                extra_metrics=physics_metrics
            ),
        )

if __name__ == "__main__":
    main()
