"""Coupled PhysicsNeMo DeepONet reactor and FNO wall experiment."""

# pylint: disable=import-error,wrong-import-position

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from physicsnemo.models.fno import FNO
from physicsnemo.models.mlp.fully_connected import FullyConnected
from ray import tune
from torch import nn
from torch.utils.data import DataLoader

from chem_operator.experiments import (
    CompositeLossTrainer,
    EvaluationOutcome,
    LossTerm,
    StreamingRegressionMetrics,
    run_operator,
)
from scripts.pfr_heat.common import (
    EXPECTED_PDES,
    PATHS,
    PFR_INPUTS,
    PFR_OUTPUTS,
    WALL_CONDITIONS,
    WALL_GAS,
    WALL_OUTPUTS,
    experiment_spec,
    fit_normalizers,
    make_adapter,
    parse_args,
    raw_dataset,
)
from scripts.pfr_heat.fno import (
    InformerCache,
    PHYSICS_WEIGHT_KEYS,
    physical_predictions,
    physics_losses,
    supervised_loss,
)


class CoupledDeepONetFNO(nn.Module):
    """Multi-output reactor DeepONet coupled to a cylindrical-wall FNO."""

    def __init__(self, config, pfr_normalizer, wall_normalizer):
        super().__init__()
        self.pfr_normalizer = pfr_normalizer
        self.wall_normalizer = wall_normalizer
        latent_width = int(config["latent_width"])
        depth = int(config["depth"])
        activation = str(config["activation"])
        self.latent_width = latent_width
        self.branch = FullyConnected(
            in_features=len(PFR_INPUTS),
            out_features=len(PFR_OUTPUTS) * latent_width,
            layer_size=int(config["branch_width"]),
            num_layers=depth,
            activation_fn=activation,
        )
        self.trunk = FullyConnected(
            in_features=1,
            out_features=latent_width,
            layer_size=int(config["trunk_width"]),
            num_layers=depth,
            activation_fn=activation,
        )
        self.pfr_bias = nn.Parameter(torch.zeros(len(PFR_OUTPUTS)))
        self.wall = FNO(
            in_channels=1 + len(WALL_CONDITIONS),
            out_channels=1,
            dimension=2,
            latent_channels=int(config["wall_latent_channels"]),
            num_fno_layers=int(config["wall_n_layers"]),
            num_fno_modes=[
                int(config["wall_modes_z"]),
                int(config["wall_modes_r"]),
            ],
            padding=int(config.get("wall_padding", 4)),
            decoder_layers=int(config.get("wall_decoder_layers", 1)),
            decoder_layer_size=int(config.get("wall_decoder_layer_size", 32)),
            coord_features=True,
        )

    def forward(
        self,
        pfr_branch: torch.Tensor,
        z: torch.Tensor,
        wall_conditions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict normalized reactor and wall fields for one batch."""
        if pfr_branch.ndim != 2 or pfr_branch.shape[-1] != len(PFR_INPUTS):
            raise ValueError(
                f"pfr_branch must have shape [batch, {len(PFR_INPUTS)}]."
            )
        if z.ndim != 2 or z.shape[0] != pfr_branch.shape[0]:
            raise ValueError("z must have shape [batch, n_z].")
        if (
            wall_conditions.ndim != 4
            or wall_conditions.shape[0] != pfr_branch.shape[0]
            or wall_conditions.shape[1] != len(WALL_CONDITIONS)
            or wall_conditions.shape[2] != z.shape[1]
        ):
            raise ValueError(
                "wall_conditions must have shape "
                f"[batch, {len(WALL_CONDITIONS)}, n_z, n_r]."
            )

        branch = self.branch(pfr_branch).reshape(
            pfr_branch.shape[0], len(PFR_OUTPUTS), self.latent_width
        )
        trunk = self.trunk((2.0 * z - 1.0).unsqueeze(-1))
        pfr = torch.einsum("bcl,bzl->bcz", branch, trunk)
        pfr = pfr + self.pfr_bias[None, :, None]

        gas = self.pfr_normalizer.denormalize(pfr[:, 3], "T_gas")
        gas = self.wall_normalizer.normalize(gas, WALL_GAS)
        gas = gas[..., None].expand(-1, -1, wall_conditions.shape[-1])
        wall = self.wall(torch.cat((gas[:, None], wall_conditions), dim=1))
        return {"pfr": pfr, "wall": wall}


def model_from_config(
    config: Mapping[str, Any],
    pfr_normalizer,
    wall_normalizer,
    device: torch.device,
) -> CoupledDeepONetFNO:
    """Construct the hybrid model on the requested device."""
    return CoupledDeepONetFNO(config, pfr_normalizer, wall_normalizer).to(device)


def search_space(variant="physics") -> dict[str, Any]:
    """Return the independent DeepONet, wall-FNO, and loss search space."""
    space = {
        "branch_width": tune.choice([64, 128]),
        "trunk_width": tune.choice([64, 128]),
        "depth": tune.choice([2, 3]),
        "latent_width": tune.choice([16, 32]),
        "activation": tune.choice(["silu", "gelu", "tanh"]),
        "wall_modes_z": tune.choice([8, 12, 16]),
        "wall_modes_r": tune.choice([4, 6, 8]),
        "wall_latent_channels": tune.choice([8, 16, 24]),
        "wall_n_layers": tune.choice([3, 4]),
        "wall_padding": tune.choice([0, 4]),
        "wall_decoder_layers": tune.choice([1, 2]),
        "wall_decoder_layer_size": tune.choice([16, 32]),
        "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
        "batch_size": tune.choice([1, 2, 4]),
    }
    if variant == "data":
        space.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant == "physics":
        space.update({
            key: tune.loguniform(1.0e-4, 1.0e-1)
            for key in PHYSICS_WEIGHT_KEYS
        })
    else:
        raise ValueError(f"Unknown variant {variant!r}.")
    return space


MODEL_IDS = {
    "data": "deeponet_fno_data",
    "physics": "deeponet_fno_physics",
}
MODEL_ID = MODEL_IDS["physics"]


def make_trainer(config, context, pfr, wall, shape, *, variant="physics"):
    """Build the end-to-end data- and physics-informed trainer."""
    config = dict(config)
    if variant == "data":
        config.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant != "physics":
        raise ValueError(f"Unknown variant {variant!r}.")
    physics_active = any(
        float(config.get(key, 0.0)) > 0.0 for key in PHYSICS_WEIGHT_KEYS
    )
    cache = InformerCache(context.device)

    def adapt(batch, device, dtype):
        batch = {
            key: value.to(device=device, dtype=dtype)
            for key, value in batch.items()
        }
        inputs = (batch["pfr_branch"], batch["z"], batch["wall_conditions"])
        targets = {"pfr": batch["pfr_y"], "wall": batch["wall_y"]}
        return inputs, targets, batch

    def forward(model, inputs, batch):
        prediction = model(*inputs)
        if physics_active or not model.training:
            batch["physics_losses"] = physics_losses(
                prediction, batch, pfr, wall, cache
            )
        else:
            zero = prediction["pfr"].new_zeros(())
            batch["physics_losses"] = {
                name: zero for name in ("species", "gas", "solid", "bc")
            }
        return prediction

    def metric(prediction, target, _batch):
        predicted = physical_predictions(prediction, pfr, wall)
        reference = physical_predictions(target, pfr, wall)
        return tuple(
            torch.cat([value.flatten(1) for value in values], dim=1)
            for values in (predicted, reference)
        )

    terms = [LossTerm(
        "data_loss",
        lambda prediction, _target, batch: supervised_loss(prediction, batch),
    )]
    for name, key in (
        ("species", "lambda_f"),
        ("gas", "lambda_g"),
        ("solid", "lambda_s"),
        ("bc", "lambda_bc"),
    ):
        terms.append(
            LossTerm(
                name + "_loss",
                lambda _prediction, _target, batch, loss_name=name: batch[
                    "physics_losses"
                ][loss_name],
                float(config[key]),
            )
        )

    return CompositeLossTrainer(
        lambda cfg: model_from_config(cfg, pfr, wall, context.device),
        config,
        batch_adapter=adapt,
        forward_adapter=forward,
        loss_terms=terms,
        metric_adapter=metric,
        checkpoint_metadata={
            "pfr_normalizer": pfr.state_dict(),
            "wall_normalizer": wall.state_dict(),
            "shape": list(shape),
            "pdes": EXPECTED_PDES,
            "pfr_inputs": list(PFR_INPUTS),
            "pfr_outputs": list(PFR_OUTPUTS),
            "wall_conditions": list(WALL_CONDITIONS),
            "wall_outputs": list(WALL_OUTPUTS),
            "components": {
                "reactor": "PhysicsNeMo FullyConnected DeepONet",
                "wall": "PhysicsNeMo FNO",
            },
        },
    )


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-positional-arguments
def evaluate_run(trainer, data, context, pfr, wall, cases=2):
    """Evaluate physical-unit fields, residuals, and saved reconstructions."""
    scores = StreamingRegressionMetrics()
    fields = {
        name: StreamingRegressionMetrics()
        for name in PFR_OUTPUTS + WALL_OUTPUTS
    }
    stored = {
        name: []
        for name in (
            "pfr_reference",
            "pfr_prediction",
            "wall_reference",
            "wall_prediction",
            "z",
            "r",
        )
    }
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(
            data, batch_size=int(trainer.config["batch_size"])
        ):
            prediction = trainer.model(
                batch["pfr_branch"].to(context.device),
                batch["z"].to(context.device),
                batch["wall_conditions"].to(context.device),
            )
            predicted_physical = tuple(
                value.cpu()
                for value in physical_predictions(prediction, pfr, wall)
            )
            target_physical = physical_predictions(
                {"pfr": batch["pfr_y"], "wall": batch["wall_y"]},
                pfr,
                wall,
            )
            predicted_pfr = torch.cat(
                (predicted_physical[0], predicted_physical[1][:, None]), dim=1
            )
            target_pfr = torch.cat(
                (target_physical[0], target_physical[1][:, None]), dim=1
            )
            scores.update(
                torch.cat(
                    [value.flatten(1) for value in predicted_physical], dim=1
                ),
                torch.cat(
                    [value.flatten(1) for value in target_physical], dim=1
                ),
            )
            for index, name in enumerate(PFR_OUTPUTS):
                fields[name].update(predicted_pfr[:, index], target_pfr[:, index])
            fields["T_solid"].update(
                predicted_physical[2], target_physical[2]
            )
            count = max(0, min(cases - len(stored["z"]), len(target_pfr)))
            for index in range(count):
                for key, value in (
                    ("pfr_reference", target_pfr),
                    ("pfr_prediction", predicted_pfr),
                    ("wall_reference", target_physical[2][:, None]),
                    ("wall_prediction", predicted_physical[2][:, None]),
                    ("z", batch["z"]),
                    ("r", batch["r"]),
                ):
                    stored[key].append(value[index].numpy())
    elapsed = time.perf_counter() - started
    metrics = scores.compute()
    for name, value in fields.items():
        metrics.update(
            {f"{name}/{key}": score for key, score in value.compute().items()}
        )
    diagnostics = trainer.evaluate_losses(data, context)
    metrics.update(
        {key: value for key, value in diagnostics.items() if key.endswith("_loss")}
    )
    return EvaluationOutcome(
        metrics,
        {
            **{key: np.asarray(value) for key, value in stored.items()},
            "case_ids": np.arange(len(stored["z"])),
            "pfr_labels": np.asarray(PFR_OUTPUTS),
            "wall_labels": np.asarray(WALL_OUTPUTS),
        },
        elapsed,
    )


def parse_cli_args():
    """Parse the shared workflow arguments plus the loss variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("both", "data", "physics"), default="physics"
    )
    return parse_args(parser=parser)


def main():
    """Tune, train, and evaluate a canonical hybrid experiment run."""
    args = parse_cli_args()
    raw = raw_dataset(args.data_dir, "train")
    try:
        pfr, wall, shape = fit_normalizers(raw)
    finally:
        raw.close()
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        model_id = MODEL_IDS[variant]
        run_operator(
            args,
            PATHS,
            experiment_spec(args, model_id),
            lambda split: make_adapter(args.data_dir, split, pfr, wall, shape),
            lambda config, context, selected=variant: make_trainer(
                config, context, pfr, wall, shape, variant=selected
            ),
            search_space(variant),
            lambda trainer, data, context: evaluate_run(
                trainer, data, context, pfr, wall, args.plot_cases
            ),
        )


if __name__ == "__main__":
    main()
