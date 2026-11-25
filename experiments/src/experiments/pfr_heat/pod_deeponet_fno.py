"""Coupled PhysicsNeMo POD-DeepONet reactor and FNO wall experiment."""

# pylint: disable=import-error,wrong-import-position

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from ray import tune
from torch import nn
from torch.utils.data import Dataset, Subset

from chem_operator.experiments import run_operator
from chem_operator.models import PODTransform, fit_incremental_pod_dataset
from experiments.pfr_heat import deeponet_fno
from experiments.pfr_heat.common import (
    PATHS,
    PFR_INPUTS,
    PFR_OUTPUTS,
    experiment_spec,
    fit_normalizers,
    make_adapter,
    raw_dataset,
)
from experiments.pfr_heat.deeponet_fno import (
    check_inputs,
    couple_wall,
    evaluate_run,
    parse_cli_args,
    wall_fno,
)

MODEL_IDS = {
    "data": "pod_deeponet_fno_data",
    "physics": "pod_deeponet_fno_physics",
}
MODEL_ID = MODEL_IDS["physics"]
POD_VARIANCE_THRESHOLD = 0.999


class CoupledPODDeepONetFNO(nn.Module):
    """Reactor POD-DeepONet coupled to a cylindrical-wall FNO.

    The branch MLP predicts coefficients of a fixed POD basis fitted on the
    normalized reactor fields ``[F_A, F_B, F_C, T_gas] x z``; the POD mean
    plays the role of the DeepONet output bias.
    """

    def __init__(self, config, pod: PODTransform, pfr_normalizer, wall_normalizer):
        super().__init__()
        if len(pod.output_shape) != 2 or pod.output_shape[0] != len(PFR_OUTPUTS):
            raise ValueError(
                f"POD basis must represent [{len(PFR_OUTPUTS)}, n_z] reactor "
                f"fields, got {pod.output_shape}."
            )
        self.pfr_normalizer = pfr_normalizer
        self.wall_normalizer = wall_normalizer
        self.output_shape = tuple(pod.output_shape)
        self.register_buffer("pod_mean", torch.as_tensor(pod.mean).float())
        self.register_buffer("pod_basis", torch.as_tensor(pod.basis).float())
        self.branch = FullyConnected(
            in_features=len(PFR_INPUTS),
            out_features=pod.n_components,
            layer_size=int(config["branch_width"]),
            num_layers=int(config["depth"]),
            activation_fn=str(config["activation"]),
        )
        self.wall = wall_fno(config)

    def forward(
        self,
        pfr_branch: torch.Tensor,
        z: torch.Tensor,
        wall_conditions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict normalized reactor and wall fields for one batch."""
        check_inputs(pfr_branch, z, wall_conditions)
        if z.shape[1] != self.output_shape[1]:
            raise ValueError(
                f"POD basis was fitted on {self.output_shape[1]} axial points; "
                f"received {z.shape[1]}."
            )
        coefficients = self.branch(pfr_branch)
        pfr = torch.einsum("bk,dk->bd", coefficients, self.pod_basis)
        pfr = (pfr + self.pod_mean).reshape(-1, *self.output_shape)
        wall = couple_wall(
            self.wall, pfr, wall_conditions,
            self.pfr_normalizer, self.wall_normalizer,
        )
        return {"pfr": pfr, "wall": wall}


def model_from_config(
    config: Mapping[str, Any],
    pod: PODTransform,
    pfr_normalizer,
    wall_normalizer,
    device: torch.device,
) -> CoupledPODDeepONetFNO:
    """Construct the POD hybrid model on the requested device."""
    return CoupledPODDeepONetFNO(
        config, pod, pfr_normalizer, wall_normalizer
    ).to(device)


class _ReactorTargets(Dataset):
    """Expose normalized reactor fields as POD fitting targets."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"target": self.dataset[index]["pfr_y"]}


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def fit_reactor_pod(data_dir, pfr, wall, shape, max_cases=None) -> PODTransform:
    """Fit the reactor POD basis on the (optionally truncated) training split."""
    raw, data = make_adapter(data_dir, "train", pfr, wall, shape)
    try:
        if max_cases is not None:
            data = Subset(data, range(min(len(data), max_cases)))
        pod = fit_incremental_pod_dataset(
            _ReactorTargets(data), variance_threshold=POD_VARIANCE_THRESHOLD
        )
    finally:
        raw.close()
    print(
        f"Reactor POD: {pod.n_components} components explain "
        f"{pod.cumulative_explained_variance:.6f} of the variance."
    )
    return pod


def search_space(variant="physics", pod: PODTransform | None = None) -> dict[str, Any]:
    """Return the branch, wall-FNO, and loss search space."""
    space = deeponet_fno.search_space(variant)
    space.pop("trunk_width")
    space["branch_width"] = tune.choice([128, 256, 512])
    if pod is None:
        space.pop("latent_width")
    else:
        space["latent_width"] = pod.n_components
    return space


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def make_trainer(config, context, pod, pfr, wall, shape, *, variant="physics"):
    """Build the end-to-end trainer around the POD reactor model."""
    return deeponet_fno.make_trainer(
        config, context, pfr, wall, shape, variant=variant,
        build_model=lambda cfg: model_from_config(
            cfg, pod, pfr, wall, context.device
        ),
        metadata={
            "pod": pod.state_dict(),
            "components": {
                "reactor": "PhysicsNeMo FullyConnected POD-DeepONet",
                "wall": "PhysicsNeMo FNO",
            },
        },
    )


def main():
    """Tune, train, and evaluate a canonical POD hybrid experiment run."""
    args = parse_cli_args()
    raw = raw_dataset(args.data_dir, "train")
    try:
        pfr, wall, shape = fit_normalizers(raw)
    finally:
        raw.close()
    pod = fit_reactor_pod(args.data_dir, pfr, wall, shape, args.max_cases)
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        model_id = MODEL_IDS[variant]
        run_operator(
            args,
            PATHS,
            experiment_spec(args, model_id),
            lambda split: make_adapter(args.data_dir, split, pfr, wall, shape, preload=getattr(args, "cache_data", False)),
            lambda config, context, selected=variant: make_trainer(
                config, context, pod, pfr, wall, shape, variant=selected
            ),
            search_space(variant, pod),
            lambda trainer, data, context: evaluate_run(
                trainer, data, context, pfr, wall, args.plot_cases
            ),
        )


if __name__ == "__main__":
    main()
