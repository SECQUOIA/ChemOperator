"""DeepONet architecture, data loading, and loss helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from .deepxde import DeepONetAdapter
from .pod import PODTransform


@dataclass(frozen=True)
class CoordinateScaler:
    minimum: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, coordinates: np.ndarray) -> "CoordinateScaler":
        minimum = np.min(coordinates, axis=0)
        maximum = np.max(coordinates, axis=0)
        span = np.maximum(maximum - minimum, 1e-12)
        return cls(minimum=minimum, span=span)

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        return (2.0 * (coordinates - self.minimum) / self.span - 1.0).astype(
            np.float32,
            copy=False,
        )

    def transform_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        minimum = torch.as_tensor(
            self.minimum,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        span = torch.as_tensor(
            self.span,
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        return 2.0 * (coordinates - minimum) / span - 1.0


@dataclass(frozen=True)
class DeepONetTrainingConfig:
    """DeepONet architecture and optimization settings."""

    loss: Literal["relative_l2", "mse"] = "relative_l2"
    epochs: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    width: int = 128
    latent_width: int = 8
    branch_hidden_layers: int = 2
    trunk_hidden_layers: int = 2
    activation: str = "tanh"
    display_every: int = 100
    seed: int = 7

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible architecture and training metadata."""

        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> "DeepONetTrainingConfig":
        """Restore a configuration previously produced by :meth:`to_dict`."""

        return cls(**dict(values))


def _network(
    branch_width: int,
    trunk_width: int,
    output_width: int,
    config: DeepONetTrainingConfig,
):
    import deepxde as dde

    if output_width == 1:
        branch_last = config.latent_width
        strategy = None
    else:
        branch_last = config.latent_width * output_width
        strategy = "split_branch"
    branch_layers = [
        branch_width,
        *([config.width] * config.branch_hidden_layers),
        branch_last,
    ]
    trunk_layers = [
        trunk_width,
        *([config.width] * config.trunk_hidden_layers),
        config.latent_width,
    ]
    return dde.nn.DeepONetCartesianProd(
        branch_layers,
        trunk_layers,
        config.activation,
        "Glorot normal",
        num_outputs=output_width,
        multi_output_strategy=strategy,
        regularization=("l2", config.weight_decay),
    )


class _PODOnlyDeepONet(torch.nn.Module):
    """DeepXDE PODDeepONet with a fixed POD-only trunk and PCA mean shift."""

    def __init__(
        self,
        branch_width: int,
        pod: PODTransform,
        config: DeepONetTrainingConfig,
    ):
        super().__init__()
        import deepxde as dde

        self.network = dde.nn.PODDeepONet(
            np.array(pod.basis, dtype=np.float32, copy=True),
            [
                branch_width,
                *([config.width] * config.branch_hidden_layers),
                pod.n_components,
            ],
            config.activation,
            "Glorot normal",
            layer_sizes_trunk=None,
            regularization=("l2", config.weight_decay),
        )
        basis = self.network.pod_basis
        del self.network.pod_basis
        self.network.register_buffer("pod_basis", basis)
        self.register_buffer(
            "pod_mean",
            torch.tensor(np.array(pod.mean, copy=True), dtype=torch.float32),
        )

    @property
    def branch(self) -> torch.nn.Module:
        return self.network.branch

    @property
    def trunk(self) -> None:
        return None

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.network(inputs) + self.pod_mean


@dataclass
class DeepONetTrainingHistory:
    """Minimal loss history shared by training and artifact serialization."""

    steps: list[int]
    loss_train: list[list[float]]
    loss_test: list[list[float]]
    loss_name: str = "relative_l2"
    best_epoch: int | None = None
    best_valid_loss: float = float("inf")


TensorTransform = Callable[[torch.Tensor], torch.Tensor]
MetricReporter = Callable[[Mapping[str, float | int]], None]
CheckpointCallback = Callable[
    [torch.nn.Module, Mapping[str, float | int]],
    None,
]


def deeponet_parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    """Count trainable parameters in the branch, trunk, and full network."""

    def trainable(module: torch.nn.Module) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    branch = getattr(model, "branch", None)
    trunk = getattr(model, "trunk", None)
    if not isinstance(branch, torch.nn.Module):
        raise TypeError("DeepONet model does not expose a branch module.")
    return {
        "n_params": trainable(model),
        "n_params_branch": trainable(branch),
        "n_params_trunk": 0 if trunk is None else trainable(trunk),
    }


def collate_deeponet_trajectories(
    samples: Sequence[Mapping[str, Any]],
    *,
    coordinate_scaler: CoordinateScaler,
    target_transform: TensorTransform | None = None,
) -> dict[str, Any]:
    """Collate lazily loaded trajectories on one shared trunk grid."""

    if not samples:
        raise ValueError("Cannot collate an empty trajectory batch.")
    reference_trunk = samples[0]["trunk"]
    for sample in samples[1:]:
        candidate = sample["trunk"]
        if candidate.shape != reference_trunk.shape or not torch.allclose(
            candidate,
            reference_trunk,
            rtol=1e-5,
            atol=1e-8,
        ):
            raise ValueError(
                "Cartesian-product batches require a shared trunk grid."
            )
    targets = torch.stack([sample["target"] for sample in samples])
    if target_transform is not None:
        targets = target_transform(targets)
    return {
        "branch": torch.stack([sample["branch"] for sample in samples]),
        "trunk": coordinate_scaler.transform_tensor(reference_trunk),
        "target": targets,
        "coordinates": tuple(sample["coordinate"] for sample in samples),
        "labels": tuple(samples[0]["labels"]),
    }


def make_deeponet_dataloader(
    dataset: DeepONetAdapter,
    *,
    batch_size: int,
    shuffle: bool,
    coordinate_scaler: CoordinateScaler,
    target_transform: TensorTransform | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Build a loader that reads only the trajectories in the current batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    # DataLoader indices and its generator are CPU-side even when the model is
    # on a worker-local accelerator. Do not inherit process-global defaults.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        generator=generator,
        collate_fn=partial(
            collate_deeponet_trajectories,
            coordinate_scaler=coordinate_scaler,
            target_transform=target_transform,
        ),
    )


def _loss_target(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if prediction.ndim + 1 == target.ndim and target.shape[-1] == 1:
        return target[..., 0]
    return target


def _with_output_channel(
    prediction: torch.Tensor,
    output_width: int,
) -> torch.Tensor:
    if output_width == 1 and prediction.ndim == 2:
        return prediction.unsqueeze(-1)
    return prediction


def relative_l2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean trajectory-wise ``||prediction-target||₂ / ||target||₂``."""

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes for relative L2."
        )
    error = (prediction - target).reshape(target.shape[0], -1)
    reference = target.reshape(target.shape[0], -1)
    denominator = torch.linalg.vector_norm(reference, dim=1)
    denominator = denominator.clamp_min(torch.finfo(target.dtype).eps)
    return (
        torch.linalg.vector_norm(error, dim=1) / denominator
    ).mean()


def _loader_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pin_memory: bool,
    loss_name: Literal["relative_l2", "mse"],
) -> float:
    training = optimizer is not None
    model.train(training)
    accumulated_loss = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    parameter = next(model.parameters())
    model_dtype = parameter.dtype
    with context:
        for batch in loader:
            branch = batch["branch"].to(
                device,
                dtype=model_dtype,
                non_blocking=pin_memory,
            )
            trunk = batch["trunk"].to(
                device,
                dtype=model_dtype,
                non_blocking=pin_memory,
            )
            target = batch["target"].to(
                device,
                dtype=model_dtype,
                non_blocking=pin_memory,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model((branch, trunk))
            target = _loss_target(prediction, target)
            if loss_name == "relative_l2":
                loss = relative_l2_loss(prediction, target)
                batch_count = target.shape[0]
            elif loss_name == "mse":
                loss = torch.nn.functional.mse_loss(prediction, target)
                batch_count = target.numel()
            else:
                raise ValueError(f"Unsupported loss {loss_name!r}.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            accumulated_loss += float(loss.detach()) * batch_count
            count += batch_count
    return accumulated_loss / max(count, 1)
