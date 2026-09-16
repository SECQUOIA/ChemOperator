"""Incremental POD transforms and fitting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .deepxde import DeepONetAdapter


@dataclass(frozen=True)
class PODTransform:
    """Trajectory-output POD transform fitted by incremental PCA."""

    mean: np.ndarray
    basis: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: float
    output_shape: tuple[int, ...]

    @property
    def n_components(self) -> int:
        return self.basis.shape[1]

    def flatten(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states)
        if tuple(states.shape[-len(self.output_shape) :]) != self.output_shape:
            raise ValueError(
                f"POD states must end in {self.output_shape}, got "
                f"{tuple(states.shape)}."
            )
        leading = states.shape[: -len(self.output_shape)]
        return states.reshape(leading + (self.mean.size,))

    def flatten_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if tuple(states.shape[-len(self.output_shape) :]) != self.output_shape:
            raise ValueError(
                f"POD states must end in {self.output_shape}, got "
                f"{tuple(states.shape)}."
            )
        leading = tuple(states.shape[: -len(self.output_shape)])
        return states.reshape(leading + (self.mean.size,))

    def encode(self, states: np.ndarray) -> np.ndarray:
        flattened = self.flatten(states)
        return np.einsum(
            "...d,dk->...k", flattened - self.mean, self.basis
        ).astype(np.float32, copy=False)

    def encode_tensor(self, states: torch.Tensor) -> torch.Tensor:
        flattened = self.flatten_tensor(states)
        mean = torch.tensor(
            self.mean, dtype=states.dtype, device=states.device
        )
        basis = torch.tensor(
            self.basis, dtype=states.dtype, device=states.device
        )
        return torch.einsum("...d,dk->...k", flattened - mean, basis)

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        flattened = np.einsum(
            "...k,dk->...d", coefficients, self.basis
        ) + self.mean
        return flattened.reshape(
            flattened.shape[:-1] + self.output_shape
        ).astype(np.float32, copy=False)

    def decode_tensor(self, coefficients: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.mean,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        basis = torch.tensor(
            self.basis,
            dtype=coefficients.dtype,
            device=coefficients.device,
        )
        flattened = (
            torch.einsum("...k,dk->...d", coefficients, basis) + mean
        )
        return flattened.reshape(
            tuple(flattened.shape[:-1]) + self.output_shape
        )

    def unflatten_tensor(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.mean.size:
            raise ValueError(
                f"Flattened POD output has {states.shape[-1]} values; "
                f"expected {self.mean.size}."
            )
        return states.reshape(tuple(states.shape[:-1]) + self.output_shape)


def fit_incremental_pod(
    trajectory_batches: np.ndarray | Iterable[np.ndarray],
    *,
    variance_threshold: float = 0.999,
    n_components: int | None = None,
) -> PODTransform:
    """Fit trajectory-output IPCA from batches shaped ``[B, *output_shape]``.

    Components are truncated at the first cumulative explained-variance value
    meeting ``variance_threshold``.
    """

    if not 0.0 < variance_threshold <= 1.0:
        raise ValueError("variance_threshold must be in (0, 1].")
    batches = iter(trajectory_batches)
    try:
        first = np.asarray(next(batches), dtype=np.float32)
    except StopIteration:
        raise ValueError("At least one trajectory batch is required for POD fitting.")
    if first.ndim < 2:
        raise ValueError("POD batches must have shape [batch, *output_shape].")
    output_shape = tuple(first.shape[1:])
    flattened = first.reshape(first.shape[0], -1)
    n_features = flattened.shape[1]
    if n_components is None:
        n_components = min(n_features, flattened.shape[0])
    n_components = int(n_components)
    if n_components < 1:
        raise ValueError("n_components must be positive.")
    if flattened.shape[0] < n_components:
        raise ValueError("The first IPCA batch is smaller than n_components.")

    from sklearn.decomposition import IncrementalPCA

    ipca = IncrementalPCA(n_components=n_components)
    ipca.partial_fit(flattened)
    for batch in batches:
        batch = np.asarray(batch, dtype=np.float32)
        if tuple(batch.shape[1:]) != output_shape:
            raise ValueError("POD trajectory output shapes do not match.")
        flattened = batch.reshape(batch.shape[0], -1)
        if batch.shape[0] < n_components:
            raise ValueError(
                "Every IPCA trajectory batch must have at least as many "
                "trajectories as n_components."
            )
        ipca.partial_fit(flattened)

    ratios = np.nan_to_num(ipca.explained_variance_ratio_, nan=0.0)
    cumulative = np.minimum(np.cumsum(ratios), 1.0)
    if cumulative[-1] + 1e-7 < variance_threshold:
        raise ValueError(
            f"The IPCA batches explain only {cumulative[-1]:.6f} of variance; "
            "fit more components."
        )
    retained = int(np.searchsorted(cumulative, variance_threshold) + 1)
    basis = ipca.components_[:retained].T.astype(np.float32, copy=False)
    return PODTransform(
        mean=ipca.mean_.astype(np.float32, copy=False),
        basis=basis,
        explained_variance_ratio=ratios[:retained].astype(np.float32, copy=False),
        cumulative_explained_variance=float(cumulative[retained - 1]),
        output_shape=output_shape,
    )


def _collate_pod_trajectories(
    samples: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    return torch.stack([sample["target"] for sample in samples])


def fit_incremental_pod_dataset(
    dataset: DeepONetAdapter,
    *,
    variance_threshold: float = 0.999,
    max_components: int | None = None,
    num_workers: int = 0,
) -> PODTransform:
    """Fit trajectory POD lazily, increasing IPCA rank until the threshold."""

    n_trajectories = len(dataset)
    if n_trajectories < 2:
        raise ValueError("Trajectory POD requires at least two trajectories.")
    sample = dataset[0]["target"]
    n_features = sample.numel()
    limit = min(n_trajectories, n_features)
    if max_components is not None:
        limit = min(limit, max_components)
    candidate = min(16, limit)
    while True:
        if n_trajectories < 2 * candidate:
            batches = [list(range(n_trajectories))]
        else:
            n_batches = n_trajectories // candidate
            quotient, remainder = divmod(n_trajectories, n_batches)
            sizes = [
                quotient + (index < remainder)
                for index in range(n_batches)
            ]
            batches = []
            start = 0
            for size in sizes:
                batches.append(list(range(start, start + size)))
                start += size
        loader = DataLoader(
            dataset,
            batch_sampler=batches,
            num_workers=num_workers,
            collate_fn=_collate_pod_trajectories,
        )
        try:
            return fit_incremental_pod(
                (batch.detach().cpu().numpy() for batch in loader),
                variance_threshold=variance_threshold,
                n_components=candidate,
            )
        except ValueError as error:
            if "fit more components" not in str(error) or candidate == limit:
                raise
            candidate = min(2 * candidate, limit)
