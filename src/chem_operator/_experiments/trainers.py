"""Reusable trainers for DeepONet and Fourier neural operator families."""

from __future__ import annotations

import inspect
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chem_operator._models.deeponet import (
    CoordinateScaler,
    DeepONetTrainingConfig,
    _network,
    _PODOnlyDeepONet,
    _with_output_channel,
    deeponet_parameter_counts,
    make_deeponet_dataloader,
)
from chem_operator._models.pod import PODTransform
from chem_operator._models.training import train_deeponet_lazy

from .device import resolve_device, seed_worker
from .metrics import History, MetricEvent, StreamingRegressionMetrics
from .types import RunContext, TrainingOutcome

Reporter = Callable[[Mapping[str, float | int]], None]
BatchAdapter = Callable[
    [Any, torch.device, torch.dtype],
    tuple[Any, Any, Mapping[str, Any]],
]
LossFunction = Callable[[Any, Any, Mapping[str, Any]], torch.Tensor]
MetricAdapter = Callable[
    [Any, Any, Mapping[str, Any]],
    tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
]
BatchSize = Callable[[Any, Mapping[str, Any]], int]


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters without making model-family assumptions."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=destination.suffix or ".pt",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu").clone()
        for name, value in model.state_dict().items()
    }


def _verify_state_dict(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> None:
    if expected.keys() != actual.keys():
        raise RuntimeError("Reloaded checkpoint has different state-dict keys.")
    for name in expected:
        if not torch.equal(expected[name].cpu(), actual[name].detach().cpu()):
            raise RuntimeError(f"Reloaded checkpoint tensor {name!r} differs.")


class DeepONetTrainer:
    """DeepONet epoch loop, early stopping, checkpointing, and prediction."""

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        config: DeepONetTrainingConfig,
        *,
        coordinate_scaler: CoordinateScaler | None = None,
        pod: PODTransform | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        early_stopping_patience: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        reporter: Reporter | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.coordinate_scaler = coordinate_scaler
        self.pod = pod
        self.target_transform = target_transform
        self.early_stopping_patience = early_stopping_patience
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.reporter = reporter
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.model: torch.nn.Module | None = None
        self._architecture: dict[str, int] | None = None
        self._history: tuple[MetricEvent, ...] = ()

    def fit(
        self,
        train: Any,
        validation: Any,
        context: RunContext,
    ) -> TrainingOutcome:
        device = resolve_device(context.device)
        seed_worker(context.seed)
        if self.config.seed != context.seed:
            raise ValueError(
                "DeepONetTrainingConfig.seed must match RunContext.seed."
            )
        if self.coordinate_scaler is None:
            first = train[0]
            self.coordinate_scaler = CoordinateScaler.fit(
                _numpy(first["trunk"])
            )
        sample = train[0]
        sample_target = sample["target"]
        transform = self.target_transform
        if self.pod is not None:
            transform = self.pod.flatten_tensor
        if transform is not None:
            sample_target = transform(sample_target)
        self._architecture = {
            "branch_width": int(sample["branch"].shape[-1]),
            "trunk_width": int(sample["trunk"].shape[-1]),
            "output_width": int(sample_target.shape[-1]),
        }

        history = History()

        def record(metrics: Mapping[str, float | int]) -> None:
            epoch = int(metrics["epoch"])
            history.record(epoch, "train", "objective", float(metrics["train_loss"]))
            history.record(epoch, "val", "objective", float(metrics["valid_loss"]))
            history.record(
                epoch,
                "val",
                "relative_l2",
                float(metrics["valid_relative_l2"]),
            )
            if self.reporter is not None:
                self.reporter(metrics)

        started = time.perf_counter()
        model, legacy_history = train_deeponet_lazy(
            train,
            validation,
            config=self.config,
            coordinate_scaler=self.coordinate_scaler,
            target_transform=self.target_transform,
            pod=self.pod,
            reporter=record,
            early_stopping_patience=self.early_stopping_patience,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            device=device,
            dtype=context.dtype,
        )
        elapsed = time.perf_counter() - started
        self.model = model
        self._history = history.events
        checkpoint = self.save_checkpoint(context.paths.checkpoint(".pt"))
        expected = _cpu_state_dict(model)
        reloaded = self._model_from_payload(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            device=device,
            dtype=context.dtype,
        )
        _verify_state_dict(expected, reloaded.state_dict())
        self.model = reloaded.eval()
        best_epoch = legacy_history.best_epoch
        if best_epoch is None:
            raise RuntimeError("DeepONet training did not select a best epoch.")
        validation_relative_l2 = next(
            event.value
            for event in reversed(self._history)
            if event.epoch == best_epoch
            and event.split == "val"
            and event.metric == "relative_l2"
        )
        return TrainingOutcome(
            history=self._history,
            best_epoch=best_epoch,
            timings={"final_training_seconds": elapsed},
            checkpoint=checkpoint,
            metadata={
                **deeponet_parameter_counts(self.model),
                "config": asdict(self.config),
                "checkpoint_reload_verified": True,
            },
            metrics={"validation_relative_l2": validation_relative_l2},
        )

    def predict(self, data: Any, context: RunContext) -> torch.Tensor:
        if self.model is None or self.coordinate_scaler is None:
            raise RuntimeError("Fit or load a DeepONet checkpoint before prediction.")
        device = resolve_device(context.device)
        self.model.to(device=device, dtype=context.dtype).eval()
        loader = make_deeponet_dataloader(
            data,
            batch_size=self.config.batch_size,
            shuffle=False,
            coordinate_scaler=self.coordinate_scaler,
            target_transform=self.target_transform,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            seed=context.seed,
        )
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                branch = batch["branch"].to(device=device, dtype=context.dtype)
                trunk = batch["trunk"].to(device=device, dtype=context.dtype)
                prediction = self.model((branch, trunk))
                if self.pod is not None:
                    prediction = self.pod.unflatten_tensor(prediction)
                else:
                    prediction = _with_output_channel(
                        prediction,
                        int(batch["target"].shape[-1]),
                    )
                predictions.append(prediction.detach().cpu())
        if not predictions:
            raise RuntimeError("Prediction loader produced no batches.")
        return torch.cat(predictions)

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.model is None or self.coordinate_scaler is None or self._architecture is None:
            raise RuntimeError("Fit or load a DeepONet before saving a checkpoint.")
        pod_state = None
        if self.pod is not None:
            pod_state = {
                "mean": torch.as_tensor(self.pod.mean),
                "basis": torch.as_tensor(self.pod.basis),
                "explained_variance_ratio": torch.as_tensor(
                    self.pod.explained_variance_ratio
                ),
                "cumulative_explained_variance": (
                    self.pod.cumulative_explained_variance
                ),
                "output_shape": list(self.pod.output_shape),
            }
        payload = {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "trainer": "deeponet",
            "config": asdict(self.config),
            "architecture": self._architecture,
            "coordinate_scaler": {
                "minimum": torch.as_tensor(self.coordinate_scaler.minimum),
                "span": torch.as_tensor(self.coordinate_scaler.span),
            },
            "pod": pod_state,
            "metadata": self.checkpoint_metadata,
            "state_dict": _cpu_state_dict(self.model),
        }
        return _atomic_torch_save(payload, path)

    def load_checkpoint(self, path: str | Path, context: RunContext) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.config = DeepONetTrainingConfig.from_dict(payload["config"])
        scaler = payload["coordinate_scaler"]
        self.coordinate_scaler = CoordinateScaler(
            minimum=_numpy(scaler["minimum"]),
            span=_numpy(scaler["span"]),
        )
        self.pod = _pod_from_state(payload.get("pod"))
        self.checkpoint_metadata = dict(payload.get("metadata", {}))
        self._architecture = {
            name: int(value) for name, value in payload["architecture"].items()
        }
        self.model = self._model_from_payload(
            payload,
            device=resolve_device(context.device),
            dtype=context.dtype,
        ).eval()

    @staticmethod
    def _model_from_payload(
        payload: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        if payload.get("checkpoint_version") != DeepONetTrainer.CHECKPOINT_VERSION:
            raise ValueError("Unsupported DeepONet checkpoint version.")
        config = DeepONetTrainingConfig.from_dict(payload["config"])
        architecture = payload["architecture"]
        pod = _pod_from_state(payload.get("pod"))
        if pod is None:
            model = _network(
                int(architecture["branch_width"]),
                int(architecture["trunk_width"]),
                int(architecture["output_width"]),
                config,
            )
        else:
            model = _PODOnlyDeepONet(int(architecture["branch_width"]), pod, config)
        model = model.to(device=device, dtype=dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _pod_from_state(state: Mapping[str, Any] | None) -> PODTransform | None:
    if state is None:
        return None
    return PODTransform(
        mean=_numpy(state["mean"]),
        basis=_numpy(state["basis"]),
        explained_variance_ratio=_numpy(state["explained_variance_ratio"]),
        cumulative_explained_variance=float(
            state["cumulative_explained_variance"]
        ),
        output_shape=tuple(int(value) for value in state["output_shape"]),
    )


@dataclass(frozen=True, slots=True)
class LossTerm:
    """One named, weighted component of a composite FNO objective."""

    name: str
    function: LossFunction
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("loss term name must be non-empty.")
        if self.weight < 0 or not np.isfinite(self.weight):
            raise ValueError("loss term weight must be finite and non-negative.")


def mse_loss_term(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch: Mapping[str, Any],
) -> torch.Tensor:
    del batch
    return torch.nn.functional.mse_loss(prediction, target)


def default_fno_batch_adapter(
    batch: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, torch.Tensor, Mapping[str, Any]]:
    """Adapt the standard ``FNOAdapter`` mapping to trainer inputs."""
    if not isinstance(batch, Mapping) or "x" not in batch or "y" not in batch:
        raise TypeError("Default FNO batches must be mappings containing x and y.")
    inputs = batch["x"].to(device=device, dtype=dtype)
    target = batch["y"].to(device=device, dtype=dtype)
    moved = {
        name: (
            value.to(device=device, dtype=dtype)
            if isinstance(value, torch.Tensor) and value.is_floating_point()
            else value.to(device=device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in batch.items()
    }
    return inputs, target, moved


def default_metric_adapter(
    prediction: Any,
    target: Any,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    """Select tensor predictions and targets for shared regression metrics."""
    del batch
    if not isinstance(prediction, (torch.Tensor, np.ndarray)) or not isinstance(
        target, (torch.Tensor, np.ndarray)
    ):
        raise TypeError(
            "Non-tensor outputs require an explicit metric_adapter that returns "
            "physical prediction and target tensors."
        )
    return prediction, target


def default_batch_size(target: Any, batch: Mapping[str, Any]) -> int:
    """Infer a batch dimension from the first tensor-like target value."""
    del batch
    leaf = _first_tensor(target)
    if leaf.ndim == 0:
        raise ValueError("A target tensor must include a batch dimension.")
    return int(leaf.shape[0])


def _first_tensor(value: Any) -> torch.Tensor | np.ndarray:
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    elif isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    raise TypeError("No tensor-like value was found.")


def _call_model(model: torch.nn.Module, inputs: Any) -> Any:
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, Mapping):
        return model(**inputs)
    return model(inputs)


def _detach_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {name: _detach_cpu(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_cpu(item) for item in value]
    raise TypeError(
        "FNO predictions must be tensors or nested mappings/sequences of tensors."
    )


def _concatenate_batches(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(values)
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values):
            raise ValueError("Prediction mappings have inconsistent keys.")
        return {
            name: _concatenate_batches([value[name] for value in values])
            for name in keys
        }
    if isinstance(first, tuple):
        if any(len(value) != len(first) for value in values):
            raise ValueError("Prediction tuples have inconsistent lengths.")
        return tuple(
            _concatenate_batches([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        if any(len(value) != len(first) for value in values):
            raise ValueError("Prediction lists have inconsistent lengths.")
        return [
            _concatenate_batches([value[index] for value in values])
            for index in range(len(first))
        ]
    raise TypeError(f"Cannot concatenate prediction type {type(first).__name__}.")


class FNOTrainer:
    """Model-factory-driven FNO trainer with optional composite loss terms."""

    CHECKPOINT_VERSION = 1

    def __init__(  # pylint: disable=too-many-arguments
        self,
        model_factory: Callable[[Mapping[str, Any]], torch.nn.Module],
        config: Mapping[str, Any],
        *,
        loss_terms: Sequence[LossTerm] | None = None,
        batch_adapter: BatchAdapter = default_fno_batch_adapter,
        prediction_transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        metric_adapter: MetricAdapter = default_metric_adapter,
        batch_size: BatchSize = default_batch_size,
        selection_metric: str = "relative_l2",
        early_stopping_patience: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        reporter: Reporter | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_factory = model_factory
        self.config = dict(config)
        self.loss_terms = tuple(loss_terms or (LossTerm("data_loss", mse_loss_term),))
        if not self.loss_terms:
            raise ValueError("FNOTrainer requires at least one loss term.")
        if early_stopping_patience is not None and early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        self.batch_adapter = batch_adapter
        self.prediction_transform = prediction_transform
        self.target_transform = target_transform
        self.metric_adapter = metric_adapter
        self.batch_size = batch_size
        if not selection_metric:
            raise ValueError("selection_metric must be non-empty.")
        self.selection_metric = selection_metric
        self.early_stopping_patience = early_stopping_patience
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.reporter = reporter
        self.checkpoint_metadata = dict(checkpoint_metadata or {})
        self.model: torch.nn.Module | None = None
        self._history: tuple[MetricEvent, ...] = ()

    def fit(  # pylint: disable=too-many-locals
        self,
        train: Any,
        validation: Any,
        context: RunContext,
    ) -> TrainingOutcome:
        epochs = int(self.config.get("epochs", 0))
        batch_size = int(self.config.get("batch_size", 0))
        if epochs < 1 or batch_size < 1:
            raise ValueError("FNO config requires positive epochs and batch_size.")
        device = resolve_device(context.device)
        seed_worker(context.seed)
        model = self.model_factory(self.config).to(
            device=device,
            dtype=context.dtype,
        )
        self.model = model
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["learning_rate"]),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )
        train_loader = self._loader(
            train,
            batch_size=batch_size,
            shuffle=True,
            seed=context.seed,
        )
        validation_loader = self._loader(
            validation,
            batch_size=batch_size,
            shuffle=False,
            seed=context.seed,
        )
        history = History()
        best_metric = float("inf")
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            train_metrics = self._epoch(
                train_loader,
                device=device,
                dtype=context.dtype,
                optimizer=optimizer,
            )
            validation_metrics = self._epoch(
                validation_loader,
                device=device,
                dtype=context.dtype,
                optimizer=None,
            )
            for name, value in train_metrics.items():
                history.record(epoch, "train", name, value)
            for name, value in validation_metrics.items():
                history.record(epoch, "val", name, value)
            if self.selection_metric not in validation_metrics:
                raise KeyError(
                    f"Validation metrics do not contain selection metric "
                    f"{self.selection_metric!r}."
                )
            selected = validation_metrics[self.selection_metric]
            is_best = selected < best_metric
            if is_best:
                best_metric = selected
                best_epoch = epoch
                best_state = _cpu_state_dict(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
            report = {
                "epoch": epoch,
                "train_objective": train_metrics["objective"],
                "valid_objective": validation_metrics["objective"],
                f"valid_{self.selection_metric}": selected,
                f"best_valid_{self.selection_metric}": best_metric,
                "best_epoch": best_epoch,
                "is_best": int(is_best),
                "n_params": count_parameters(model),
            }
            if "relative_l2" in validation_metrics:
                report["valid_relative_l2"] = validation_metrics["relative_l2"]
            if self.reporter is not None:
                self.reporter(report)
            if (
                self.early_stopping_patience is not None
                and stale_epochs >= self.early_stopping_patience
            ):
                break
        elapsed = time.perf_counter() - started
        if best_state is None:
            raise RuntimeError("FNO training did not produce a valid checkpoint.")
        model.load_state_dict(best_state, strict=True)
        model.eval()
        self._history = history.events
        checkpoint = self.save_checkpoint(context.paths.checkpoint(".pt"))
        reloaded = self._model_from_payload(
            torch.load(checkpoint, map_location="cpu", weights_only=True),
            device=device,
            dtype=context.dtype,
        )
        _verify_state_dict(best_state, reloaded.state_dict())
        self.model = reloaded.eval()
        return TrainingOutcome(
            history=self._history,
            best_epoch=best_epoch,
            timings={"final_training_seconds": elapsed},
            checkpoint=checkpoint,
            metadata={
                "n_params": count_parameters(self.model),
                "config": dict(self.config),
                "loss_terms": [term.name for term in self.loss_terms],
                "selection_metric": self.selection_metric,
                "checkpoint_reload_verified": True,
            },
            metrics={f"validation_{self.selection_metric}": best_metric},
        )

    def _epoch(
        self,
        loader: DataLoader,
        *,
        device: torch.device,
        dtype: torch.dtype,
        optimizer: torch.optim.Optimizer | None,
    ) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("FNO model has not been constructed.")
        training = optimizer is not None
        self.model.train(training)
        totals = {term.name: 0.0 for term in self.loss_terms}
        objective_total = 0.0
        samples = 0
        shared = StreamingRegressionMetrics()
        gradient_context = torch.enable_grad() if training else torch.no_grad()
        with gradient_context:
            for raw_batch in loader:
                inputs, target, batch = self.batch_adapter(raw_batch, device, dtype)
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                prediction = _call_model(self.model, inputs)
                count = self.batch_size(target, batch)
                objective = _first_tensor(prediction).new_zeros(())
                for term in self.loss_terms:
                    value = term.function(prediction, target, batch)
                    if value.ndim != 0:
                        value = value.mean()
                    objective = objective + term.weight * value
                    totals[term.name] += float(value.detach()) * count
                if optimizer is not None:
                    objective.backward()
                    optimizer.step()
                objective_total += float(objective.detach()) * count
                samples += count
                metric_prediction = (
                    prediction
                    if self.prediction_transform is None
                    else self.prediction_transform(prediction)
                )
                metric_target = (
                    target
                    if self.target_transform is None
                    else self.target_transform(target)
                )
                metric_prediction, metric_target = self.metric_adapter(
                    metric_prediction,
                    metric_target,
                    batch,
                )
                shared.update(metric_prediction, metric_target)
        if samples == 0:
            raise RuntimeError("FNO DataLoader produced no samples.")
        return {
            "objective": objective_total / samples,
            **{name: value / samples for name, value in totals.items()},
            **shared.compute(),
        }

    def _loader(
        self,
        data: Any,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        return DataLoader(
            data,
            batch_size=min(batch_size, len(data)),
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.pin_memory,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )

    def predict(self, data: Any, context: RunContext) -> Any:
        if self.model is None:
            raise RuntimeError("Fit or load an FNO checkpoint before prediction.")
        device = resolve_device(context.device)
        self.model.to(device=device, dtype=context.dtype).eval()
        loader = self._loader(
            data,
            batch_size=int(self.config["batch_size"]),
            shuffle=False,
            seed=context.seed,
        )
        predictions: list[Any] = []
        with torch.no_grad():
            for raw_batch in loader:
                inputs, _, _ = self.batch_adapter(raw_batch, device, context.dtype)
                predictions.append(_detach_cpu(_call_model(self.model, inputs)))
        if not predictions:
            raise RuntimeError("Prediction loader produced no batches.")
        return _concatenate_batches(predictions)

    def save_checkpoint(self, path: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Fit or load an FNO before saving a checkpoint.")
        return _atomic_torch_save(
            {
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "trainer": "fno",
                "config": dict(self.config),
                "metadata": self.checkpoint_metadata,
                "state_dict": _cpu_state_dict(self.model),
            },
            path,
        )

    def load_checkpoint(self, path: str | Path, context: RunContext) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.config = dict(payload["config"])
        self.checkpoint_metadata = dict(payload.get("metadata", {}))
        self.model = self._model_from_payload(
            payload,
            device=resolve_device(context.device),
            dtype=context.dtype,
        ).eval()

    def _model_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.nn.Module:
        if payload.get("checkpoint_version") != self.CHECKPOINT_VERSION:
            raise ValueError("Unsupported FNO checkpoint version.")
        model = self.model_factory(payload["config"]).to(device=device, dtype=dtype)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class CompositeLossTrainer(FNOTrainer):
    """Descriptive alias for an FNO trainer configured with named loss terms."""


def accepts_context(factory: Callable[..., Any]) -> bool:
    """Return whether a trainer factory explicitly accepts a run context."""
    parameters = inspect.signature(factory).parameters.values()
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == "context"
        for parameter in parameters
    )
