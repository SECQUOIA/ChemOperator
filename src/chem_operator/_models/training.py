"""Lazy DeepONet training and Ray trainable entry point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .deeponet import (CheckpointCallback, CoordinateScaler,
                       DeepONetTrainingConfig, DeepONetTrainingHistory,
                       MetricReporter, TensorTransform, _loader_loss, _network,
                       _PODOnlyDeepONet, deeponet_parameter_counts,
                       make_deeponet_dataloader)
from .deepxde import DeepXDEAdapter
from .pod import PODTransform


def train_deeponet_lazy(
    train: DeepXDEAdapter,
    validation: DeepXDEAdapter,
    *,
    config: DeepONetTrainingConfig,
    coordinate_scaler: CoordinateScaler,
    target_transform: TensorTransform | None = None,
    pod: PODTransform | None = None,
    reporter: MetricReporter | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    early_stopping_patience: int | None = None,
    initial_state_dict: Mapping[str, torch.Tensor] | None = None,
    start_epoch: int = 0,
    num_workers: int = 0,
    pin_memory: bool = False,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, DeepONetTrainingHistory]:
    """Train a Cartesian DeepONet from lazy trajectory ``DataLoader`` batches."""

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if start_epoch < 0 or start_epoch >= config.epochs:
        raise ValueError("start_epoch must be in [0, config.epochs).")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive.")
    if config.branch_hidden_layers < 1:
        raise ValueError("branch_hidden_layers must be positive.")
    if pod is None and config.trunk_hidden_layers < 1:
        raise ValueError("trunk_hidden_layers must be positive for DeepONet.")
    if config.loss not in {"relative_l2", "mse"}:
        raise ValueError("loss must be 'relative_l2' or 'mse'.")
    if pod is not None and target_transform is not None:
        raise ValueError("Use pod or target_transform, not both.")
    if train.format != "cartesian_product" or validation.format != train.format:
        raise ValueError("Lazy DeepONet training requires Cartesian adapters.")
    sample = train[0]
    sample_target = sample["target"]
    if pod is not None:
        target_transform = pod.flatten_tensor
    if target_transform is not None:
        sample_target = target_transform(sample_target)
    branch_width = int(sample["branch"].shape[-1])
    trunk_width = int(sample["trunk"].shape[-1])

    import deepxde as dde

    dde.config.set_random_seed(config.seed)
    selected_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if pod is None:
        output_width = int(sample_target.shape[-1])
        model = _network(
            branch_width,
            trunk_width,
            output_width,
            config,
        )
    else:
        model = _PODOnlyDeepONet(branch_width, pod, config)
    model = model.to(selected_device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict, strict=True)
    parameter_counts = deeponet_parameter_counts(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = make_deeponet_dataloader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        coordinate_scaler=coordinate_scaler,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=config.seed,
    )
    valid_loader = make_deeponet_dataloader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        coordinate_scaler=coordinate_scaler,
        target_transform=target_transform,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=config.seed,
    )
    history = DeepONetTrainingHistory([], [], [], config.loss)
    best_valid_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(start_epoch + 1, config.epochs + 1):
        train_loss = _loader_loss(
            model,
            train_loader,
            device=selected_device,
            optimizer=optimizer,
            pin_memory=pin_memory,
            loss_name=config.loss,
        )
        valid_loss = _loader_loss(
            model,
            valid_loader,
            device=selected_device,
            optimizer=None,
            pin_memory=pin_memory,
            loss_name=config.loss,
        )
        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_loss
            history.best_epoch = epoch
            history.best_valid_loss = valid_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.steps.append(epoch)
        history.loss_train.append([train_loss])
        history.loss_test.append([valid_loss])
        metrics: dict[str, float | int] = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "best_valid_loss": best_valid_loss,
            "best_epoch": history.best_epoch or epoch,
            "is_best": int(is_best),
            "epoch": epoch,
            "branch_hidden_layers": config.branch_hidden_layers,
            "trunk_hidden_layers": (
                0 if pod is not None else config.trunk_hidden_layers
            ),
            **parameter_counts,
        }
        if checkpoint_callback is not None:
            checkpoint_callback(model, metrics)
        if reporter is not None:
            reporter(metrics)
        if reporter is None and (
            epoch == 1
            or epoch == config.epochs
            or epoch % max(1, config.display_every) == 0
        ):
            print(
                f"epoch={epoch:5d} train_loss={train_loss:.6e} "
                f"valid_loss={valid_loss:.6e}"
            )
        if (
            early_stopping_patience is not None
            and stale_epochs >= early_stopping_patience
        ):
            break
    if best_state is None:
        raise RuntimeError("DeepONet training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return model, history


def tune_deeponet_hyperparameters(
    config: Mapping[str, Any],
    *,
    train: DeepXDEAdapter,
    validation: DeepXDEAdapter,
    coordinate_scaler: CoordinateScaler,
    pod: PODTransform | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> None:
    """Ray trainable body backed by lazy datasets rather than NumPy arrays."""

    from ray import tune

    training_config = DeepONetTrainingConfig(
        loss=str(config.get("loss", "relative_l2")),
        epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
        width=int(config["width"]),
        latent_width=int(config["latent_width"]),
        branch_hidden_layers=int(config.get("branch_hidden_layers", 2)),
        trunk_hidden_layers=int(config.get("trunk_hidden_layers", 2)),
        activation=str(config["activation"]),
        display_every=1,
        seed=int(config["seed"]),
    )
    train_deeponet_lazy(
        train,
        validation,
        config=training_config,
        coordinate_scaler=coordinate_scaler,
        pod=pod,
        reporter=tune.report,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
