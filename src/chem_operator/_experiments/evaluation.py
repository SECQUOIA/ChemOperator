"""Reusable physical-unit evaluators and configuration adapters."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from chem_operator._models.deeponet import DeepONetTrainingConfig
from chem_operator._normalization.base import Normalizer

from .metrics import StreamingRegressionMetrics
from .trainers import DeepONetTrainer
from .types import EvaluationOutcome, RunContext


def deeponet_training_config(
    config: Mapping[str, Any],
    *,
    epoch_multiplier: float = 1.0,
    display_every: int = 10,
) -> DeepONetTrainingConfig:
    if epoch_multiplier <= 0:
        raise ValueError("epoch_multiplier must be positive.")
    return DeepONetTrainingConfig(
        loss=str(config.get("loss", "relative_l2")),
        epochs=max(1, int(int(config["epochs"]) * epoch_multiplier)),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
        batch_size=int(config["batch_size"]),
        width=int(config["width"]),
        latent_width=int(config["latent_width"]),
        branch_hidden_layers=int(config.get("branch_hidden_layers", 2)),
        trunk_hidden_layers=int(config.get("trunk_hidden_layers", 2)),
        activation=str(config["activation"]),
        display_every=display_every,
        seed=int(config["seed"]),
    )


def evaluate_deeponet(
    trainer: DeepONetTrainer,
    data: Any,
    context: RunContext,
    *,
    normalizer: Normalizer,
    reconstruction_cases: int = 2,
) -> EvaluationOutcome:
    if reconstruction_cases < 1:
        raise ValueError("reconstruction_cases must be positive.")
    started = time.perf_counter()
    normalized_prediction = trainer.predict(data, context)
    inference_seconds = time.perf_counter() - started
    normalized_reference = torch.stack(
        [torch.as_tensor(data[index]["target"]) for index in range(len(data))]
    )
    prediction = normalizer.denormalize_flattened(normalized_prediction, "variable")
    reference = normalizer.denormalize_flattened(normalized_reference, "variable")
    metrics = StreamingRegressionMetrics()
    metrics.update(prediction, reference)
    count = min(reconstruction_cases, len(data))
    first = data[0]
    coordinates = np.stack(
        [np.asarray(data[index]["coordinate"]) for index in range(count)]
    )
    return EvaluationOutcome(
        metrics=metrics.compute(),
        reconstructions={
            "case_ids": np.arange(count),
            "coordinates": coordinates,
            "reference": reference[:count].detach().cpu().numpy(),
            "prediction": prediction[:count].detach().cpu().numpy(),
            "labels": np.asarray(tuple(str(value) for value in first["labels"])),
        },
        inference_seconds=inference_seconds,
    )


__all__ = ["deeponet_training_config", "evaluate_deeponet"]
