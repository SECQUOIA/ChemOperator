"""Shared orchestration for model-owned operator experiments.

Scientific adapters and losses stay with their problem; this module owns run
identity, trial readers, CLI overrides, and portable evaluation arrays.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .runner import ExperimentRunner, ExperimentSpec
from .seeds import resolve_train_config, train_final
from .types import EvaluationOutcome
from .tuning import Tuner, RayRuntimeConfig, build_tuning_config
from .workflow import (
    WorkflowStages, add_run_arguments, add_workflow_arguments, final_epochs, parse_workflow_args,
    run_context_from_namespace, tuning_resources,
)
from .artifacts import fingerprint_path
from .metrics import StreamingRegressionMetrics
from chem_operator._models.deeponet import relative_l2_loss


def parse_operator_args(paths, *, tune_epochs, samples, description=None, parser=None):
    """Parse the standard operator CLI with the problem's tuning budget.

    Problems set only the tuning defaults. ``--epochs`` is an optional override of
    final-training epochs; read the resolved value with :func:`final_epochs`.
    """
    parser = parser or argparse.ArgumentParser(description=description)
    add_workflow_arguments(parser)
    add_run_arguments(parser, default_runs_root=paths.root / "artifacts" / "runs")
    parser.add_argument("--data-dir", type=Path, default=paths.data)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Final-training epochs (default: 2 x --tune-epochs).")
    parser.add_argument("--tune-epochs", type=int, default=tune_epochs)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parse_workflow_args(parser)
    if (args.epochs is not None and args.epochs < 1) or min(args.tune_epochs, args.samples) < 1:
        parser.error("Epochs and samples must be positive.")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive.")
    return args


def operator_spec(args, paths, problem, model, fields, units, coordinates, *, channels=None):
    return ExperimentSpec(
        problem_id=problem, model_id=model, benchmark_protocol_id="operator-cartesian-v1",
        dataset_fingerprints={name: fingerprint_path(args.data_dir / f"{problem}_{split}.h5")
                              for name, split in (("train", "train"), ("validation", "valid"), ("test", "test"))},
        fields=fields, channels=channels or fields, units=units, coordinates=coordinates,
        selected_test_case_ids=range(min(args.plot_cases, args.max_cases or args.plot_cases)),
        tuning_budget={"samples": args.samples, "epochs": args.tune_epochs}, project_root=paths.root,
    )


@contextmanager
def opened_datasets(factory, splits, maximum=None):
    opened = []
    try:
        datasets = []
        for split in splits:
            raw, data = factory(split)
            opened.append(raw)
            datasets.append(Subset(data, range(min(len(data), maximum))) if maximum else data)
        yield tuple(datasets)
    finally:
        for raw in opened:
            raw.close()


def operator_ray_runtime(paths, *, cpus, gpus, runtime_env=None):
    """Local Ray runtime for operator tuning, with its temporary directory under ``paths.ray``.

    Workers run in the driver's environment and import installed packages, so no
    ``runtime_env`` is needed locally. A caller targeting a remote cluster may pass
    one (for example Ray's ``working_dir``, ``py_modules`` or ``pip``); it is used unchanged.
    """
    return RayRuntimeConfig(temp_dir=paths.ray.resolve(), num_cpus=cpus, num_gpus=gpus,
                            object_store_memory=100 * 1024**2, runtime_env=runtime_env)


def run_operator(args, paths, spec, dataset_factory, trainer_factory, search_space, evaluator,
                 *, seed=42, selection_metric="relative_l2", after_run=None, runtime_env=None):
    """Tune, train, and evaluate one operator model under the canonical artifact contract.

    Tuning trials and final training both build their trainer with
    ``trainer_factory(config, context)``. The library sets ``config["seed"]`` to the
    context seed and ``config["epochs"]`` to ``--tune-epochs`` in trials and to
    :func:`final_epochs` in final training, so ``search_space`` sets neither.
    Tuning resources come from the ``--cpus-per-trial``, ``--gpus-per-trial``,
    ``--concurrent-trials`` and ``--time-budget`` flags. ``runtime_env`` is passed to
    Ray unchanged (for example for a remote cluster).
    """
    stages = WorkflowStages.from_namespace(args)
    context = run_context_from_namespace(args, problem=spec.problem_id, model=spec.model_id, seed=seed,
                                         provenance={"max_cases": args.max_cases})
    with opened_datasets(dataset_factory, ("test",), args.max_cases) as (test,):
        spec = replace(spec, selected_test_case_ids=range(min(args.plot_cases, len(test))))
    runner = ExperimentRunner(context, spec)

    def trial_data(_context):
        return opened_datasets(dataset_factory, ("train", "valid"), args.max_cases)

    def trial_trainer(config, *, context):
        return trainer_factory(config, context)

    tuning = None
    if stages.tune:
        tuner = Tuner(trial_trainer, search_space, dataset_factory=trial_data, config=build_tuning_config(
            metric=f"best_valid_{selection_metric}", samples=args.samples, epochs=args.tune_epochs,
            seed=context.seed, device=context.device, **tuning_resources(args),
            temp_dir=paths.ray, runtime_env=runtime_env))
        tuning = runner.tune(tuner, None, None)
    if stages.train:
        config, source = resolve_train_config(stages.train_config, runner, tuning=tuning)
        config["epochs"] = final_epochs(args)

        def train_one(run_runner, run_context, run_config):
            trainer = trainer_factory(dict(run_config, seed=run_context.seed), run_context)
            with opened_datasets(dataset_factory, ("train", "valid", "test"), args.max_cases) as data:
                result = run_runner.run(trainer, data[0], data[1], data[2],
                                        config=trainer.resolved_config(), evaluator=evaluator,
                                        tuning=tuning if run_runner is runner else None)
            if after_run is not None:
                try:
                    after_run(trainer, result)
                except BaseException as error:
                    run_runner.artifacts.mark_failed(error)
                    raise
            return result

        train_final(runner, stages, config=config, tuning=tuning, train_one=train_one,
                    train_config_source=source)
    print(f"Run artifacts: {context.paths.run_dir}")
    return context


def unwrap_dataset(data):
    return unwrap_dataset(data.dataset) if isinstance(data, Subset) else data


def evaluate_fields(trainer, data, context, *, labels, coordinate_names, cases=2, extra_metrics=None,
                    predict=None, batch_size=None, normalized_relative_l2=False):
    """Stream physical-unit scores and save only bounded case reconstructions.

    ``predict(trainer, data, context)`` may supply the complete normalized
    channel-first prediction in one timed call instead of batched model
    forwards. ``normalized_relative_l2`` adds the mean per-case relative L2
    in normalized space, overall and per label.
    """
    dataset = unwrap_dataset(data)
    metrics = StreamingRegressionMetrics()
    per_field = [StreamingRegressionMetrics() for _ in labels]
    arrays = {"case_ids": [], "reference": [], "prediction": []}
    arrays.update({axis: [] for axis in coordinate_names})
    per_case, per_case_fields = [], []
    if batch_size is None:
        batch_size = int(trainer.config["batch_size"])
    started = time.perf_counter()
    # DeepXDE can change PyTorch's default device; keep freshly read references on CPU.
    with torch.no_grad(), torch.device("cpu"):
        if predict is None:
            model = trainer.model.to(context.device).eval()
            predicted, offset = None, 0
        else:
            predicted, offset = predict(trainer, data, context).detach().cpu(), 0
            if len(predicted) != len(data):
                raise ValueError(f"predict returned {len(predicted)} cases for {len(data)} references.")
        for batch in DataLoader(data, batch_size=batch_size):
            target = batch["y"]
            if predicted is None:
                normalized = model(batch["x"].to(context.device)).cpu()
            else:
                normalized = predicted[offset:offset + len(target)]
                offset += len(target)
            if normalized_relative_l2:
                per_case.append(relative_l2_loss(normalized, target, reduction="none", eps=1.0e-8))
                per_case_fields.append(relative_l2_loss(
                    normalized.flatten(0, 1), target.flatten(0, 1), reduction="none", eps=1.0e-8,
                ).reshape(target.shape[:2]))
            prediction = dataset.denormalize_output(normalized)
            reference = dataset.denormalize_output(target)
            metrics.update(prediction, reference)
            for index, metric in enumerate(per_field):
                metric.update(prediction[:, index], reference[:, index])
            take = min(cases - len(arrays["case_ids"]), len(reference))
            for index in range(max(0, take)):
                arrays["case_ids"].append(len(arrays["case_ids"]))
                arrays["reference"].append(reference[index].numpy())
                arrays["prediction"].append(prediction[index].numpy())
                for axis in coordinate_names:
                    arrays[axis].append(batch[axis][index].numpy())
    elapsed = time.perf_counter() - started
    scores = metrics.compute()
    for label, metric in zip(labels, per_field):
        scores.update({f"{label}/{key}": value for key, value in metric.compute().items()})
    if normalized_relative_l2:
        scores["normalized_relative_l2"] = float(torch.cat(per_case).mean())
        fields = torch.cat(per_case_fields).mean(dim=0)
        scores.update({f"normalized_relative_l2_{label}": float(fields[index])
                       for index, label in enumerate(labels)})
    if extra_metrics:
        scores.update(extra_metrics(trainer, data, context))
    return EvaluationOutcome(scores, {**{key: np.asarray(value) for key, value in arrays.items()},
        "labels": np.asarray(labels), "coordinate_names": np.asarray(coordinate_names)}, elapsed)
