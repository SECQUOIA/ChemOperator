"""Shared orchestration for model-owned operator experiments.

Scientific adapters and losses stay with their problem; this module owns run
identity, trial readers, CLI overrides, and portable evaluation arrays.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from dataclasses import replace
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .runner import ExperimentRunner, ExperimentSpec
from .types import EvaluationOutcome
from .tuning import Tuner, TuningConfig, RayRuntimeConfig
from .workflow import WorkflowStages, add_run_arguments, add_workflow_arguments, run_context_from_namespace
from .artifacts import fingerprint_path
from .metrics import StreamingRegressionMetrics


def parse_operator_args(paths, *, epochs, tune_epochs, samples, description=None, parser=None):
    parser = parser or argparse.ArgumentParser(description=description)
    add_workflow_arguments(parser)
    add_run_arguments(parser, default_runs_root=paths.root / "artifacts" / "runs")
    parser.add_argument("--data-dir", type=Path, default=paths.data)
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--tune-epochs", type=int, default=tune_epochs)
    parser.add_argument("--samples", type=int, default=samples)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()
    if args.generate or args.plot:
        parser.error("Use generate_dataset.py or plot.py as separate entry points.")
    if min(args.epochs, args.tune_epochs, args.samples, args.plot_cases) < 1:
        parser.error("Epochs, samples, and plot cases must be positive.")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive.")
    if not args.tune and not args.train:
        parser.error("Select --tune or --train.")
    return args


def operator_spec(args, paths, problem, model, stem, fields, units, coordinates, *, channels=None):
    return ExperimentSpec(
        problem_id=problem, model_id=model, benchmark_protocol_id="operator-cartesian-v1",
        dataset_fingerprints={name: fingerprint_path(args.data_dir / f"{stem}_{split}.h5")
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


def run_operator(args, paths, spec, dataset_factory, trainer_factory, search_space, evaluator,
                 *, seed=42, selection_metric="relative_l2", cpus=2, after_run=None):
    stages = WorkflowStages.from_namespace(args)
    context = run_context_from_namespace(args, problem=spec.problem_id, model=spec.model_id, seed=seed,
                                         provenance={"max_cases": args.max_cases})
    with opened_datasets(dataset_factory, ("test",), args.max_cases) as (test,):
        spec = replace(spec, selected_test_case_ids=range(min(args.plot_cases, len(test))))
    runner = ExperimentRunner(context, spec)

    def trial_data(_context):
        return opened_datasets(dataset_factory, ("train", "valid"), args.max_cases)

    def trial_trainer(config, context):
        return trainer_factory(dict(config, epochs=args.tune_epochs), context)

    tuning = None
    if stages.tune:
        tuning = runner.tune(Tuner(
            trial_trainer, dict(search_space, epochs=args.tune_epochs), dataset_factory=trial_data,
            config=TuningConfig(
                metric=f"best_valid_{selection_metric}", mode="min", num_samples=args.samples,
                max_epochs=args.tune_epochs, grace_period=max(1, args.tune_epochs // 3),
                max_concurrent_trials=1, optuna_seed=seed,
                resources_per_trial={"cpu": cpus, "gpu": int(context.device.type == "cuda")},
                ray_runtime=RayRuntimeConfig(temp_dir=paths.ray.resolve(), num_cpus=cpus,
                    num_gpus=int(context.device.type == "cuda"), object_store_memory=100 * 1024**2,
                    runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(filter(None,
                        (str(paths.root), str(paths.root / "src"), os.environ.get("PYTHONPATH"))))}}),
            )), None, None)
    if stages.train:
        if tuning is not None:
            config = dict(tuning.best_config)
        elif args.train_config == "best":
            config = runner.artifacts.read_best_config()
        else:
            config = json.loads(Path(args.train_config).read_text())
        config["epochs"] = args.epochs
        trainer = trainer_factory(config, context)
        with opened_datasets(dataset_factory, ("train", "valid", "test"), args.max_cases) as data:
            result = runner.run(trainer, data[0], data[1], data[2], config=trainer.config, evaluator=evaluator, tuning=tuning)
        if after_run is not None:
            try:
                after_run(trainer, result)
            except BaseException as error:
                runner.artifacts.mark_failed(error)
                raise
    print(f"Run artifacts: {context.paths.run_dir}")
    return context


def unwrap_dataset(data):
    return unwrap_dataset(data.dataset) if isinstance(data, Subset) else data


def evaluate_fields(trainer, data, context, *, labels, coordinate_names, cases=2, extra_metrics=None):
    """Stream physical-unit scores and save only bounded case reconstructions."""
    dataset = unwrap_dataset(data)
    metrics = StreamingRegressionMetrics()
    per_field = [StreamingRegressionMetrics() for _ in labels]
    arrays = {"case_ids": [], "reference": [], "prediction": []}
    arrays.update({axis: [] for axis in coordinate_names})
    started = time.perf_counter()
    model = trainer.model.to(context.device).eval()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=int(trainer.config["batch_size"])):
            prediction = dataset.denormalize_output(model(batch["x"].to(context.device)).cpu())
            reference = dataset.denormalize_output(batch["y"])
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
    if extra_metrics:
        scores.update(extra_metrics(trainer, data, context))
    return EvaluationOutcome(scores, {**{key: np.asarray(value) for key, value in arrays.items()},
        "labels": np.asarray(labels), "coordinate_names": np.asarray(coordinate_names)}, elapsed)
