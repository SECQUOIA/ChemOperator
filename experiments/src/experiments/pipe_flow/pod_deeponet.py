"""Tune and train a POD-DeepONet for steady pipe-flow profiles."""

import argparse
from functools import partial
import gc
from typing import Any

from experiments.pipe_flow.common import (
    PATHS,
    PIN_MEMORY,
    PROBLEM_ID,
    SEED,
    adapter,
    experiment_spec,
    final_data,
    limited,
    parse_args,
    prepare_normalizer,
    raw_dataset,
    tuning_data,
    tuning_settings,
)
from ray import tune

from chem_operator.experiments import (
    DeepONetTrainer,
    ExperimentRunner,
    Tuner,
    WorkflowStages,
    deeponet_training_config,
    evaluate_deeponet,
    final_epochs,
    resolve_train_config,
    run_context_from_namespace,
    train_final,
)
from chem_operator.models import PODTransform, fit_incremental_pod_dataset
from chem_operator.normalization import ZScoreNormalizer


MODEL_ID = "pod_deeponet"
POD_VARIANCE_THRESHOLD = 0.999


def search_space(pod_components: int) -> dict[str, Any]:
    return {
        "loss": "relative_l2",
        "width": tune.choice([64, 128]),
        "latent_width": pod_components,
        "branch_hidden_layers": tune.choice([2, 3]),
        "trunk_hidden_layers": 0,
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
        "batch_size": tune.choice([64, 128, 256]),
    }


def prepare_pod(normalizer: ZScoreNormalizer, args: argparse.Namespace) -> PODTransform:
    """Fit the POD basis on the selected training cases (``--max-cases``)."""
    raw = raw_dataset(args.data_dir, "train")
    try:
        pod = fit_incremental_pod_dataset(
            adapter(limited(raw, args.max_cases), normalizer),
            variance_threshold=POD_VARIANCE_THRESHOLD,
            num_workers=getattr(args, "num_workers", 0),
        )
    finally:
        raw.close()
    gc.collect()
    return pod


def main() -> None:
    args = parse_args()
    stages = WorkflowStages.from_namespace(args)
    normalizer = prepare_normalizer(args)
    pod = prepare_pod(normalizer, args)
    context = run_context_from_namespace(
        args,
        problem=PROBLEM_ID,
        model=MODEL_ID,
        seed=SEED,
        provenance={"max_cases": args.max_cases},
    )
    runner = ExperimentRunner(context, experiment_spec(MODEL_ID, args))
    tuning = None
    result = None

    def make_trainer(config, context=None):
        """Build the trainer for tuning trials and final training alike."""
        del context
        return DeepONetTrainer(
            deeponet_training_config(config),
            pod=pod,
            pin_memory=PIN_MEMORY,
            checkpoint_metadata={"normalizer": normalizer.state_dict()},
        )

    if stages.tune:
        PATHS.ray.mkdir(parents=True, exist_ok=True)
        tuning = runner.tune(
            Tuner(
                make_trainer,
                search_space(pod.n_components),
                config=tuning_settings(args, context),
                dataset_factory=partial(tuning_data, args=args, normalizer=normalizer),
            ),
            None,
            None,
            experiment_name=f"{PROBLEM_ID}_{MODEL_ID}_{context.paths.run_dir.name}",
        )
        result = tuning

    if stages.train:
        config, source = resolve_train_config(stages.train_config, runner, tuning=tuning)
        config["epochs"] = final_epochs(args)

        def train_one(run_runner, run_context, run_config):
            trainer = make_trainer(run_config, run_context)
            with final_data(args, normalizer=normalizer) as (training, validation, test):
                return run_runner.run(
                    trainer,
                    training,
                    validation,
                    test,
                    config=run_config,
                    tuning=tuning if run_runner is runner else None,
                    evaluator=partial(
                        evaluate_deeponet,
                        normalizer=normalizer,
                        reconstruction_cases=args.plot_cases,
                    ),
                )

        result = train_final(
            runner, stages, config=config, tuning=tuning,
            train_one=train_one, train_config_source=source,
        )

    print(f"{MODEL_ID}: {result}")


if __name__ == "__main__":
    main()
