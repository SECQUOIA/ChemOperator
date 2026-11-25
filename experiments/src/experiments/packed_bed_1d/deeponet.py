"""Tune and train a direct DeepONet for packed-bed profiles."""

from functools import partial
from typing import Any

from experiments.packed_bed_1d.common import (
    PATHS,
    PIN_MEMORY,
    PROBLEM_ID,
    SEED,
    experiment_spec,
    final_data,
    parse_args,
    prepare_normalizer,
    tuning_data,
    tuning_settings,
)
from ray import tune

from chem_operator.experiments import (
    DeepONetTrainer,
    ExperimentRunner,
    WorkflowStages,
    deeponet_training_config,
    deeponet_tuner,
    evaluate_deeponet,
    final_epochs,
    resolve_train_config,
    run_context_from_namespace,
    train_final,
)


MODEL_ID = "deeponet"


def search_space(epochs: int) -> dict[str, Any]:
    return {
        "loss": "relative_l2",
        "width": tune.choice([128, 256, 512]),
        "latent_width": tune.choice([16, 32, 64]),
        "branch_hidden_layers": tune.choice([2, 3, 4]),
        "trunk_hidden_layers": tune.choice([2, 3, 4]),
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": epochs,
        "seed": SEED,
    }


def main() -> None:
    args = parse_args()
    stages = WorkflowStages.from_namespace(args)
    normalizer = prepare_normalizer(args)
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

    if stages.tune:
        PATHS.ray.mkdir(parents=True, exist_ok=True)
        tuning = runner.tune(
            deeponet_tuner(
                search_space=search_space(args.tune_epochs),
                pod=None,
                dataset_factory=partial(tuning_data, args=args, normalizer=normalizer),
                context=context,
                settings=tuning_settings(args),
                pin_memory=PIN_MEMORY,
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
            trainer = DeepONetTrainer(
                deeponet_training_config(run_config),
                pin_memory=PIN_MEMORY,
                checkpoint_metadata={"normalizer": normalizer.state_dict()},
            )
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
