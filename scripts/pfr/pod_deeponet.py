"""Tune and train a POD-DeepONet for PFR-chain trajectories."""

from functools import partial
import gc
from typing import Any

from common import (
    DATALOADER_WORKERS,
    MAX_EPOCHS,
    PATHS,
    PIN_MEMORY,
    PROBLEM_ID,
    RECONSTRUCTION_CASES,
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
    validate_stages,
)
from ray import tune

from chem_operator.experiments import (
    DeepONetTrainer,
    ExperimentRunner,
    WorkflowStages,
    deeponet_training_config,
    deeponet_tuner,
    evaluate_deeponet,
    run_context_from_namespace,
)
from chem_operator.models import PODTransform, fit_incremental_pod_dataset
from chem_operator.normalization import ZScoreNormalizer


MODEL_ID = "pod_deeponet"
POD_VARIANCE_THRESHOLD = 0.9995


def search_space(pod_components: int) -> dict[str, Any]:
    return {
        "loss": "relative_l2",
        "width": tune.choice([512, 768, 1024, 1280]),
        "latent_width": pod_components,
        "branch_hidden_layers": tune.choice([2, 3, 4]),
        "trunk_hidden_layers": 0,
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(5e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": 16,
        "epochs": MAX_EPOCHS,
        "seed": SEED,
    }


def prepare_pod(normalizer: ZScoreNormalizer) -> PODTransform:
    raw = raw_dataset("train")
    try:
        pod = fit_incremental_pod_dataset(
            adapter(limited(raw), normalizer),
            variance_threshold=POD_VARIANCE_THRESHOLD,
            num_workers=DATALOADER_WORKERS,
        )
    finally:
        raw.close()
    gc.collect()
    return pod


def main() -> None:
    args = parse_args()
    stages = WorkflowStages.from_namespace(args)
    validate_stages(stages)
    normalizer = prepare_normalizer()
    pod = prepare_pod(normalizer)
    context = run_context_from_namespace(
        args, problem=PROBLEM_ID, model=MODEL_ID, seed=SEED
    )
    runner = ExperimentRunner(context, experiment_spec(MODEL_ID))
    tuning = None
    result = None

    if stages.tune:
        PATHS.ray.mkdir(parents=True, exist_ok=True)
        tuning = runner.tune(
            deeponet_tuner(
                search_space=search_space(pod.n_components),
                pod=pod,
                dataset_factory=partial(tuning_data, normalizer=normalizer),
                context=context,
                settings=tuning_settings(),
                num_workers=DATALOADER_WORKERS,
                pin_memory=PIN_MEMORY,
            ),
            None,
            None,
            experiment_name=f"{PROBLEM_ID}_{MODEL_ID}_{context.paths.run_dir.name}",
        )
        result = tuning

    if stages.train:
        config = (
            dict(tuning.best_config)
            if tuning is not None
            else runner.artifacts.read_best_config()
        )
        trainer = DeepONetTrainer(
            deeponet_training_config(config, epoch_multiplier=2.0),
            pod=pod,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
            checkpoint_metadata={"normalizer": normalizer.state_dict()},
        )
        with final_data(normalizer=normalizer) as (training, validation, test):
            result = runner.run(
                trainer,
                training,
                validation,
                test,
                config=config,
                tuning=tuning,
                evaluator=partial(
                    evaluate_deeponet,
                    normalizer=normalizer,
                    reconstruction_cases=RECONSTRUCTION_CASES,
                ),
            )

    print(f"{MODEL_ID}: {result}")


if __name__ == "__main__":
    main()
