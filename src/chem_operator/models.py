"""Public compatibility facade for model adapters and training utilities."""

from chem_operator._models.adapters import (AutoencoderAdapter,
                                            NeuralOperatorAdapter)
from chem_operator._models.arrays import FNOChannel, OperatorArrays
from chem_operator._models.deeponet import (CheckpointCallback,
                                            CoordinateScaler,
                                            DeepONetBenchmarkConfig,
                                            DeepONetBenchmarkResult,
                                            DeepONetTrainingHistory,
                                            deeponet_parameter_counts,
                                            make_deeponet_dataloader,
                                            relative_l2_loss)
from chem_operator._models.deepxde import DeepXDEAdapter
from chem_operator._models.fno import FNOAdapter
from chem_operator._models.pod import (PODTransform, fit_incremental_pod,
                                       fit_incremental_pod_dataset)
from chem_operator._models.statistics import (fit_fno_zscore_normalizer,
                                              fit_zscore_normalizer)
from chem_operator._models.training import (train_deeponet_lazy,
                                            tune_deeponet_hyperparameters)


def __getattr__(name: str):
    """Load the legacy benchmark runner only for compatibility callers."""

    if name == "run_deepxde_benchmark":
        from chem_operator._models.legacy_benchmarks import run_deepxde_benchmark

        return run_deepxde_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AutoencoderAdapter",
    "CheckpointCallback",
    "CoordinateScaler",
    "DeepONetBenchmarkConfig",
    "DeepONetBenchmarkResult",
    "DeepONetTrainingHistory",
    "DeepXDEAdapter",
    "FNOAdapter",
    "FNOChannel",
    "NeuralOperatorAdapter",
    "OperatorArrays",
    "PODTransform",
    "deeponet_parameter_counts",
    "fit_incremental_pod",
    "fit_incremental_pod_dataset",
    "fit_fno_zscore_normalizer",
    "fit_zscore_normalizer",
    "make_deeponet_dataloader",
    "relative_l2_loss",
    "run_deepxde_benchmark",
    "train_deeponet_lazy",
    "tune_deeponet_hyperparameters",
]
