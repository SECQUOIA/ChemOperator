"""Public compatibility facade for model adapters and training utilities."""

from chem_operator._models.adapters import (AutoencoderAdapter,
                                            NeuralOperatorAdapter)
from chem_operator._models.arrays import (FNOChannel, ModelDataAdapter,
                                          OperatorArrays, ReferenceSample)
from chem_operator._models.deeponet import (CheckpointCallback,
                                            CoordinateScaler,
                                            DeepONetTrainingConfig,
                                            DeepONetTrainingHistory,
                                            deeponet_parameter_counts,
                                            make_deeponet_dataloader,
                                            relative_l2_loss)
from chem_operator._models.deepxde import DeepONetAdapter, DeepXDEAdapter
from chem_operator._models.fno import FNOAdapter
from chem_operator._models.pod import (PODTransform, fit_incremental_pod,
                                       fit_incremental_pod_dataset)
from chem_operator._models.statistics import (fit_fno_zscore_normalizer,
                                              fit_zscore_normalizer)
from chem_operator._models.training import (train_deeponet_lazy,
                                            tune_deeponet_hyperparameters)
from chem_operator._experiments.trainers import (CompositeLossTrainer,
                                                  DeepONetTrainer, FNOTrainer,
                                                  LossTerm)

__all__ = [
    "AutoencoderAdapter",
    "CheckpointCallback",
    "CoordinateScaler",
    "CompositeLossTrainer",
    "DeepONetTrainer",
    "DeepONetAdapter",
    "DeepONetComparisonResult",
    "DeepONetTrainingConfig",
    "DeepONetTrainingHistory",
    "DeepXDEAdapter",
    "FNOAdapter",
    "FNOChannel",
    "FNOTrainer",
    "LossTerm",
    "NeuralOperatorAdapter",
    "ModelDataAdapter",
    "OperatorArrays",
    "PODTransform",
    "ReferenceSample",
    "deeponet_parameter_counts",
    "fit_incremental_pod",
    "fit_incremental_pod_dataset",
    "fit_fno_zscore_normalizer",
    "fit_zscore_normalizer",
    "make_deeponet_dataloader",
    "relative_l2_loss",
    "train_deeponet_lazy",
    "tune_deeponet_hyperparameters",
]
