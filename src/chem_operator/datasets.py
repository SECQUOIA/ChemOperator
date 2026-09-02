"""Public compatibility facade for simulation and operator datasets."""

from chem_operator._datasets.generation import SimulationDatasetGenerator
from chem_operator._datasets.operator import (ChemOperatorDataset,
                                              raw_steps_to_possible_sample_t0s)
from chem_operator._datasets.records import (CaseParameters, CaseSimulator,
                                             SimulationRecord)
from chem_operator.dataset_processing import (DataProcessor, FieldPacker,
                                              NormalizationConfig,
                                              PackedFieldLayout,
                                              ProcessedDataset,
                                              TargetTransformConfig)
from chem_operator.normalization import (IdentityNormalizer,
                                         MinMaxNormalization, MinMaxNormalizer,
                                         Normalizer, NormalizerState,
                                         RMSNormalization, RMSNormalizer,
                                         ZScoreNormalization, ZScoreNormalizer,
                                         normalizer_from_state_dict)

__all__ = [
    "CaseParameters",
    "CaseSimulator",
    "ChemOperatorDataset",
    "DataProcessor",
    "FieldPacker",
    "IdentityNormalizer",
    "MinMaxNormalization",
    "MinMaxNormalizer",
    "NormalizationConfig",
    "Normalizer",
    "NormalizerState",
    "PackedFieldLayout",
    "ProcessedDataset",
    "RMSNormalization",
    "RMSNormalizer",
    "SimulationDatasetGenerator",
    "SimulationRecord",
    "TargetTransformConfig",
    "ZScoreNormalization",
    "ZScoreNormalizer",
    "normalizer_from_state_dict",
    "raw_steps_to_possible_sample_t0s",
]
