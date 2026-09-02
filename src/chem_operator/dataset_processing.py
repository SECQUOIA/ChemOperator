"""Public compatibility facade for structured dataset processing."""

from chem_operator._dataset_processing.config import (NormalizationConfig,
                                                      PackedFieldLayout,
                                                      TargetTransformConfig)
from chem_operator._dataset_processing.dataset import ProcessedDataset
from chem_operator._dataset_processing.packing import FieldPacker
from chem_operator._dataset_processing.processor import DataProcessor

__all__ = [
    "DataProcessor",
    "FieldPacker",
    "NormalizationConfig",
    "PackedFieldLayout",
    "ProcessedDataset",
    "TargetTransformConfig",
]
