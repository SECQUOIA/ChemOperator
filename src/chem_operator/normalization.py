"""Public compatibility facade for tensor normalization utilities."""

from chem_operator._normalization.base import (FieldMode, IdentityNormalizer,
                                               Normalizer, NormalizerState)
from chem_operator._normalization.minmax import MinMaxNormalizer
from chem_operator._normalization.rms import RMSNormalizer
from chem_operator._normalization.serialization import \
    normalizer_from_state_dict
from chem_operator._normalization.zscore import ZScoreNormalizer

ZScoreNormalization = ZScoreNormalizer
RMSNormalization = RMSNormalizer
MinMaxNormalization = MinMaxNormalizer


__all__ = [
    "FieldMode",
    "IdentityNormalizer",
    "MinMaxNormalization",
    "MinMaxNormalizer",
    "Normalizer",
    "NormalizerState",
    "RMSNormalization",
    "RMSNormalizer",
    "ZScoreNormalization",
    "ZScoreNormalizer",
    "normalizer_from_state_dict",
]
