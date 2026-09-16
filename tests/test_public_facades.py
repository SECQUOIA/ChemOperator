"""Compatibility checks for the public modules backed by private packages."""

import chem_operator.dataset_processing as processing
import chem_operator.datasets as datasets
import chem_operator.experiments as experiments
import chem_operator.models as models
import chem_operator.normalization as normalization


def test_public_facade_exports_resolve() -> None:
    for module in (processing, datasets, experiments, models, normalization):
        assert module.__all__
        for name in module.__all__:
            assert hasattr(module, name), f"{module.__name__}.{name} is missing"


def test_dataset_convenience_exports_keep_identity() -> None:
    assert datasets.DataProcessor is processing.DataProcessor
    assert datasets.FieldPacker is processing.FieldPacker
    assert datasets.ZScoreNormalizer is normalization.ZScoreNormalizer
    assert (
        datasets.normalizer_from_state_dict
        is normalization.normalizer_from_state_dict
    )


def test_legacy_deeponet_benchmark_exports_are_removed() -> None:
    assert not hasattr(models, "DeepONetBenchmarkConfig")
    assert not hasattr(models, "DeepONetBenchmarkResult")
    assert not hasattr(models, "run_deepxde_benchmark")


def test_variant_specific_deeponet_runner_is_not_public() -> None:
    assert not hasattr(experiments, "run_deeponet_variant")
