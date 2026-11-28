"""Data-loading defaults: cache casting, size estimates, and CLI resolution."""

from __future__ import annotations

import argparse
import json
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from chem_operator._experiments import data_loading
from chem_operator._experiments.trainers import _resolve_num_workers
from chem_operator.datasets import ChemOperatorDataset, TrajectoryResamplingDataset
from chem_operator.experiments import (
    OperatorTrainer,
    RunContext,
    add_run_arguments,
    resolve_data_cache,
    run_context_from_namespace,
)


def _write_dataset(path, *, n_cases: int = 3, n_steps: int = 7) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as h5:
        cases = h5.create_group("cases")
        for case_index in range(n_cases):
            case = cases.create_group(f"{case_index:06d}")
            # Irregular, increasing times so resampling interpolates.
            times = np.cumsum(rng.uniform(0.1, 1.0, n_steps))
            times[0] = 0.0
            case.create_group("coordinates").create_dataset("t", data=times)
            fields = case.create_group("fields")
            # Values with no exact float32 form expose any casting difference.
            fields.create_dataset("u", data=rng.normal(size=(n_steps, 3)) / 3.0)
            fields.create_dataset("T", data=900.0 + rng.normal(size=n_steps) / 7.0)
            fields.create_dataset("region", data=np.arange(n_steps, dtype=np.int64))
            constants = case.create_group("constants")
            constants.attrs["k"] = 0.1 * (case_index + 1)
            case.attrs["metadata"] = json.dumps({"record_idx": case_index})


def _dataset(path, **options) -> ChemOperatorDataset:
    defaults = dict(
        task="next_step",
        coordinate_name="t",
        input_fields=("u", "T", "region"),
        output_fields=("u", "T", "region"),
        constant_inputs=("k",),
    )
    return ChemOperatorDataset(path, **{**defaults, **options})


def _assert_same(lazy, cached) -> None:
    if isinstance(lazy, dict):
        assert lazy.keys() == cached.keys()
        for key in lazy:
            _assert_same(lazy[key], cached[key])
    elif isinstance(lazy, torch.Tensor):
        assert lazy.dtype == cached.dtype
        torch.testing.assert_close(lazy, cached, rtol=0, atol=0)
    else:
        assert lazy == cached


def _cache_nbytes(dataset: ChemOperatorDataset) -> int:
    return sum(
        array.nbytes
        for case in dataset._case_cache.values()
        for group in ("fields", "coordinates")
        for array in case[group].values()
    )


@pytest.mark.parametrize(
    ("dtype", "cached_dtype"),
    (
        (torch.float32, np.float32),
        (torch.float64, np.float64),
        (None, None),
        # NumPy has no bfloat16, so the cache keeps the stored dtype.
        (torch.bfloat16, None),
    ),
)
def test_preload_casts_once_and_matches_lazy_samples(tmp_path, dtype, cached_dtype) -> None:
    path = tmp_path / "data.h5"
    _write_dataset(path)
    lazy = _dataset(path, dtype=dtype)
    cached = _dataset(path, dtype=dtype, preload=True)
    try:
        for index in range(len(lazy)):
            _assert_same(lazy[index], cached[index])
        fields = next(iter(cached._case_cache.values()))["fields"]
        if cached_dtype is None:
            assert fields["u"].dtype == np.float64
            assert fields["region"].dtype == np.int64
        else:
            assert {array.dtype for array in fields.values()} == {np.dtype(cached_dtype)}
    finally:
        lazy.close()
        cached.close()


@pytest.mark.parametrize("dtype", (torch.float32, None))
def test_estimated_cache_nbytes_matches_a_warmed_cache(tmp_path, dtype) -> None:
    path = tmp_path / "data.h5"
    _write_dataset(path)
    dataset = _dataset(path, dtype=dtype, preload=True)
    try:
        estimate = dataset.estimated_cache_nbytes()
        assert dataset._case_cache == {}, "estimating must not read data"
        for index in range(len(dataset)):
            dataset[index]
        assert estimate == _cache_nbytes(dataset)
    finally:
        dataset.close()


def test_pickling_a_warmed_dataset_drops_its_cache(tmp_path) -> None:
    path = tmp_path / "data.h5"
    _write_dataset(path)
    dataset = _dataset(path, dtype=torch.float32, preload=True)
    try:
        expected = dataset[0]
        assert dataset._case_cache
        clone = pickle.loads(pickle.dumps(dataset))
        try:
            assert clone._case_cache == {}
            _assert_same(expected, clone[0])
        finally:
            clone.close()
    finally:
        dataset.close()


@pytest.mark.parametrize("subset", (False, True))
def test_resampling_estimate_matches_its_warmed_cache(tmp_path, subset) -> None:
    path = tmp_path / "data.h5"
    _write_dataset(path)
    raw = _dataset(
        path,
        task="operator_cartesian",
        input_fields=("u", "T"),
        output_fields=("u", "T"),
        dtype=torch.float32,
    )
    source = Subset(raw, [2, 0]) if subset else raw
    target = torch.linspace(0.0, 2.0, 11)
    dataset = TrajectoryResamplingDataset(
        source, coordinate_name="t", target_coordinate=target, cache=True
    )
    try:
        estimate = dataset.estimated_cache_nbytes()
        assert dataset._cache == {}
        for index in range(len(dataset)):
            dataset[index]
        actual = 0
        for sample in dataset._cache.values():
            for window in ("input", "output"):
                actual += sum(value.nbytes for value in sample[f"{window}_fields"].values())
                actual += sample[f"{window}_coordinates"]["t"].nbytes
        assert estimate == actual
    finally:
        raw.close()


class _Estimated:
    def __init__(self, nbytes: int, closed: list[str], split: str) -> None:
        self.nbytes = nbytes
        self.closed = closed
        self.split = split

    def estimated_cache_nbytes(self) -> int:
        return self.nbytes

    def close(self) -> None:
        self.closed.append(self.split)


def _args(**values) -> argparse.Namespace:
    return argparse.Namespace(**{"data_cache": "auto", "num_workers": 0, **values})


def _available(monkeypatch, nbytes: int) -> None:
    monkeypatch.setattr(
        data_loading.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=nbytes),
    )


def test_auto_cache_compares_all_splits_with_a_quarter_of_free_ram(monkeypatch) -> None:
    closed: list[str] = []
    factory = lambda split: _Estimated(100, closed, split)  # noqa: E731
    _available(monkeypatch, 4 * 300)

    args = _args()
    assert resolve_data_cache(args, factory) is True
    assert args.cache_data is True
    assert args.data_cache_estimate_bytes == 300
    assert closed == ["train", "valid", "test"]

    _available(monkeypatch, 4 * 300 - 4)
    assert resolve_data_cache(_args(), factory) is False


def test_auto_cache_counts_one_copy_per_worker(monkeypatch) -> None:
    factory = lambda split: _Estimated(100, [], split)  # noqa: E731
    _available(monkeypatch, 4 * 300)
    args = _args(num_workers=2)
    assert resolve_data_cache(args, factory) is False
    assert args.data_cache_estimate_bytes == 600


def test_auto_cache_counts_one_copy_per_concurrent_trial(monkeypatch) -> None:
    """Each running tuning trial opens its own datasets, on top of its workers."""
    factory = lambda split: _Estimated(100, [], split)  # noqa: E731
    _available(monkeypatch, 4 * 1200)
    args = _args(num_workers=2, concurrent_trials=2)
    assert resolve_data_cache(args, factory) is True
    assert args.data_cache_estimate_bytes == 1200
    args = _args(concurrent_trials=3)
    resolve_data_cache(args, factory)
    assert args.data_cache_estimate_bytes == 900


def test_explicit_cache_modes_skip_the_estimate() -> None:
    def factory(split):
        raise AssertionError("on/off must not open datasets")

    on, off = _args(data_cache="on"), _args(data_cache="off")
    assert resolve_data_cache(on, factory) is True
    assert resolve_data_cache(off, factory) is False
    assert on.data_cache_estimate_bytes is None


def test_auto_cache_stays_lazy_when_size_is_unknown(monkeypatch) -> None:
    _available(monkeypatch, 10**12)
    args = _args()
    assert resolve_data_cache(args, lambda split: object()) is False
    assert args.cache_data is False


def test_run_arguments_add_worker_and_cache_flags(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser, default_runs_root=tmp_path)
    defaults = parser.parse_args([])
    assert defaults.num_workers == 0
    assert defaults.data_cache == "auto"
    chosen = parser.parse_args(["--num-workers", "3", "--data-cache", "off"])
    assert (chosen.num_workers, chosen.data_cache) == (3, "off")
    for bad in (["--num-workers", "-1"], ["--data-cache", "maybe"]):
        with pytest.raises(SystemExit):
            parser.parse_args(bad)


def test_run_context_carries_workers_and_records_data_loading(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser, default_runs_root=tmp_path)
    args = parser.parse_args(["--device", "cpu", "--num-workers", "2", "--data-cache", "on"])
    resolve_data_cache(args, lambda split: None)
    context = run_context_from_namespace(args, problem="p", model="m", seed=1)

    assert context.num_workers == 2
    assert context.provenance["data_loading"] == {
        "num_workers": 2,
        "data_cache": "on",
        "cache_data": True,
        "cache_estimate_bytes": None,
    }
    assert context.trial("t0").num_workers == 2
    assert context.seed_child(3).num_workers == 2
    with pytest.raises(ValueError):
        RunContext(paths=tmp_path, seed=0, dtype=torch.float32, device="cpu", num_workers=-1)


def test_trainers_default_to_the_run_worker_count(tmp_path) -> None:
    context = RunContext(
        paths=tmp_path, seed=0, dtype=torch.float32, device="cpu", num_workers=2
    )
    assert _resolve_num_workers(None, context) == 2
    assert _resolve_num_workers(0, context) == 0

    data = TensorDataset(torch.zeros(4, 1))
    inherited = OperatorTrainer(lambda config: None, {"batch_size": 2})
    loader = inherited._loader(data, batch_size=2, shuffle=False, seed=0, context=context)
    assert loader.num_workers == 2 and loader.persistent_workers
    explicit = OperatorTrainer(lambda config: None, {"batch_size": 2}, num_workers=0)
    loader = explicit._loader(data, batch_size=2, shuffle=False, seed=0, context=context)
    assert loader.num_workers == 0
