"""Resolve machine-dependent data-loading choices for experiments."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

import psutil

DATA_CACHE_MODES = ("auto", "on", "off")
DEFAULT_RAM_FRACTION = 0.25


def _megabytes(value: int) -> str:
    return f"{value / 1024**2:.0f} MB"


def estimate_cache_nbytes(
    factory: Callable[[str], Any],
    splits: Sequence[str],
) -> int:
    """Sum ``estimated_cache_nbytes()`` over lazily opened splits.

    ``factory(split)`` must return a dataset built without caching. Each one
    is closed after its estimate, so no data stays in memory.
    """
    total = 0
    for split in splits:
        dataset = factory(split)
        try:
            total += int(dataset.estimated_cache_nbytes())
        finally:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
    return total


def resolve_data_cache(
    args: argparse.Namespace,
    factory: Callable[[str], Any],
    splits: Sequence[str] = ("train", "valid", "test"),
    *,
    ram_fraction: float = DEFAULT_RAM_FRACTION,
) -> bool:
    """Decide whether datasets keep their arrays in memory for this run.

    ``--data-cache on`` and ``off`` are used as given. ``auto`` caches when the
    estimated bytes of every split opened together, multiplied by
    ``max(1, --num-workers)`` because each DataLoader worker fills its own
    cache and by ``--concurrent-trials`` because each running tuning trial opens
    its own datasets, fit within ``ram_fraction`` of the RAM available now.

    The result is stored on ``args`` as ``cache_data`` (with the estimate as
    ``data_cache_estimate_bytes``) so dataset factories and the run manifest
    can read it.
    """
    mode = getattr(args, "data_cache", "auto")
    if mode not in DATA_CACHE_MODES:
        raise ValueError(f"--data-cache must be one of {DATA_CACHE_MODES}.")
    if not 0 < ram_fraction <= 1:
        raise ValueError("ram_fraction must be in (0, 1].")

    estimate: int | None = None
    if mode == "auto":
        copies = max(1, int(getattr(args, "num_workers", 0)))
        copies *= max(1, int(getattr(args, "concurrent_trials", 1)))
        try:
            estimate = estimate_cache_nbytes(factory, splits) * copies
        except (AttributeError, TypeError, OSError, KeyError) as error:
            print(f"data cache: auto -> off (could not estimate size: {error})")
            enabled = False
        else:
            budget = int(psutil.virtual_memory().available * ram_fraction)
            enabled = estimate <= budget
            print(
                f"data cache: auto -> {'on' if enabled else 'off'} "
                f"(est. {_megabytes(estimate)}; budget {_megabytes(budget)})"
            )
    else:
        enabled = mode == "on"
        print(f"data cache: {mode}")

    args.cache_data = enabled
    args.data_cache_estimate_bytes = estimate
    return enabled


def data_loading_provenance(args: argparse.Namespace) -> dict[str, Any]:
    """Return the data-loading facts to record in a run manifest."""
    return {
        "num_workers": int(getattr(args, "num_workers", 0)),
        "data_cache": getattr(args, "data_cache", None),
        "cache_data": getattr(args, "cache_data", None),
        "cache_estimate_bytes": getattr(args, "data_cache_estimate_bytes", None),
    }
