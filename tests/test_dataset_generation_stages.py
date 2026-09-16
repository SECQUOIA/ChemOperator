"""Tests for dataset generation used by model-script ``--generate``."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import chem_operator.datasets as datasets


@pytest.mark.parametrize(
    ("script", "dataset_name"),
    (
        ("scripts/pfr_heat/generate_dataset.py", "pfr_heat"),
        ("scripts/q2d/generate_dataset.py", "q2d_cmr"),
        (
            "scripts/pipe_flow_transient/generate_dataset.py",
            "pipe_flow_transient",
        ),
    ),
)
def test_generate_dataset_is_explicit_and_overwrites_all_splits(
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    dataset_name: str,
) -> None:
    calls: list[tuple] = []

    class RecordingGenerator:
        def __init__(self, simulator: object, output_path: Path) -> None:
            calls.append(("init", simulator, output_path))

        def generate_splits(self, n_cases: int) -> dict[str, list]:
            calls.append(("generate", n_cases))
            return {"train": [], "valid": [], "test": []}

        def save_splits(
            self,
            splits: dict[str, list],
            overwrite: bool = False,
        ) -> None:
            calls.append(("save", splits, overwrite))

    monkeypatch.setattr(
        datasets,
        "SimulationDatasetGenerator",
        RecordingGenerator,
    )
    script_path = Path(__file__).resolve().parents[1] / script
    specification = importlib.util.spec_from_file_location(
        f"test_{dataset_name}_generate_dataset",
        script_path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    assert calls == []
    module.generate_dataset(n_cases=7)

    assert calls[0][0] == "init"
    assert calls[0][2].name == dataset_name
    assert calls[1] == ("generate", 7)
    assert calls[2][0] == "save"
    assert calls[2][2] is True
