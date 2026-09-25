"""Tests for dataset generation used by model-script ``--generate``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import chem_operator.datasets as datasets


class RecordingGenerator(datasets.SimulationDatasetGenerator):
    def __init__(self, output_path: Path) -> None:
        super().__init__(SimpleNamespace(name="tiny"), output_path, seed=7)
        self.generated: list[tuple[str, int, int]] = []

    def generate_split(self, split: str, n_cases: int, seed: int) -> list:
        self.generated.append((split, n_cases, seed))
        return []


def test_generate_missing_splits_preserves_existing_files(tmp_path: Path) -> None:
    generator = RecordingGenerator(tmp_path)
    generator.save_split("train", [], overwrite=False)
    original = (tmp_path / "tiny_train.h5").read_bytes()
    generated = generator.generate_missing_splits(n_cases=10)
    assert set(generated) == {"valid", "test"}
    assert generator.generated == [("valid", 1, 8), ("test", 1, 9)]
    assert (tmp_path / "tiny_train.h5").read_bytes() == original

    generator.generated.clear()
    assert generator.generate_missing_splits(n_cases=10) == {}
    assert generator.generated == []
    assert (tmp_path / "tiny_train.h5").read_bytes() == original


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
