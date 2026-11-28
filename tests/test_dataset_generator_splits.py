"""SimulationDatasetGenerator writes only the splits that are missing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
