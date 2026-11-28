"""Conventional filesystem paths for runnable research examples."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

StrPath = str | PathLike[str]

@dataclass(frozen=True)
class ExamplePaths:
    """Project, dataset, and output paths derived from an example script."""

    root: Path
    example: Path
    output: Path
    data: Path | None

    @property
    def datasets(self) -> Path:
        """Return the project-level dataset directory."""
        return self.root / "datasets"

    @property
    def plots(self) -> Path:
        """Return the project-level plot directory, keyed by problem ID below it."""
        return self.root / "artifacts" / "plots"

    @property
    def ray(self) -> Path:
        """Return the project-level Ray temporary directory."""
        return self.root / ".ray"

    @classmethod
    def from_script(
        cls,
        script_file: StrPath,
        dataset: StrPath | None = None,
        *,
        root: StrPath,
    ) -> "ExamplePaths":
        """Build paths for *script_file* below the caller-supplied project *root*.

        A relative *dataset* is resolved below ``root / "datasets"``.
        """
        script = Path(script_file).resolve()
        root = Path(root).resolve()
        example = script.parent
        dataset_path = None
        if dataset is not None:
            candidate = Path(dataset)
            dataset_path = candidate if candidate.is_absolute() else root / "datasets" / candidate
        return cls(
            root=root,
            example=example,
            output=example / "results" / script.stem,
            data=dataset_path,
        )

    def resolve(self, path: StrPath) -> Path:
        """Resolve a user or configuration path relative to the project root."""
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate
