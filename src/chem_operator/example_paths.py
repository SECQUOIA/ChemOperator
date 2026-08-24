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
    def ray(self) -> Path:
        """Return the project-level Ray temporary directory."""
        return self.root / ".ray"

    @classmethod
    def from_script(
        cls,
        script_file: StrPath,
        dataset: StrPath | None = None,
    ) -> "ExamplePaths":
        """Build paths for *script_file*, locating the project by ``pyproject.toml``."""
        script = Path(script_file).resolve()
        root = cls._find_root(script.parent)
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

    @staticmethod
    def _find_root(start: Path) -> Path:
        for directory in (start, *start.parents):
            if (directory / "pyproject.toml").is_file():
                return directory
        raise FileNotFoundError(
            f"Could not find pyproject.toml above example directory: {start}"
        )
