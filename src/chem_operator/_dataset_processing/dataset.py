"""Lazy processed dataset wrapper."""

from typing import Any

from torch.utils.data import Dataset

from .processor import DataProcessor


class ProcessedDataset(Dataset):
    """Apply a ``DataProcessor`` lazily around another map-style dataset."""

    def __init__(self, raw_dataset: Dataset, processor: DataProcessor):
        self.raw_dataset = raw_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.processor(self.raw_dataset[index])

    def close(self) -> None:
        close = getattr(self.raw_dataset, "close", None)
        if close is not None:
            close()
