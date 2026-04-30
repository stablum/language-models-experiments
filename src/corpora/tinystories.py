"""Dataset loading for the TinyStories corpus."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset


DATASET_ID = "roneneldan/TinyStories"
DEFAULT_SPLIT = None
AVAILABLE_SPLITS = ("train", "validation")
SPLIT_NOTE = (
    "TinyStories exposes train and validation source splits. The project treats "
    "source splits as input shards and creates reusable train/validation "
    "partitions from the selected source rows."
)
TEXT_COLUMN = "text"


def load_tinystories_dataset(
    *,
    dataset_id: str = DATASET_ID,
    split: str | None = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    if split is None:
        return load_dataset(dataset_id, streaming=streaming)
    return load_dataset(
        dataset_id,
        split=split,
        streaming=streaming,
    )
