"""Dataset loading for the TinyStories corpus."""

from __future__ import annotations

from typing import Any

from src.corpora import loading


DATASET_ID = "roneneldan/TinyStories"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DEFAULT_SPLIT = None
AVAILABLE_SPLITS = ("train", "validation")
SPLIT_NOTE = (
    "TinyStories exposes train and validation source splits. The project treats "
    "source splits as input shards and creates reusable train/validation "
    "partitions from the selected source rows."
)
TEXT_COLUMN = "text"


def load_dataset(
    *,
    dataset_id: str = DATASET_ID,
    revision: str | None = DATASET_REVISION,
    split: str | None = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    if dataset_id != DATASET_ID and revision == DATASET_REVISION:
        revision = None
    return loading.load_hf_dataset(
        dataset_id,
        revision=revision,
        split=split,
        streaming=streaming,
    )
