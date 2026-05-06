"""Dataset loading for the Europarl parallel corpus."""

from __future__ import annotations

from typing import Any

import datasets


DATASET_ID = "Helsinki-NLP/europarl"
DEFAULT_CONFIG = "en-fr"
DEFAULT_SPLIT = None
AVAILABLE_SPLITS = ("train",)
SPLIT_NOTE = (
    "Europarl exposes one source split, train, for each language-pair "
    "configuration. The project creates reusable train/validation partitions "
    "from that source split."
)
TEXT_COLUMN = "translation.en"


def load_dataset(
    *,
    dataset_id: str = DATASET_ID,
    split: str | None = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    if split is None:
        return datasets.load_dataset(
            dataset_id,
            DEFAULT_CONFIG,
            streaming=streaming,
        )
    return datasets.load_dataset(
        dataset_id,
        DEFAULT_CONFIG,
        split=split,
        streaming=streaming,
    )
