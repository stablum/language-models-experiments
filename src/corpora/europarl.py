"""Dataset loading for the Europarl parallel corpus."""

from __future__ import annotations

from typing import Any

from src.corpora import loading


DATASET_ID = "Helsinki-NLP/europarl"
DATASET_REVISION = "ab45e286aef3fb5780067100cb5a1132b52b7949"
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
    revision: str | None = DATASET_REVISION,
    split: str | None = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    if dataset_id != DATASET_ID and revision == DATASET_REVISION:
        revision = None
    return loading.load_hf_dataset(
        dataset_id,
        config=DEFAULT_CONFIG,
        revision=revision,
        split=split,
        streaming=streaming,
    )
