"""Dataset loading for the BabyLM 2026 Strict-Small corpus."""

from __future__ import annotations

from typing import Any

from src.corpora import loading


DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
DATASET_REVISION = "c92ab16b4f08858304b0815706065b3354d8fc0a"
DEFAULT_SPLIT = None
AVAILABLE_SPLITS = ("train",)
SPLIT_NOTE = (
    "BabyLM 2026 Strict-Small exposes one source split, train. The project "
    "creates reusable train/validation partitions from the merged source data."
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
