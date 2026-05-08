"""Dataset loading for the BabyLM 2026 Strict-Small corpus."""

from __future__ import annotations

from typing import Any

from src.corpora import loading


DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
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
    split: str | None = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    return loading.load_hf_dataset(
        dataset_id,
        split=split,
        streaming=streaming,
    )
