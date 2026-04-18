"""Dataset loading for the BabyLM 2026 Strict-Small corpus."""

from __future__ import annotations

from typing import Any

from datasets import load_dataset


DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
DEFAULT_SPLIT = "train"
TEXT_COLUMN = "text"


def load_babylm_dataset(
    *,
    dataset_id: str = DATASET_ID,
    split: str = DEFAULT_SPLIT,
    streaming: bool = False,
) -> Any:
    return load_dataset(
        dataset_id,
        split=split,
        streaming=streaming,
    )
