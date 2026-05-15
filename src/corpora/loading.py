"""Shared Hugging Face dataset loading helpers for corpus modules."""

from __future__ import annotations

from typing import Any

import datasets


def load_hf_dataset(
    dataset_id: str,
    *,
    config: str | None = None,
    revision: str | None = None,
    split: str | None = None,
    streaming: bool = False,
) -> Any:
    args = (dataset_id,) if config is None else (dataset_id, config)
    if split is None:
        return datasets.load_dataset(
            *args,
            revision=revision,
            streaming=streaming,
        )
    return datasets.load_dataset(
        *args,
        revision=revision,
        split=split,
        streaming=streaming,
    )
