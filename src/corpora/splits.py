"""Text-corpus adapters for reusable data partitioning."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.corpora import text as corpus_text
from src.ml_core.data import splits as data_splits


def load_partition_texts(
    corpus_definition: Any,
    *,
    dataset_id: str,
    plan: data_splits.DataSplitPlan,
    partition: str,
    streaming: bool,
    text_column: str,
    limit: int | None,
) -> Iterable[str]:
    dataset = corpus_definition.load(
        dataset_id=dataset_id,
        revision=plan.dataset_revision,
        split=plan.source_split,
        streaming=streaming,
    )
    rows = data_splits.iter_partition_rows(
        dataset,
        partition=partition,
        plan=plan,
    )
    return corpus_text.iter_text_column(
        rows,
        text_column=text_column,
        limit=limit,
    )
