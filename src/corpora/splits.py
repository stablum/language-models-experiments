"""Text-corpus adapters for reusable data partitioning."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.corpora import text as corpus_text
from src.ml_core.data.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    PROJECT_PARTITIONS,
    SPLIT_METHOD,
    SPLIT_PLAN_ARTIFACT,
    SPLIT_PLAN_SCHEMA_VERSION,
    TRAIN_PARTITION,
    VALIDATION_PARTITION,
    DataSplitPlan,
    assign_partition,
    attach_split_plan_to_json_model,
    build_data_split_plan,
    count_partition_rows,
    data_split_plan_from_payload,
    dataset_row_count,
    is_split_mapping,
    iter_merged_source_rows,
    iter_partition_rows,
    ordered_source_splits,
    partitioned_metric_names,
    read_model_split_plan,
    read_split_plan,
    source_row_counts,
    source_split_label,
    split_plan_clearml_parameters,
    split_ratio_label,
    write_split_plan,
)


def load_partition_texts(
    corpus_definition: Any,
    *,
    dataset_id: str,
    plan: DataSplitPlan,
    partition: str,
    streaming: bool,
    text_column: str,
    limit: int | None,
) -> Iterable[str]:
    dataset = corpus_definition.load(
        dataset_id=dataset_id,
        split=plan.source_split,
        streaming=streaming,
    )
    rows = iter_partition_rows(
        dataset,
        partition=partition,
        plan=plan,
    )
    return corpus_text.iter_text_column(
        rows,
        text_column=text_column,
        limit=limit,
    )
