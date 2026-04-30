"""Reusable project-level train/validation partitioning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.corpora.text import iter_text_column


DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SPLIT_SEED = 42
TRAIN_PARTITION = "train"
VALIDATION_PARTITION = "validation"
PROJECT_PARTITIONS = (TRAIN_PARTITION, VALIDATION_PARTITION)
SPLIT_METHOD = "deterministic_blake2b_source_split_row_index"
SPLIT_PLAN_SCHEMA_VERSION = 1
SPLIT_PLAN_ARTIFACT = "data-split-plan-json"


@dataclass(frozen=True)
class DataSplitPlan:
    split_id: str
    corpus: str
    dataset_id: str
    source_split: str | None
    source_splits: tuple[str, ...]
    train_ratio: float
    validation_ratio: float
    split_seed: int
    split_method: str = SPLIT_METHOD
    schema_version: int = SPLIT_PLAN_SCHEMA_VERSION

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_id": self.split_id,
            "split_method": self.split_method,
            "corpus": self.corpus,
            "dataset_id": self.dataset_id,
            "source_split": self.source_split,
            "source_splits": list(self.source_splits),
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "split_seed": self.split_seed,
            "partitions": list(PROJECT_PARTITIONS),
        }


def build_data_split_plan(
    *,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    source_splits: Iterable[str],
    train_ratio: float,
    split_seed: int,
) -> DataSplitPlan:
    normalized_source_splits = tuple(source_splits)
    validation_ratio = round(1.0 - train_ratio, 12)
    identity = {
        "schema_version": SPLIT_PLAN_SCHEMA_VERSION,
        "split_method": SPLIT_METHOD,
        "corpus": corpus,
        "dataset_id": dataset_id,
        "source_split": source_split,
        "source_splits": list(normalized_source_splits),
        "train_ratio": train_ratio,
        "validation_ratio": validation_ratio,
        "split_seed": split_seed,
    }
    split_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return DataSplitPlan(
        split_id=split_id,
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        source_splits=normalized_source_splits,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
    )


def data_split_plan_from_payload(payload: Mapping[str, Any]) -> DataSplitPlan | None:
    if int(payload.get("schema_version", 0)) != SPLIT_PLAN_SCHEMA_VERSION:
        return None

    try:
        split_id = str(payload["split_id"])
        corpus = str(payload["corpus"])
        dataset_id = str(payload["dataset_id"])
        source_split_value = payload.get("source_split")
        source_split = None if source_split_value is None else str(source_split_value)
        source_splits = tuple(str(item) for item in payload.get("source_splits", ()))
        train_ratio = float(payload["train_ratio"])
        validation_ratio = float(payload["validation_ratio"])
        split_seed = int(payload["split_seed"])
        split_method = str(payload.get("split_method", SPLIT_METHOD))
    except (KeyError, TypeError, ValueError):
        return None

    if split_method != SPLIT_METHOD:
        return None

    return DataSplitPlan(
        split_id=split_id,
        corpus=corpus,
        dataset_id=dataset_id,
        source_split=source_split,
        source_splits=source_splits,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        split_seed=split_seed,
        split_method=split_method,
    )


def write_split_plan(path: Path, plan: DataSplitPlan) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan.to_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_split_plan(path: Path) -> DataSplitPlan | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    return data_split_plan_from_payload(payload)


def read_model_split_plan(model_path: Path) -> DataSplitPlan | None:
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    data_split = payload.get("data_split")
    if not isinstance(data_split, Mapping):
        return None
    return data_split_plan_from_payload(data_split)


def attach_split_plan_to_json_model(model_path: Path, plan: DataSplitPlan) -> None:
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return

    payload["data_split"] = plan.to_payload()
    model_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def split_plan_clearml_parameters(plan: DataSplitPlan) -> dict[str, object]:
    return {
        "split_id": plan.split_id,
        "split_method": plan.split_method,
        "source_split": source_split_label(plan.source_split),
        "source_splits": list(plan.source_splits),
        "train_ratio": plan.train_ratio,
        "validation_ratio": plan.validation_ratio,
        "split_seed": plan.split_seed,
        "train_partition": TRAIN_PARTITION,
        "validation_partition": VALIDATION_PARTITION,
    }


def source_split_label(source_split: str | None) -> str:
    if source_split is None:
        return "all available source splits"
    return source_split


def split_ratio_label(plan: DataSplitPlan) -> str:
    return f"{plan.train_ratio:.3f}/{plan.validation_ratio:.3f}"


def partitioned_metric_names(
    metrics: Mapping[str, object],
    *,
    partition: str,
) -> dict[str, object]:
    return {
        f"{partition}/{name}": value
        for name, value in metrics.items()
    }


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
    return iter_text_column(
        rows,
        text_column=text_column,
        limit=limit,
    )


def count_partition_rows(
    dataset: Any,
    *,
    partition: str,
    plan: DataSplitPlan,
    limit: int | None,
) -> int | None:
    if partition not in PROJECT_PARTITIONS:
        raise ValueError(f"Unknown data partition: {partition}")
    if limit == 0:
        return 0

    source_counts = source_row_counts(dataset, plan=plan)
    if source_counts is None:
        return limit

    selected_count = 0
    for source_split, row_count in source_counts:
        for row_index in range(row_count):
            row_partition = assign_partition(
                source_split=source_split,
                row_index=row_index,
                train_ratio=plan.train_ratio,
                split_seed=plan.split_seed,
            )
            if row_partition != partition:
                continue

            selected_count += 1
            if limit is not None and selected_count >= limit:
                return selected_count
    return selected_count


def iter_partition_rows(
    dataset: Any,
    *,
    partition: str,
    plan: DataSplitPlan,
) -> Iterator[Mapping[str, Any]]:
    if partition not in PROJECT_PARTITIONS:
        raise ValueError(f"Unknown data partition: {partition}")

    for source_split, row_index, row in iter_merged_source_rows(dataset, plan=plan):
        row_partition = assign_partition(
            source_split=source_split,
            row_index=row_index,
            train_ratio=plan.train_ratio,
            split_seed=plan.split_seed,
        )
        if row_partition == partition:
            yield row


def iter_merged_source_rows(
    dataset: Any,
    *,
    plan: DataSplitPlan,
) -> Iterator[tuple[str, int, Mapping[str, Any]]]:
    if is_split_mapping(dataset):
        for source_split in ordered_source_splits(dataset, plan.source_splits):
            for row_index, row in enumerate(dataset[source_split]):
                yield source_split, row_index, row
        return

    source_split = plan.source_split or "dataset"
    for row_index, row in enumerate(dataset):
        yield source_split, row_index, row


def source_row_counts(
    dataset: Any,
    *,
    plan: DataSplitPlan,
) -> tuple[tuple[str, int], ...] | None:
    if is_split_mapping(dataset):
        counts: list[tuple[str, int]] = []
        for source_split in ordered_source_splits(dataset, plan.source_splits):
            row_count = dataset_row_count(dataset[source_split])
            if row_count is None:
                return None
            counts.append((source_split, row_count))
        return tuple(counts)

    row_count = dataset_row_count(dataset)
    if row_count is None:
        return None
    return ((plan.source_split or "dataset", row_count),)


def dataset_row_count(dataset: Any) -> int | None:
    num_rows = getattr(dataset, "num_rows", None)
    if isinstance(num_rows, int):
        return num_rows

    try:
        return len(dataset)
    except TypeError:
        return None


def is_split_mapping(dataset: Any) -> bool:
    return isinstance(dataset, Mapping) and all(
        hasattr(value, "__iter__")
        for value in dataset.values()
    )


def ordered_source_splits(
    dataset: Mapping[str, Any],
    preferred_order: Iterable[str],
) -> tuple[str, ...]:
    available = tuple(str(key) for key in dataset.keys())
    ordered: list[str] = [
        split
        for split in preferred_order
        if split in dataset
    ]
    ordered.extend(
        split
        for split in sorted(available)
        if split not in ordered
    )
    return tuple(ordered)


def assign_partition(
    *,
    source_split: str,
    row_index: int,
    train_ratio: float,
    split_seed: int,
) -> str:
    key = f"{split_seed}\0{source_split}\0{row_index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    score = int.from_bytes(digest, byteorder="big", signed=False) / float(1 << 64)
    if score < train_ratio:
        return TRAIN_PARTITION
    return VALIDATION_PARTITION
