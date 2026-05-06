"""Helpers for reading text fields from corpus rows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from src.corpora.normalization import TextNormalization
from src.corpora.normalization import normalize_text


def iter_rows(
    dataset: Iterable[Mapping[str, Any]],
    limit: int | None,
) -> Iterator[Mapping[str, Any]]:
    for index, row in enumerate(dataset, start=1):
        if limit is not None and index > limit:
            break
        yield row


def iter_text_column(
    dataset: Iterable[Mapping[str, Any]],
    *,
    text_column: str,
    limit: int | None,
    text_normalization: TextNormalization = "none",
) -> Iterator[str]:
    for row in iter_rows(dataset, limit):
        value = text_column_value(row, text_column)
        text = "" if value is None else str(value)
        yield normalize_text(text, text_normalization)


def text_column_value(row: Mapping[str, Any], text_column: str) -> Any:
    if text_column in row:
        return row[text_column]

    value: Any = row
    for part in text_column.split("."):
        if isinstance(value, Mapping) and part in value:
            value = value[part]
            continue

        available = _available_column_paths(row)
        raise KeyError(
            f"Text column {text_column!r} was not found. Available columns: {available}"
        )
    return value


def _available_column_paths(row: Mapping[str, Any]) -> str:
    paths: list[str] = []

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested_value in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                paths.append(child)
                visit(child, nested_value)

    visit("", row)
    return ", ".join(paths)
