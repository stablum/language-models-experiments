"""Helpers for reading text fields from corpus rows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any


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
) -> Iterator[str]:
    for row in iter_rows(dataset, limit):
        if text_column not in row:
            available = ", ".join(row.keys())
            raise KeyError(
                f"Text column {text_column!r} was not found. Available columns: {available}"
            )

        value = row[text_column]
        yield "" if value is None else str(value)
