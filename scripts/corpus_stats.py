"""Reusable helpers for scanning text corpora and printing simple stats."""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class LongExample:
    char_count: int
    row_number: int
    token_count: int = field(compare=False)
    preview: str = field(compare=False)


@dataclass
class CorpusStats:
    rows: int = 0
    nonempty_rows: int = 0
    total_chars: int = 0
    total_whitespace_tokens: int = 0
    total_newlines: int = 0
    char_lengths: list[int] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)
    longest_examples: list[LongExample] = field(default_factory=list)

    def add_text(
        self,
        text: str,
        *,
        row_number: int,
        top_n_lengths: int,
        preview_chars: int,
    ) -> None:
        char_count = len(text)
        token_count = len(text.split())

        self.rows += 1
        self.total_chars += char_count
        self.total_whitespace_tokens += token_count
        self.total_newlines += text.count("\n")
        self.char_lengths.append(char_count)
        self.token_lengths.append(token_count)

        if text.strip():
            self.nonempty_rows += 1

        if top_n_lengths <= 0:
            return

        preview = " ".join(text.split())[:preview_chars]
        example = LongExample(
            char_count=char_count,
            row_number=row_number,
            token_count=token_count,
            preview=preview,
        )
        if len(self.longest_examples) < top_n_lengths:
            heapq.heappush(self.longest_examples, example)
        elif example.char_count > self.longest_examples[0].char_count:
            heapq.heapreplace(self.longest_examples, example)


def percentile(sorted_values: list[int], percent: float) -> float:
    if not sorted_values:
        return 0.0

    rank = (len(sorted_values) - 1) * (percent / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (rank - lower)


def mean(total: int, count: int) -> float:
    return total / count if count else 0.0


def iter_rows(
    dataset: Iterable[Mapping[str, Any]],
    limit: int | None,
) -> Iterable[Mapping[str, Any]]:
    for index, row in enumerate(dataset, start=1):
        if limit is not None and index > limit:
            break
        yield row


def scan_text_column(
    dataset: Iterable[Mapping[str, Any]],
    *,
    text_column: str,
    limit: int | None,
    top_n_lengths: int,
    preview_chars: int,
) -> CorpusStats:
    stats = CorpusStats()

    for row_number, row in enumerate(iter_rows(dataset, limit), start=1):
        if text_column not in row:
            available = ", ".join(row.keys())
            raise KeyError(
                f"Text column {text_column!r} was not found. Available columns: {available}"
            )

        value = row[text_column]
        text = "" if value is None else str(value)
        stats.add_text(
            text,
            row_number=row_number,
            top_n_lengths=top_n_lengths,
            preview_chars=preview_chars,
        )

    return stats


def print_distribution(label: str, values: list[int], total: int) -> None:
    sorted_values = sorted(values)
    print(f"{label} per row:")
    print(f"  min:    {sorted_values[0] if sorted_values else 0:,.0f}")
    print(f"  mean:   {mean(total, len(values)):,.2f}")
    print(f"  median: {percentile(sorted_values, 50):,.0f}")
    print(f"  p95:    {percentile(sorted_values, 95):,.0f}")
    print(f"  max:    {sorted_values[-1] if sorted_values else 0:,.0f}")


def print_corpus_report(
    *,
    dataset_label: str,
    split: str | None,
    mode: str | None,
    limit: int | None,
    reported_rows: int | None,
    features: Any | None,
    stats: CorpusStats,
) -> None:
    print(f"Dataset: {dataset_label}")
    if split is not None:
        print(f"Split: {split}")
    if mode is not None:
        print(f"Mode: {mode}")
    if limit is not None:
        print(f"Limit: first {limit:,} rows")
    if reported_rows is not None:
        print(f"Rows reported by dataset: {reported_rows:,}")
    if features is not None:
        print(f"Features: {features}")

    print()
    print(f"Rows scanned: {stats.rows:,}")
    print(f"Non-empty rows: {stats.nonempty_rows:,}")
    print(f"Empty rows: {stats.rows - stats.nonempty_rows:,}")
    print(f"Total characters: {stats.total_chars:,}")
    print(f"Total newlines inside rows: {stats.total_newlines:,}")
    print(f"Total whitespace tokens: {stats.total_whitespace_tokens:,}")
    print()
    print_distribution("Characters", stats.char_lengths, stats.total_chars)
    print()
    print_distribution(
        "Whitespace tokens",
        stats.token_lengths,
        stats.total_whitespace_tokens,
    )

    if stats.longest_examples:
        print()
        print("Longest rows:")
        for example in sorted(stats.longest_examples, reverse=True):
            print(
                f"  row {example.row_number:,}: "
                f"{example.char_count:,} chars, {example.token_count:,} tokens"
            )
            print(f"    {example.preview}")
