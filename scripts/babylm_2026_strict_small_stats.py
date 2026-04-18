#!/usr/bin/env python3
"""Print basic statistics for the BabyLM 2026 Strict-Small corpus."""

from __future__ import annotations

import argparse

from corpus_stats import print_corpus_report, scan_text_column
from datasets import load_dataset


DATASET_ID = "BabyLM-community/BabyLM-2026-Strict-Small"
SPLIT = "train"
TEXT_COLUMN = "text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the BabyLM 2026 Strict-Small corpus from Hugging Face and "
            "print row, character, and simple whitespace-token statistics."
        )
    )
    parser.add_argument("--dataset-id", default=DATASET_ID, help="Hugging Face dataset ID.")
    parser.add_argument("--split", default=SPLIT, help="Dataset split to load.")
    parser.add_argument(
        "--text-column",
        default=TEXT_COLUMN,
        help="Column containing corpus text.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream rows instead of downloading the full dataset first.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Scan only the first N rows. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--top-n-lengths",
        type=int,
        default=5,
        help="Show the N longest rows by character count. Use 0 to hide examples.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="Characters to show from each longest-row preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = load_dataset(
        args.dataset_id,
        split=args.split,
        streaming=args.streaming,
    )

    stats = scan_text_column(
        dataset,
        text_column=args.text_column,
        limit=args.limit,
        top_n_lengths=args.top_n_lengths,
        preview_chars=args.preview_chars,
    )

    print_corpus_report(
        dataset_label=args.dataset_id,
        split=args.split,
        mode="streaming" if args.streaming else "download/cache",
        limit=args.limit,
        reported_rows=getattr(dataset, "num_rows", None),
        features=getattr(dataset, "features", None),
        stats=stats,
    )


if __name__ == "__main__":
    main()
