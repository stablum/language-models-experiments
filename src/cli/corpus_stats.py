"""Generic Click CLI for registered corpus statistics."""

from __future__ import annotations

import click

from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import (
    DEFAULT_CORPUS_NAME,
    corpus_names,
    get_corpus,
)
from src.corpora.stats import print_corpus_report, scan_text_column


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Load a registered corpus and print row, character, "
        "and simple whitespace-token statistics."
    ),
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to scan.",
)
@click.option("--dataset-id", default=None, help="Override the registered Hugging Face dataset ID.")
@click.option("--split", default=None, help="Override the registered dataset split.")
@click.option("--text-column", default=None, help="Override the registered text column.")
@click.option(
    "--streaming",
    is_flag=True,
    help="Stream rows instead of downloading the full dataset first.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=0),
    default=None,
    help="Scan only the first N rows. Useful for quick smoke tests.",
)
@click.option(
    "--top-n-lengths",
    type=click.IntRange(min=0),
    default=5,
    show_default=True,
    help="Show the N longest rows by character count. Use 0 to hide examples.",
)
@click.option(
    "--preview-chars",
    type=click.IntRange(min=0),
    default=120,
    show_default=True,
    help="Characters to show from each longest-row preview.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(TEXT_NORMALIZATION_MODES),
    default=DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before computing stats.",
)
def main(
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    top_n_lengths: int,
    preview_chars: int,
    text_normalization: str,
) -> None:
    corpus_definition = get_corpus(corpus)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column

    dataset = corpus_definition.load(
        dataset_id=resolved_dataset_id,
        split=resolved_split,
        streaming=streaming,
    )

    stats = scan_text_column(
        dataset,
        text_column=resolved_text_column,
        limit=limit,
        top_n_lengths=top_n_lengths,
        preview_chars=preview_chars,
        text_normalization=text_normalization,
    )

    print_corpus_report(
        dataset_label=resolved_dataset_id,
        split=resolved_split,
        mode="streaming" if streaming else "download/cache",
        limit=limit,
        reported_rows=getattr(dataset, "num_rows", None),
        features=getattr(dataset, "features", None),
        stats=stats,
    )
    click.echo(f"Text normalization: {text_normalization}")


if __name__ == "__main__":
    main()
