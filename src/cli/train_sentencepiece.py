"""Generic Click CLI for training SentencePiece tokenizers."""

from __future__ import annotations

from pathlib import Path

import click

from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.text import iter_text_column
from src.tokenizers.sentencepiece_training import train_sentencepiece


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a SentencePiece tokenizer from a registered corpus.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to train on.",
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
    help="Train on only the first N rows. Useful for smoke tests.",
)
@click.option(
    "--vocab-size",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="SentencePiece vocabulary size.",
)
@click.option(
    "--output-prefix",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output prefix for .model and .vocab files.",
)
@click.option(
    "--model-type",
    type=click.Choice(("unigram", "bpe", "char", "word")),
    default="unigram",
    show_default=True,
    help="SentencePiece model type.",
)
@click.option(
    "--character-coverage",
    type=float,
    default=1.0,
    show_default=True,
    help="Fraction of characters covered by the model.",
)
@click.option(
    "--hard-vocab-limit/--no-hard-vocab-limit",
    default=True,
    show_default=True,
    help="Require SentencePiece to produce exactly vocab-size pieces.",
)
@click.option(
    "--max-sentence-length",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum sentence length passed to SentencePiece.",
)
def main(
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    vocab_size: int,
    output_prefix: Path | None,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
) -> None:
    corpus_definition = get_corpus(corpus)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_output_prefix = output_prefix or Path(
        "artifacts",
        "tokenizers",
        f"{corpus}-sentencepiece-{vocab_size}",
    )

    dataset = corpus_definition.load(
        dataset_id=resolved_dataset_id,
        split=resolved_split,
        streaming=streaming,
    )
    texts = iter_text_column(
        dataset,
        text_column=resolved_text_column,
        limit=limit,
    )

    model_path, vocab_path = train_sentencepiece(
        texts,
        output_prefix=resolved_output_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        hard_vocab_limit=hard_vocab_limit,
        max_sentence_length=max_sentence_length,
    )

    click.echo(f"Corpus: {corpus}")
    click.echo(f"Dataset: {resolved_dataset_id}")
    click.echo(f"Split: {resolved_split}")
    click.echo(f"Text column: {resolved_text_column}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    click.echo(f"Model: {model_path}")
    click.echo(f"Vocabulary: {vocab_path}")


if __name__ == "__main__":
    main()
