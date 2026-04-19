"""Generic Click CLI for training token-level bigram models."""

from __future__ import annotations

from pathlib import Path

import click

from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.text import iter_text_column
from src.models.bigram import train_bigram_model


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a simple autoregressive bigram model from a registered corpus.",
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
    "--tokenizer-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="SentencePiece .model file to tokenize the corpus.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output JSON path for the trained bigram model.",
)
@click.option(
    "--smoothing",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="Add-k smoothing value stored with the model.",
)
def main(
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    tokenizer_model: Path | None,
    output: Path | None,
    smoothing: float,
) -> None:
    corpus_definition = get_corpus(corpus)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_tokenizer_model = tokenizer_model or Path(
        "artifacts",
        "tokenizers",
        f"{corpus}-sentencepiece-1000.model",
    )
    resolved_output = output or Path(
        "artifacts",
        "models",
        f"{corpus}-sentencepiece-bigram.json",
    )

    if not resolved_tokenizer_model.exists():
        raise click.ClickException(
            f"Tokenizer model not found: {resolved_tokenizer_model}. "
            "Train it first with src.cli.train_sentencepiece."
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

    summary = train_bigram_model(
        texts,
        tokenizer_model=resolved_tokenizer_model,
        output_path=resolved_output,
        smoothing=smoothing,
    )

    click.echo(f"Corpus: {corpus}")
    click.echo(f"Dataset: {resolved_dataset_id}")
    click.echo(f"Split: {resolved_split}")
    click.echo(f"Text column: {resolved_text_column}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    click.echo(f"Tokenizer: {resolved_tokenizer_model}")
    click.echo(f"Bigram model: {summary.output_path}")
    click.echo(f"Vocabulary size: {summary.vocab_size:,}")
    click.echo(f"Sequences: {summary.sequence_count:,}")
    click.echo(f"Tokens: {summary.token_count:,}")
    click.echo(f"Transitions: {summary.transition_count:,}")


if __name__ == "__main__":
    main()
