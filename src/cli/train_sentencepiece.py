"""Generic Click CLI for training SentencePiece tokenizers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.config import configured_command
from src.cli.output import stage_title
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.text import iter_text_column
from src.tracking.clearml import clearml_options, clearml_settings, start_clearml_run
from src.tokenizers.sentencepiece_training import train_sentencepiece


@configured_command(
    "train_sentencepiece",
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
    "--artifact-name",
    default=None,
    help="Base name for the tokenizer artifacts stored in ClearML.",
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
@click.option(
    "--text-normalization",
    type=click.Choice(TEXT_NORMALIZATION_MODES),
    default=DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before tokenizer training.",
)
@clearml_options
def main(
    corpus: str,
    dataset_id: str | None,
    split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    vocab_size: int,
    artifact_name: str | None,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    text_normalization: str,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = get_corpus(corpus)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_split = split or corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_artifact_name = artifact_name or f"{corpus}-sentencepiece-{vocab_size}"
    task_id: str | None = None
    task_url: str | None = None

    click.echo(stage_title(1, 1, "Tokenizer training"))
    with (
        TemporaryDirectory(prefix="lme-tokenizer-") as staging_root,
        start_clearml_run(
            clearml_settings(
                project_name=clearml_project,
                task_name=clearml_task_name,
                config_file=clearml_config_file,
                output_uri=clearml_output_uri,
                tags=clearml_tags,
            ),
            default_task_name=f"train sentencepiece {corpus} vocab-{vocab_size}",
            task_type="training",
        ) as clearml_run,
    ):
        resolved_output_prefix = Path(staging_root, resolved_artifact_name)
        task_id = clearml_run.task_id
        task_url = clearml_run.task_url
        clearml_run.connect_parameters(
            {
                "command": "src.cli.train_sentencepiece",
                "artifact_store": "clearml",
                "corpus": corpus,
                "dataset_id": resolved_dataset_id,
                "split": resolved_split,
                "text_column": resolved_text_column,
                "streaming": streaming,
                "limit": limit,
                "vocab_size": vocab_size,
                "artifact_name": resolved_artifact_name,
                "model_type": model_type,
                "character_coverage": character_coverage,
                "hard_vocab_limit": hard_vocab_limit,
                "max_sentence_length": max_sentence_length,
                "text_normalization": text_normalization,
            }
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
            text_normalization=text_normalization,
        )

        clearml_run.log_metrics(
            "Tokenizer training",
            {
                "vocab_size": vocab_size,
                "character_coverage": character_coverage,
                "hard_vocab_limit": hard_vocab_limit,
                "limit": limit,
            },
        )
        clearml_run.upload_artifact(
            "sentencepiece-model",
            model_path,
            metadata={"corpus": corpus, "vocab_size": vocab_size},
        )
        clearml_run.upload_artifact(
            "sentencepiece-vocabulary",
            vocab_path,
            metadata={"corpus": corpus, "vocab_size": vocab_size},
        )
        clearml_run.register_model(
            name=model_path.stem,
            model_path=model_path,
            framework="custom",
            tags=("tokenizer", corpus),
            comment="SentencePiece tokenizer model.",
        )

    click.echo(f"Corpus: {corpus}")
    click.echo(f"Dataset: {resolved_dataset_id}")
    click.echo(f"Split: {resolved_split}")
    click.echo(f"Text column: {resolved_text_column}")
    click.echo(f"Text normalization: {text_normalization}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Model artifact: sentencepiece-model")
    click.echo("Vocabulary artifact: sentencepiece-vocabulary")


if __name__ == "__main__":
    main()
