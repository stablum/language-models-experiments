"""Generic Click CLI for training SentencePiece tokenizers."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.config import configured_command
from src.cli.data_splits import (
    build_cli_split_plan,
    split_plan_parameter_sections,
    upload_split_plan_artifact,
)
from src.cli.output import stage_title
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    TRAIN_PARTITION,
    load_partition_texts,
    source_split_label,
    split_ratio_label,
)
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
@click.option(
    "--source-split",
    "--split",
    "source_split",
    default=None,
    help=(
        "Restrict the source dataset to one named split before project "
        "train/validation partitioning. Omit to merge all source splits."
    ),
)
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
    "--train-ratio",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    default=DEFAULT_TRAIN_RATIO,
    show_default=True,
    help="Fraction of merged source rows assigned to the reusable training partition.",
)
@click.option(
    "--split-seed",
    type=int,
    default=DEFAULT_SPLIT_SEED,
    show_default=True,
    help="Seed for the reusable deterministic train/validation partition.",
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
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
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
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = get_corpus(corpus)
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_artifact_name = artifact_name or f"{corpus}-sentencepiece-{vocab_size}"
    split_plan = build_cli_split_plan(
        corpus_definition,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    task_id: str | None = None
    task_url: str | None = None

    click.echo(stage_title(1, 1, "Tokenizer training"), color=True)
    with (
        TemporaryDirectory(prefix="lme-tokenizer-") as staging_root,
        start_clearml_run(
            clearml_settings(
                project_name=clearml_project,
                task_name=clearml_task_name,
                config_file=clearml_config_file,
                connectivity_check=clearml_connectivity_check,
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
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.train_sentencepiece",
                    "artifact_store": "clearml",
                },
                "Data": {
                    "corpus": corpus,
                    "dataset_id": resolved_dataset_id,
                    "source_split": source_split_label(resolved_source_split),
                    "training_partition": TRAIN_PARTITION,
                    "text_column": resolved_text_column,
                    "streaming": streaming,
                    "limit": limit,
                    "text_normalization": text_normalization,
                },
                "Tokenizer": {
                    "vocab_size": vocab_size,
                    "artifact_name": resolved_artifact_name,
                    "model_type": model_type,
                    "character_coverage": character_coverage,
                    "hard_vocab_limit": hard_vocab_limit,
                    "max_sentence_length": max_sentence_length,
                },
                **split_plan_parameter_sections(split_plan),
            }
        )

        texts = load_partition_texts(
            corpus_definition,
            dataset_id=resolved_dataset_id,
            plan=split_plan,
            partition=TRAIN_PARTITION,
            streaming=streaming,
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
        upload_split_plan_artifact(
            clearml_run,
            staging_dir=Path(staging_root),
            plan=split_plan,
            metadata={"corpus": corpus, "stage": "tokenizer-training"},
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
    click.echo(f"Source split: {source_split_label(resolved_source_split)}")
    click.echo(f"Training partition: {TRAIN_PARTITION}")
    click.echo(f"Split ratio train/validation: {split_ratio_label(split_plan)}")
    click.echo(f"Split seed: {split_plan.split_seed}")
    click.echo(f"Split ID: {split_plan.split_id}")
    click.echo(f"Text column: {resolved_text_column}")
    click.echo(f"Text normalization: {text_normalization}")
    if limit is not None:
        click.echo(f"Limit: first {limit:,} rows")
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Data split artifact: data-split-plan-json")
    click.echo("Model artifact: sentencepiece-model")
    click.echo("Vocabulary artifact: sentencepiece-vocabulary")


if __name__ == "__main__":
    main()
