"""Generic Click CLI for training SentencePiece tokenizers."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.pipeline_common import (
    DEFAULT_TOKENIZER_PIPELINE_NAME,
    TOKENIZER_PIPELINE_STAGE_DEPENDENCIES,
    TOKENIZER_PIPELINE_STAGES,
    TOKENIZER_STAGE,
    pipeline_options,
    pipeline_resume_option,
    resume_pipeline_controller_stage,
)
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
)
from src.tracking.clearml import clearml_options


def load_train_sentencepiece_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml", "train_sentencepiece"))
    defaults.update(load_defaults_from_sections(("tokenizer_pipeline", "tokenizer_training")))
    return defaults


@configured_command(
    "train_sentencepiece",
    default_loader=load_train_sentencepiece_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a SentencePiece tokenizer from a registered corpus.",
)
@pipeline_resume_option
@pipeline_options(default_name=DEFAULT_TOKENIZER_PIPELINE_NAME)
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
    pipeline_name: str,
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    pipeline_controller_id: str | None,
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
    if pipeline_controller_id is not None:
        if pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
                "Use --pipeline-queued with --pipeline-controller-id."
            )
        resume_pipeline_controller_stage(
            stage_name=TOKENIZER_STAGE,
            pipeline_controller_id=pipeline_controller_id,
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            controller_queue=controller_queue,
            wait=wait,
            clearml_project=clearml_project,
            clearml_task_name=clearml_task_name,
            clearml_config_file=clearml_config_file,
            clearml_connectivity_check=clearml_connectivity_check,
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            parameter_filters={
                "corpus": corpus,
                "tokenizer_model_name": resolved_artifact_name,
                "dataset_id": resolved_dataset_id,
                "source_split": resolved_source_split or "",
            },
            stage_dependencies=TOKENIZER_PIPELINE_STAGE_DEPENDENCIES,
            stage_names=TOKENIZER_PIPELINE_STAGES,
        )
        return

    from src.cli.tokenizer_pipeline import main as tokenizer_pipeline_command

    tokenizer_pipeline_command.callback(
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_local=pipeline_local,
        controller_queue=controller_queue,
        execution_queue=execution_queue,
        wait=wait,
        add_run_number=add_run_number,
        pipeline_controller_id=None,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        train_ratio=train_ratio,
        split_seed=split_seed,
        text_column=resolved_text_column,
        streaming=streaming,
        limit=limit,
        vocab_size=vocab_size,
        artifact_name=resolved_artifact_name,
        model_type=model_type,
        character_coverage=character_coverage,
        hard_vocab_limit=hard_vocab_limit,
        max_sentence_length=max_sentence_length,
        text_normalization=text_normalization,
        clearml_project=clearml_project,
        clearml_task_name=clearml_task_name,
        clearml_config_file=clearml_config_file,
        clearml_connectivity_check=clearml_connectivity_check,
        clearml_output_uri=clearml_output_uri,
        clearml_tags=clearml_tags,
    )


if __name__ == "__main__":
    main()
