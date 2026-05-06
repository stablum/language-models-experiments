"""Generic Click CLI for training registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.ml_core.cli.config import configured_command
from src.pipelines.language_model.definition import (
    pipeline_options,
    pipeline_resume_option,
    resume_pipeline_controller_stage,
)
from src.pipelines.language_model.model_training import MODEL_STAGE, MODEL_TRAINING_PIPELINE
from src.corpora import normalization
from src.corpora import registry as corpora_registry
from src.ml_core.data.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
)
from src.models import registry as model_registry
from src.ml_core.tracking import clearml_options


@configured_command(
    "train",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a registered language model from a registered corpus.",
)
@pipeline_resume_option
@pipeline_options(
    default_name=MODEL_TRAINING_PIPELINE.default_name,
    default_local=False,
    default_wait=False,
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_registry.model_names()),
    default=model_registry.default_model_name(),
    show_default=True,
    help="Registered model to train.",
)
@click.option(
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name used by model training.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpora_registry.corpus_names()),
    default=corpora_registry.default_corpus_name(),
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
    "--tokenizer-task-id",
    default=None,
    help="Deprecated. Training resolves tokenizers by --tokenizer-model-name.",
)
@click.option(
    "--tokenizer-model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Deprecated. Training resolves tokenizers by --tokenizer-model-name.",
)
@click.option(
    "--smoothing",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="Add-k smoothing value for models that use it.",
)
@click.option(
    "--unigram-weight",
    type=click.FloatRange(min=0.0),
    default=0.1,
    show_default=True,
    help="Interpolation weight for unigram probabilities in models that use it.",
)
@click.option(
    "--bigram-weight",
    type=click.FloatRange(min=0.0),
    default=0.3,
    show_default=True,
    help="Interpolation weight for bigram probabilities in models that use it.",
)
@click.option(
    "--trigram-weight",
    type=click.FloatRange(min=0.0),
    default=0.6,
    show_default=True,
    help="Interpolation weight for trigram probabilities in models that use it.",
)
@click.option(
    "--discount",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.75,
    show_default=True,
    help="Absolute discount value for models that use it.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(normalization.TEXT_NORMALIZATION_MODES),
    default=normalization.DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before model training.",
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
    model_name: str,
    tokenizer_model_name: str | None,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    tokenizer_task_id: str | None,
    tokenizer_model: Path | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    text_normalization: str,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = corpora_registry.get_corpus(corpus)
    model_definition = model_registry.get_model(model_name)
    resolved_tokenizer_model_name = str(tokenizer_model_name or "").strip()
    if not resolved_tokenizer_model_name:
        raise click.ClickException(
            "Model training requires --tokenizer-model-name, or tokenizer_model_name in [train]."
        )
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    if tokenizer_task_id is not None or tokenizer_model is not None:
        raise click.ClickException(
            "Model training now resolves tokenizer artifacts from tokenizer-training runs. "
            "Set --tokenizer-model-name instead of passing --tokenizer-task-id or "
            "--tokenizer-model."
        )
    if pipeline_local:
        raise click.ClickException(
            "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
            "Use --pipeline-queued for stage CLIs."
        )
    resume_pipeline_controller_stage(
        stage_name=MODEL_STAGE,
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
            "model": model_definition.name,
            "tokenizer_model_name": resolved_tokenizer_model_name,
            "corpus": corpus,
            "dataset_id": resolved_dataset_id,
            "source_split": resolved_source_split or "",
        },
        stage_dependencies=MODEL_TRAINING_PIPELINE.stage_dependencies,
        stage_names=MODEL_TRAINING_PIPELINE.stages,
    )


if __name__ == "__main__":
    main()
