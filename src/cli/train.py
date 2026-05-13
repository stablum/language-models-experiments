"""Generic Click CLI for training registered language models."""

from __future__ import annotations

import click

from src.cli import stage_resume
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.ml_core.data import splits as data_splits
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import model_training as model_pipeline
from src.corpora import normalization
from src.corpora import registry as corpora_registry
from src.models.core import registry as model_registry


@cli_config.configured_command(
    "train",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Train a registered language model from a registered corpus.",
)
@core_pipeline.pipeline_resume_option
@core_pipeline.pipeline_options(
    default_name=model_pipeline.MODEL_TRAINING_PIPELINE.default_name,
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
    default=data_splits.DEFAULT_TRAIN_RATIO,
    show_default=True,
    help="Fraction of merged source rows assigned to the reusable training partition.",
)
@click.option(
    "--split-seed",
    type=int,
    default=data_splits.DEFAULT_SPLIT_SEED,
    show_default=True,
    help="Seed for the reusable deterministic train/validation partition.",
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
    "--beta-2",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "Recursive interpolation beta for the bigram-vs-unigram branch. "
        "Set with --beta-3 to derive interpolation weights."
    ),
)
@click.option(
    "--beta-3",
    type=click.FloatRange(min=0.0, max=1.0),
    default=None,
    help=(
        "Recursive interpolation beta for the trigram-vs-lower-order branch. "
        "Set with --beta-2 to derive interpolation weights."
    ),
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
@tracking.clearml_options
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
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    beta_2: float | None,
    beta_3: float | None,
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
    resolved_tokenizer_model_name = stage_resume.require_tokenizer_model_name(
        tokenizer_model_name,
        action="Model training",
    )
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    model_hyperparameters = lm_model_options.model_hyperparameters_from(locals())
    stage_resume.reject_pipeline_local(pipeline_local)
    stage_resume.resume_model_training_stage(
        stage_name=lm_def.MODEL_STAGE,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        controller_queue=controller_queue,
        wait=wait,
        pipeline_controller_id=pipeline_controller_id,
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
            "text_column": resolved_text_column,
            "streaming": streaming,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "training_limit": limit,
            **model_hyperparameters,
            "text_normalization": text_normalization,
        },
    )


if __name__ == "__main__":
    main()
