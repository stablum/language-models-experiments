"""Generic Click CLI for evaluating registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import stage_resume
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.ml_core.data import splits as data_splits
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_training as model_pipeline
from src.corpora import registry as corpora_registry
from src.models.core import registry as model_registry


def load_evaluate_command_defaults(_config_section: str) -> dict[str, object]:
    return stage_resume.load_stage_command_defaults("evaluate")


@cli_config.configured_command(
    "evaluate",
    default_loader=load_evaluate_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Evaluate a registered language model on a registered corpus.",
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
    help="Registered model to evaluate.",
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
    help="Registered corpus to evaluate on.",
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
    help="Evaluate on only the first N rows. Useful for smoke tests.",
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
    "--evaluation-partition",
    "--evaluation-split",
    type=click.Choice(data_splits.PROJECT_PARTITIONS),
    default=data_splits.VALIDATION_PARTITION,
    show_default=True,
    help=(
        "Primary project partition for unpartitioned summary metrics and Optuna "
        "objectives. The evaluation stage evaluates all project partitions."
    ),
)
@click.option(
    "--model-task-id",
    default=None,
    help="Deprecated. Evaluation resumes the model dependency from the pipeline controller.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Deprecated. Evaluation resumes the model dependency from the pipeline controller.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="K value for top-k next-token accuracy.",
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
    evaluation_partition: str,
    model_task_id: str | None,
    model_path: Path | None,
    top_k: int,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    corpus_definition = corpora_registry.get_corpus(corpus)
    model_definition = model_registry.get_model(model_name)
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")

    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_tokenizer_model_name = stage_resume.require_tokenizer_model_name(
        tokenizer_model_name,
        action="Evaluation",
    )
    stage_resume.reject_deprecated_model_dependency(
        model_task_id,
        model_path,
        action="Evaluation",
    )
    stage_resume.reject_pipeline_local(pipeline_local)
    stage_resume.resume_model_training_stage(
        stage_name=lm_def.EVALUATION_STAGE,
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
            "evaluation_partition": evaluation_partition,
        },
    )


if __name__ == "__main__":
    main()
