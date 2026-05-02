"""Generic Click CLI for evaluating registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.pipeline_common import (
    DEFAULT_PIPELINE_NAME,
    EVALUATION_STAGE,
    TRAINING_PIPELINE_STAGE_DEPENDENCIES,
    TRAINING_PIPELINE_STAGES,
    pipeline_options,
    pipeline_resume_option,
    resume_pipeline_controller_stage,
)
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    PROJECT_PARTITIONS,
    VALIDATION_PARTITION,
)
from src.corpora.registry import (
    DEFAULT_CORPUS_NAME,
    corpus_names,
    get_corpus,
)
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import clearml_options


def load_evaluate_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections(("train",))
    if "model_name" in train_defaults:
        defaults["model_name"] = train_defaults["model_name"]
    if "tokenizer_model_name" in train_defaults:
        defaults["tokenizer_model_name"] = train_defaults["tokenizer_model_name"]
    defaults.update(load_defaults_from_sections(("evaluate",)))
    return defaults


@configured_command(
    "evaluate",
    default_loader=load_evaluate_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Evaluate a registered language model on a registered corpus.",
)
@pipeline_resume_option
@pipeline_options(default_name=DEFAULT_PIPELINE_NAME, default_local=False, default_wait=False)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_names()),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model to evaluate.",
)
@click.option(
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name used by the training pipeline.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
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
    "--evaluation-partition",
    "--evaluation-split",
    type=click.Choice(PROJECT_PARTITIONS),
    default=VALIDATION_PARTITION,
    show_default=True,
    help="Reusable project partition to evaluate.",
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
    corpus_definition = get_corpus(corpus)
    model_definition = get_model(model_name)
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")

    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_tokenizer_model_name = str(tokenizer_model_name or "").strip()
    if not resolved_tokenizer_model_name:
        raise click.ClickException(
            "Evaluation requires --tokenizer-model-name, or tokenizer_model_name in [train]."
        )
    if model_task_id is not None or model_path is not None:
        raise click.ClickException(
            "Evaluation now resumes the canonical ClearML pipeline DAG. "
            "Run train first in the same pipeline instead of passing --model-task-id or --model-path."
        )
    if pipeline_local:
        raise click.ClickException(
            "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
            "Use --pipeline-queued for stage CLIs."
        )
    resume_pipeline_controller_stage(
        stage_name=EVALUATION_STAGE,
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
            "evaluation_partition": evaluation_partition,
        },
        stage_dependencies=TRAINING_PIPELINE_STAGE_DEPENDENCIES,
        stage_names=TRAINING_PIPELINE_STAGES,
    )


if __name__ == "__main__":
    main()
