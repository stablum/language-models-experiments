"""ClearML PipelineController DAG for reusable tokenizer training."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.pipeline_common import (
    DEFAULT_TOKENIZER_PIPELINE_NAME,
    TOKENIZER_PIPELINE_STAGE_DEPENDENCIES,
    TOKENIZER_PIPELINE_STAGES,
    TOKENIZER_STAGE,
    assert_pipeline_finished_successfully,
    build_pipeline_controller,
    configure_pipeline_control,
    connect_controller_experiment_parameters,
    output_uri_value,
    pipeline_options,
    pipeline_resume_option,
    print_stage_task_ids,
    resume_pipeline_controller_stage,
    stage_gate_callback,
)
from src.cli.pipeline_steps import (
    pipeline_artifact_monitors,
    pipeline_metric_monitors,
)
from src.cli.stage_pipeline_steps import train_tokenizer_stage_entry
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    VALIDATION_PARTITION,
)
from src.tracking.clearml import (
    assert_clearml_endpoints_reachable,
    clearml_options,
    clearml_settings,
    configure_clearml_config_file,
)


TOKENIZER_PIPELINE_CONFIG_SECTION = "tokenizer_pipeline"
TOKENIZER_CONFIG_SECTION = "train_sentencepiece"


def load_tokenizer_pipeline_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml", TOKENIZER_CONFIG_SECTION))
    defaults.update(load_defaults_from_sections((TOKENIZER_PIPELINE_CONFIG_SECTION,)))
    return defaults


@configured_command(
    "tokenizer_pipeline",
    default_loader=load_tokenizer_pipeline_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run reusable SentencePiece tokenizer training as a ClearML Pipeline DAG.",
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
    if pipeline_local and not wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")

    corpus_definition = get_corpus(corpus)
    resolved_pipeline_name = clearml_task_name or pipeline_name
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_artifact_name = artifact_name or f"{corpus}-sentencepiece-{vocab_size}"

    parameter_filters = {
        "corpus": corpus,
        "tokenizer_model_name": resolved_artifact_name,
    }
    if pipeline_controller_id is not None:
        if pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
                "Use --pipeline-queued with --pipeline-controller-id."
            )
        resume_pipeline_controller_stage(
            stage_name=TOKENIZER_STAGE,
            pipeline_controller_id=pipeline_controller_id,
            pipeline_name=resolved_pipeline_name,
            pipeline_version=pipeline_version,
            controller_queue=controller_queue,
            wait=wait,
            clearml_project=clearml_project,
            clearml_task_name=clearml_task_name,
            clearml_config_file=clearml_config_file,
            clearml_connectivity_check=clearml_connectivity_check,
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            parameter_filters=parameter_filters,
            stage_dependencies=TOKENIZER_PIPELINE_STAGE_DEPENDENCIES,
            stage_names=TOKENIZER_PIPELINE_STAGES,
        )
        return

    settings = clearml_settings(
        project_name=clearml_project,
        task_name=resolved_pipeline_name,
        config_file=clearml_config_file,
        connectivity_check=clearml_connectivity_check,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
    )
    resolved_config_file = configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        assert_clearml_endpoints_reachable(resolved_config_file, settings.output_uri)

    pipeline = build_pipeline_controller(
        pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        clearml_tags=settings.tags,
        clearml_output_uri=settings.output_uri,
        add_run_number=add_run_number,
    )
    configure_pipeline_control(
        pipeline.task,
        run_stage=None,
        run_until_stage=None,
        updated_by="tokenizer-pipeline-cli",
    )
    connect_controller_experiment_parameters(
        pipeline.task,
        {
            "corpus": corpus,
            "tokenizer_model_name": resolved_artifact_name,
            "dataset_id": resolved_dataset_id,
            "source_split": resolved_source_split or "",
            "text_column": resolved_text_column,
            "vocab_size": vocab_size,
            "model_type": model_type,
            "text_normalization": text_normalization,
        },
    )
    add_tokenizer_pipeline_step(
        pipeline,
        clearml_project=settings.project_name,
        clearml_output_uri=settings.output_uri,
        clearml_tags=settings.tags,
        clearml_config_file=resolved_config_file if pipeline_local else None,
        execution_queue=None if pipeline_local else execution_queue,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        text_column=resolved_text_column,
        streaming=streaming,
        limit=limit,
        train_ratio=train_ratio,
        split_seed=split_seed,
        vocab_size=vocab_size,
        artifact_name=resolved_artifact_name,
        model_type=model_type,
        character_coverage=character_coverage,
        hard_vocab_limit=hard_vocab_limit,
        max_sentence_length=max_sentence_length,
        text_normalization=text_normalization,
    )

    click.echo(f"ClearML tokenizer pipeline: {settings.project_name}/{resolved_pipeline_name}")
    click.echo(f"Pipeline version: {pipeline_version}")
    click.echo(f"Tokenizer model name: {resolved_artifact_name}")
    click.echo(f"Pipeline controller task ID: {pipeline.task.id}")
    task_url = pipeline.task.get_output_log_web_page()
    if task_url:
        click.echo(f"Pipeline controller URL: {task_url}")
    click.echo(f"Stage tasks: {TOKENIZER_STAGE}")

    if pipeline_local:
        click.echo("Execution mode: local ClearML PipelineController")
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        click.echo(f"Execution mode: queued controller on {controller_queue}")
        if execution_queue is not None:
            click.echo(f"Step execution queue: {execution_queue}")
        pipeline.start(queue=controller_queue, wait=wait)

    click.echo("ClearML tokenizer pipeline submitted.")
    if wait:
        assert_pipeline_finished_successfully(pipeline)
        print_stage_task_ids(
            pipeline.task.id,
            TOKENIZER_PIPELINE_STAGES,
            stage_names=TOKENIZER_PIPELINE_STAGES,
        )
        click.echo("ClearML tokenizer pipeline run completed.")


def add_tokenizer_pipeline_step(
    pipeline: object,
    *,
    clearml_project: str,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    clearml_config_file: Path | None,
    execution_queue: str | None,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    limit: int | None,
    train_ratio: float,
    split_seed: int,
    vocab_size: int,
    artifact_name: str,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    text_normalization: str,
) -> None:
    artifact_monitors = pipeline_artifact_monitors()
    metric_monitors = pipeline_metric_monitors(VALIDATION_PARTITION)
    common_step_kwargs = {
        "clearml_output_uri": clearml_output_uri,
        "clearml_tags": "\n".join(clearml_tags),
        "clearml_config_file": str(clearml_config_file) if clearml_config_file else None,
    }
    step_options = {
        "project_name": clearml_project,
        "execution_queue": execution_queue,
        "output_uri": output_uri_value(clearml_output_uri),
        "auto_connect_frameworks": False,
        "auto_connect_arg_parser": False,
        "pre_execute_callback": stage_gate_callback,
        "tags": list(clearml_tags) if clearml_tags else None,
    }

    pipeline.add_function_step(
        name=TOKENIZER_STAGE,
        function=train_tokenizer_stage_entry,
        function_kwargs={
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "vocab_size": vocab_size,
            "artifact_name": artifact_name,
            "model_type": model_type,
            "character_coverage": character_coverage,
            "hard_vocab_limit": hard_vocab_limit,
            "max_sentence_length": max_sentence_length,
            "text_normalization": text_normalization,
            **common_step_kwargs,
        },
        task_name=TOKENIZER_STAGE,
        task_type="training",
        monitor_artifacts=artifact_monitors[TOKENIZER_STAGE],
        monitor_metrics=metric_monitors[TOKENIZER_STAGE],
        stage=TOKENIZER_STAGE,
        **step_options,
    )


if __name__ == "__main__":
    main()
