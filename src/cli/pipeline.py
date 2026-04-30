"""Mandatory ClearML PipelineController DAG for end-to-end experiments."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli.config import configured_command
from src.cli.pipeline_common import (
    DEFAULT_PIPELINE_NAME,
    EVALUATION_STAGE,
    MODEL_STAGE,
    PIPELINE_STAGES,
    QUERY_STAGE,
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
from src.cli.stage_pipeline_steps import (
    evaluate_stage_entry,
    query_stage_entry,
    train_model_stage_entry,
    train_tokenizer_stage_entry,
)
from src.corpora.normalization import DEFAULT_TEXT_NORMALIZATION, TEXT_NORMALIZATION_MODES
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names, get_corpus
from src.corpora.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    PROJECT_PARTITIONS,
    VALIDATION_PARTITION,
)
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import (
    assert_clearml_endpoints_reachable,
    clearml_options,
    clearml_settings,
    configure_clearml_config_file,
)


@configured_command(
    "pipeline",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run tokenizer training, model training, evaluation, and query as a ClearML Pipeline DAG.",
)
@pipeline_resume_option
@click.option(
    "--run-stage",
    type=click.Choice(PIPELINE_STAGES),
    default=None,
    help=(
        "Resume an existing controller and run only this stage. "
        "If --pipeline-controller-id is omitted, the newest eligible run is selected."
    ),
)
@click.option(
    "--run-until-stage",
    type=click.Choice(PIPELINE_STAGES),
    default=None,
    help="Create a new controller run and stop after this stage has run.",
)
@pipeline_options(default_name=DEFAULT_PIPELINE_NAME)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_names()),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model to train, evaluate, and query.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus to use.",
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
    help="Apply the same row limit to tokenizer training, model training, and evaluation.",
)
@click.option(
    "--tokenizer-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Train the tokenizer on only the first N rows. Overrides --limit for this stage.",
)
@click.option(
    "--training-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Train the language model on only the first N rows. Overrides --limit for this stage.",
)
@click.option(
    "--evaluation-limit",
    type=click.IntRange(min=0),
    default=None,
    help="Evaluate on only the first N rows. Overrides --limit for this stage.",
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
    help="Fraction of characters covered by the tokenizer.",
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
    "--top-k",
    type=click.IntRange(min=1),
    default=5,
    show_default=True,
    help="K value for top-k next-token accuracy.",
)
@click.option(
    "--query-prompt",
    default="Once upon",
    show_default=True,
    help="Text prefix for the final query stage.",
)
@click.option(
    "--query-max-tokens",
    type=click.IntRange(min=0),
    default=80,
    show_default=True,
    help="Maximum number of new tokens to generate in the final query stage.",
)
@click.option(
    "--query-top-k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of likely next tokens to store for the query prompt.",
)
@click.option(
    "--query-decoding",
    type=click.Choice(("sample", "most-probable")),
    default="sample",
    show_default=True,
    help="Generate the final query by sampling or by choosing the most probable next token.",
)
@click.option(
    "--query-temperature",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Sampling temperature for the final query. Ignored for most-probable decoding.",
)
@click.option(
    "--query-seed",
    type=int,
    default=1,
    show_default=True,
    help="Random seed for the final query sampling stage.",
)
@click.option(
    "--text-normalization",
    type=click.Choice(TEXT_NORMALIZATION_MODES),
    default=DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before tokenizer and model training.",
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
    run_until_stage: str | None,
    run_stage: str | None,
    pipeline_controller_id: str | None,
    model_name: str,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    tokenizer_limit: int | None,
    training_limit: int | None,
    evaluation_limit: int | None,
    vocab_size: int,
    artifact_name: str | None,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    top_k: int,
    query_prompt: str,
    query_max_tokens: int,
    query_top_k: int,
    query_decoding: str,
    query_temperature: float,
    query_seed: int | None,
    text_normalization: str,
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
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")
    if pipeline_local and not wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")

    resolved_pipeline_name = clearml_task_name or pipeline_name
    resolved_dataset_id = dataset_id or corpus_definition.dataset_id
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = text_column or corpus_definition.text_column
    resolved_artifact_name = artifact_name or f"{corpus}-sentencepiece-{vocab_size}"
    resolved_tokenizer_limit = tokenizer_limit if tokenizer_limit is not None else limit
    resolved_training_limit = training_limit if training_limit is not None else limit
    resolved_evaluation_limit = evaluation_limit if evaluation_limit is not None else limit

    if run_stage is not None and run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")
    if pipeline_controller_id is not None and run_stage is None:
        raise click.ClickException("--pipeline-controller-id must be used with --run-stage.")

    parameter_filters = {
        "model": model_definition.name,
        "corpus": corpus,
        "dataset_id": resolved_dataset_id,
        "source_split": resolved_source_split or "",
        "evaluation_partition": evaluation_partition,
    }
    if run_stage is not None or pipeline_controller_id is not None:
        if pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
                "Use --pipeline-queued when passing --run-stage or --pipeline-controller-id."
            )
        resume_pipeline_controller_stage(
            stage_name=run_stage or TOKENIZER_STAGE,
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
        run_until_stage=run_until_stage,
        updated_by="pipeline-cli",
    )
    connect_controller_experiment_parameters(
        pipeline.task,
        {
            "model": model_definition.name,
            "corpus": corpus,
            "dataset_id": resolved_dataset_id,
            "source_split": resolved_source_split or "",
            "text_column": resolved_text_column,
            "evaluation_partition": evaluation_partition,
        },
    )
    add_pipeline_steps(
        pipeline,
        clearml_project=settings.project_name,
        clearml_output_uri=settings.output_uri,
        clearml_tags=settings.tags,
        clearml_config_file=resolved_config_file if pipeline_local else None,
        execution_queue=None if pipeline_local else execution_queue,
        model_name=model_definition.name,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        text_column=resolved_text_column,
        streaming=streaming,
        train_ratio=train_ratio,
        split_seed=split_seed,
        evaluation_partition=evaluation_partition,
        tokenizer_limit=resolved_tokenizer_limit,
        training_limit=resolved_training_limit,
        evaluation_limit=resolved_evaluation_limit,
        vocab_size=vocab_size,
        artifact_name=resolved_artifact_name,
        model_type=model_type,
        character_coverage=character_coverage,
        hard_vocab_limit=hard_vocab_limit,
        max_sentence_length=max_sentence_length,
        smoothing=smoothing,
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
        discount=discount,
        top_k=top_k,
        query_prompt=query_prompt,
        query_max_tokens=query_max_tokens,
        query_top_k=query_top_k,
        query_decoding=query_decoding,
        query_temperature=query_temperature,
        query_seed=query_seed,
        text_normalization=text_normalization,
    )

    click.echo(f"ClearML pipeline: {settings.project_name}/{resolved_pipeline_name}")
    click.echo(f"Pipeline version: {pipeline_version}")
    click.echo(f"Pipeline controller task ID: {pipeline.task.id}")
    task_url = pipeline.task.get_output_log_web_page()
    if task_url:
        click.echo(f"Pipeline controller URL: {task_url}")
    if run_until_stage is not None:
        click.echo(f"Run until stage: {run_until_stage}")
    click.echo(f"Stage tasks: {TOKENIZER_STAGE}, {MODEL_STAGE}, {EVALUATION_STAGE}, {QUERY_STAGE}")

    if pipeline_local:
        click.echo("Execution mode: local ClearML PipelineController")
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        click.echo(f"Execution mode: queued controller on {controller_queue}")
        if execution_queue is not None:
            click.echo(f"Step execution queue: {execution_queue}")
        pipeline.start(queue=controller_queue, wait=wait)

    click.echo("ClearML pipeline submitted.")
    if wait:
        assert_pipeline_finished_successfully(pipeline)
        print_stage_task_ids(
            pipeline.task.id,
            (TOKENIZER_STAGE, MODEL_STAGE, EVALUATION_STAGE, QUERY_STAGE),
        )
        click.echo("ClearML pipeline run completed.")


def add_pipeline_steps(
    pipeline: object,
    *,
    clearml_project: str,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    clearml_config_file: Path | None,
    execution_queue: str | None,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    tokenizer_limit: int | None,
    training_limit: int | None,
    evaluation_limit: int | None,
    vocab_size: int,
    artifact_name: str,
    model_type: str,
    character_coverage: float,
    hard_vocab_limit: bool,
    max_sentence_length: int | None,
    smoothing: float,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
    discount: float,
    top_k: int,
    query_prompt: str,
    query_max_tokens: int,
    query_top_k: int,
    query_decoding: str,
    query_temperature: float,
    query_seed: int | None,
    text_normalization: str,
) -> None:
    artifact_monitors = pipeline_artifact_monitors()
    metric_monitors = pipeline_metric_monitors(evaluation_partition)
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
            "limit": tokenizer_limit,
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
    pipeline.add_function_step(
        name=MODEL_STAGE,
        function=train_model_stage_entry,
        function_kwargs={
            "tokenizer_task_id": f"${{{TOKENIZER_STAGE}.id}}",
            "model_name": model_name,
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": training_limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "smoothing": smoothing,
            "unigram_weight": unigram_weight,
            "bigram_weight": bigram_weight,
            "trigram_weight": trigram_weight,
            "discount": discount,
            "text_normalization": text_normalization,
            **common_step_kwargs,
        },
        parents=[TOKENIZER_STAGE],
        task_name=MODEL_STAGE,
        task_type="training",
        monitor_artifacts=artifact_monitors[MODEL_STAGE],
        monitor_metrics=metric_monitors[MODEL_STAGE],
        stage=MODEL_STAGE,
        **step_options,
    )
    pipeline.add_function_step(
        name=EVALUATION_STAGE,
        function=evaluate_stage_entry,
        function_kwargs={
            "model_task_id": f"${{{MODEL_STAGE}.id}}",
            "model_name": model_name,
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": evaluation_limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "evaluation_partition": evaluation_partition,
            "top_k": top_k,
            **common_step_kwargs,
        },
        parents=[MODEL_STAGE],
        task_name=EVALUATION_STAGE,
        task_type="testing",
        monitor_artifacts=artifact_monitors[EVALUATION_STAGE],
        monitor_metrics=metric_monitors[EVALUATION_STAGE],
        stage=EVALUATION_STAGE,
        **step_options,
    )
    pipeline.add_function_step(
        name=QUERY_STAGE,
        function=query_stage_entry,
        function_kwargs={
            "model_task_id": f"${{{MODEL_STAGE}.id}}",
            "model_name": model_name,
            "corpus": corpus,
            "prompt": query_prompt,
            "max_tokens": query_max_tokens,
            "top_k": query_top_k,
            "decoding": query_decoding,
            "temperature": query_temperature,
            "seed": query_seed,
            **common_step_kwargs,
        },
        parents=[MODEL_STAGE],
        task_name=QUERY_STAGE,
        task_type="inference",
        monitor_artifacts=artifact_monitors[QUERY_STAGE],
        monitor_metrics=metric_monitors[QUERY_STAGE],
        stage=QUERY_STAGE,
        **step_options,
    )
if __name__ == "__main__":
    main()
