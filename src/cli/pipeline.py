"""ClearML PipelineController DAG for model training experiments."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.pipeline_common import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_TOKENIZER_PIPELINE_NAME,
    EVALUATION_STAGE,
    MODEL_STAGE,
    QUERY_STAGE,
    TRAINING_PIPELINE_STAGE_DEPENDENCIES,
    TRAINING_PIPELINE_STAGES,
    assert_pipeline_finished_successfully,
    build_pipeline_controller,
    configure_pipeline_control,
    connect_controller_experiment_parameters,
    output_uri_value,
    pipeline_options,
    pipeline_resume_option,
    print_stage_task_ids,
    resolve_tokenizer_pipeline_task,
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


PIPELINE_CONFIG_SECTION = "pipeline"
TRAIN_CONFIG_SECTION = "train"
EVALUATE_CONFIG_SECTION = "evaluate"
QUERY_CONFIG_SECTION = "query"
EXPLICIT_PARAMETER_SOURCES = {"COMMANDLINE", "ENVIRONMENT"}


def _resolve_stage_default(
    ctx: click.Context | None,
    *,
    parameter_name: str,
    current_value: object,
    pipeline_defaults: Mapping[str, object],
    stage_defaults: Mapping[str, object],
    stage_key: str | None = None,
) -> object:
    if _parameter_is_explicit(ctx, parameter_name):
        return current_value
    if parameter_name in pipeline_defaults:
        return current_value

    resolved_stage_key = stage_key or parameter_name
    if resolved_stage_key in stage_defaults:
        return stage_defaults[resolved_stage_key]
    return current_value


def _resolve_consistent_stage_default(
    ctx: click.Context | None,
    *,
    parameter_name: str,
    current_value: object,
    pipeline_defaults: Mapping[str, object],
    candidates: tuple[tuple[str, Mapping[str, object], str], ...],
) -> object:
    if _parameter_is_explicit(ctx, parameter_name):
        return current_value
    if parameter_name in pipeline_defaults:
        return current_value

    values = [
        (section, defaults[key])
        for section, defaults, key in candidates
        if key in defaults
    ]
    if not values:
        return current_value

    first_value = values[0][1]
    conflicts = [
        (section, value)
        for section, value in values[1:]
        if value != first_value
    ]
    if conflicts:
        formatted_values = ", ".join(
            f"[{section}] {parameter_name}={value!r}"
            for section, value in values
        )
        raise click.ClickException(
            f"Conflicting pipeline defaults for {parameter_name!r}: {formatted_values}. "
            "Set one shared value in [defaults] or [pipeline], or make the stage sections match."
        )
    return first_value


def _resolve_stage_limit(
    ctx: click.Context | None,
    *,
    parameter_name: str,
    current_value: int | None,
    pipeline_defaults: Mapping[str, object],
    stage_defaults: Mapping[str, object],
    global_limit: int | None,
) -> int | None:
    if _parameter_is_explicit(ctx, parameter_name):
        return current_value
    if parameter_name in pipeline_defaults:
        return current_value
    if _parameter_is_explicit(ctx, "limit") or "limit" in pipeline_defaults:
        return global_limit
    if "limit" in stage_defaults:
        return stage_defaults["limit"]  # type: ignore[return-value]
    return global_limit


def _parameter_is_explicit(ctx: click.Context | None, parameter_name: str) -> bool:
    if ctx is None or parameter_name not in ctx.params:
        return False
    source = ctx.get_parameter_source(parameter_name)
    return getattr(source, "name", None) in EXPLICIT_PARAMETER_SOURCES


def load_pipeline_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections((TRAIN_CONFIG_SECTION,))
    evaluate_defaults = load_defaults_from_sections((EVALUATE_CONFIG_SECTION,))
    query_defaults = load_defaults_from_sections((QUERY_CONFIG_SECTION,))
    pipeline_defaults = load_defaults_from_sections((PIPELINE_CONFIG_SECTION,))

    defaults.update(
        _consistent_config_values(
            current_defaults=defaults,
            candidates=(
                ("train", train_defaults),
                ("evaluate", evaluate_defaults),
                ("query", query_defaults),
            ),
            parameter_names=(
                "corpus",
                "dataset_id",
                "source_split",
                "train_ratio",
                "split_seed",
                "text_column",
                "streaming",
            ),
        )
    )
    defaults.update(
        _consistent_config_values(
            current_defaults=defaults,
            candidates=(
                ("train", train_defaults),
                ("evaluate", evaluate_defaults),
                ("query", query_defaults),
            ),
            parameter_names=("model_name",),
        )
    )
    defaults.update(
        _mapped_config_values(
            train_defaults,
            {
                "tokenizer_model_name": "tokenizer_model_name",
                "smoothing": "smoothing",
                "unigram_weight": "unigram_weight",
                "bigram_weight": "bigram_weight",
                "trigram_weight": "trigram_weight",
                "discount": "discount",
                "limit": "training_limit",
                "text_normalization": "text_normalization",
            },
        )
    )
    defaults.update(
        _mapped_config_values(
            evaluate_defaults,
            {
                "evaluation_partition": "evaluation_partition",
                "top_k": "top_k",
                "limit": "evaluation_limit",
            },
        )
    )
    defaults.update(
        _mapped_config_values(
            query_defaults,
            {
                "prompt": "query_prompt",
                "max_tokens": "query_max_tokens",
                "top_k": "query_top_k",
                "decoding": "query_decoding",
                "temperature": "query_temperature",
                "seed": "query_seed",
            },
        )
    )
    defaults.update(pipeline_defaults)
    return defaults


def _consistent_config_values(
    *,
    current_defaults: Mapping[str, object],
    candidates: tuple[tuple[str, Mapping[str, object]], ...],
    parameter_names: tuple[str, ...],
) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for parameter_name in parameter_names:
        values = [
            (section, defaults[parameter_name])
            for section, defaults in candidates
            if parameter_name in defaults
        ]
        if not values:
            continue
        first_value = values[0][1]
        conflicts = [
            (section, value)
            for section, value in values[1:]
            if value != first_value
        ]
        if conflicts:
            formatted_values = ", ".join(
                f"[{section}] {parameter_name}={value!r}"
                for section, value in values
            )
            raise click.ClickException(
                f"Conflicting pipeline defaults for {parameter_name!r}: {formatted_values}. "
                "Set one shared value in [defaults] or [pipeline], or make the stage sections match."
            )
        if parameter_name not in current_defaults or first_value != current_defaults[parameter_name]:
            resolved[parameter_name] = first_value
    return resolved


def _mapped_config_values(
    defaults: Mapping[str, object],
    key_map: Mapping[str, str],
) -> dict[str, object]:
    return {
        target_key: defaults[source_key]
        for source_key, target_key in key_map.items()
        if source_key in defaults
    }


@configured_command(
    "pipeline",
    default_loader=load_pipeline_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run model training, evaluation, and query as a ClearML Pipeline DAG.",
)
@pipeline_resume_option
@click.option(
    "--run-stage",
    type=click.Choice(TRAINING_PIPELINE_STAGES),
    default=None,
    help=(
        "Resume an existing controller and run only this stage. "
        "If --pipeline-controller-id is omitted, the newest eligible run is selected."
    ),
)
@click.option(
    "--run-until-stage",
    type=click.Choice(TRAINING_PIPELINE_STAGES),
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
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name to resolve from the tokenizer pipeline.",
)
@click.option(
    "--tokenizer-pipeline-name",
    default=DEFAULT_TOKENIZER_PIPELINE_NAME,
    show_default=True,
    help="ClearML tokenizer pipeline name to search for reusable tokenizer artifacts.",
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
    help="Apply the same row limit to model training and evaluation.",
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
    run_until_stage: str | None,
    run_stage: str | None,
    pipeline_controller_id: str | None,
    model_name: str,
    tokenizer_model_name: str | None,
    tokenizer_pipeline_name: str,
    corpus: str,
    dataset_id: str | None,
    source_split: str | None,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    text_column: str | None,
    streaming: bool,
    limit: int | None,
    training_limit: int | None,
    evaluation_limit: int | None,
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
    ctx = click.get_current_context(silent=True)
    pipeline_defaults = load_defaults_from_sections((PIPELINE_CONFIG_SECTION,))
    train_defaults = load_defaults_from_sections((TRAIN_CONFIG_SECTION,))
    evaluate_defaults = load_defaults_from_sections((EVALUATE_CONFIG_SECTION,))
    query_defaults = load_defaults_from_sections((QUERY_CONFIG_SECTION,))

    corpus = _resolve_consistent_stage_default(
        ctx,
        parameter_name="corpus",
        current_value=corpus,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "corpus"),
            ("evaluate", evaluate_defaults, "corpus"),
            ("query", query_defaults, "corpus"),
        ),
    )
    model_name = _resolve_consistent_stage_default(
        ctx,
        parameter_name="model_name",
        current_value=model_name,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "model_name"),
            ("evaluate", evaluate_defaults, "model_name"),
            ("query", query_defaults, "model_name"),
        ),
    )
    tokenizer_model_name = _resolve_stage_default(
        ctx,
        parameter_name="tokenizer_model_name",
        current_value=tokenizer_model_name,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    dataset_id = _resolve_consistent_stage_default(
        ctx,
        parameter_name="dataset_id",
        current_value=dataset_id,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "dataset_id"),
            ("evaluate", evaluate_defaults, "dataset_id"),
        ),
    )
    source_split = _resolve_consistent_stage_default(
        ctx,
        parameter_name="source_split",
        current_value=source_split,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "source_split"),
            ("evaluate", evaluate_defaults, "source_split"),
        ),
    )
    train_ratio = _resolve_consistent_stage_default(
        ctx,
        parameter_name="train_ratio",
        current_value=train_ratio,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "train_ratio"),
            ("evaluate", evaluate_defaults, "train_ratio"),
        ),
    )
    split_seed = _resolve_consistent_stage_default(
        ctx,
        parameter_name="split_seed",
        current_value=split_seed,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "split_seed"),
            ("evaluate", evaluate_defaults, "split_seed"),
        ),
    )
    text_column = _resolve_consistent_stage_default(
        ctx,
        parameter_name="text_column",
        current_value=text_column,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "text_column"),
            ("evaluate", evaluate_defaults, "text_column"),
        ),
    )
    streaming = _resolve_consistent_stage_default(
        ctx,
        parameter_name="streaming",
        current_value=streaming,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "streaming"),
            ("evaluate", evaluate_defaults, "streaming"),
        ),
    )
    evaluation_partition = _resolve_stage_default(
        ctx,
        parameter_name="evaluation_partition",
        current_value=evaluation_partition,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=evaluate_defaults,
    )
    smoothing = _resolve_stage_default(
        ctx,
        parameter_name="smoothing",
        current_value=smoothing,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    unigram_weight = _resolve_stage_default(
        ctx,
        parameter_name="unigram_weight",
        current_value=unigram_weight,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    bigram_weight = _resolve_stage_default(
        ctx,
        parameter_name="bigram_weight",
        current_value=bigram_weight,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    trigram_weight = _resolve_stage_default(
        ctx,
        parameter_name="trigram_weight",
        current_value=trigram_weight,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    discount = _resolve_stage_default(
        ctx,
        parameter_name="discount",
        current_value=discount,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    top_k = _resolve_stage_default(
        ctx,
        parameter_name="top_k",
        current_value=top_k,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=evaluate_defaults,
    )
    query_prompt = _resolve_stage_default(
        ctx,
        parameter_name="query_prompt",
        current_value=query_prompt,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="prompt",
    )
    query_max_tokens = _resolve_stage_default(
        ctx,
        parameter_name="query_max_tokens",
        current_value=query_max_tokens,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="max_tokens",
    )
    query_top_k = _resolve_stage_default(
        ctx,
        parameter_name="query_top_k",
        current_value=query_top_k,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="top_k",
    )
    query_decoding = _resolve_stage_default(
        ctx,
        parameter_name="query_decoding",
        current_value=query_decoding,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="decoding",
    )
    query_temperature = _resolve_stage_default(
        ctx,
        parameter_name="query_temperature",
        current_value=query_temperature,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="temperature",
    )
    query_seed = _resolve_stage_default(
        ctx,
        parameter_name="query_seed",
        current_value=query_seed,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="seed",
    )
    text_normalization = _resolve_stage_default(
        ctx,
        parameter_name="text_normalization",
        current_value=text_normalization,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    resolved_training_limit = _resolve_stage_limit(
        ctx,
        parameter_name="training_limit",
        current_value=training_limit,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
        global_limit=limit,
    )
    resolved_evaluation_limit = _resolve_stage_limit(
        ctx,
        parameter_name="evaluation_limit",
        current_value=evaluation_limit,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=evaluate_defaults,
        global_limit=limit,
    )

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
    resolved_tokenizer_model_name = str(tokenizer_model_name or "").strip()
    if not resolved_tokenizer_model_name:
        raise click.ClickException(
            "Training pipeline requires --tokenizer-model-name, or tokenizer_model_name in [train]. "
            "Run the tokenizer pipeline first and use its tokenizer model name."
        )

    if run_stage is not None and run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")
    if pipeline_controller_id is not None and run_stage is None:
        raise click.ClickException("--pipeline-controller-id must be used with --run-stage.")

    parameter_filters = {
        "model": model_definition.name,
        "corpus": corpus,
        "tokenizer_model_name": resolved_tokenizer_model_name,
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
            stage_name=run_stage or MODEL_STAGE,
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
            stage_dependencies=TRAINING_PIPELINE_STAGE_DEPENDENCIES,
            stage_names=TRAINING_PIPELINE_STAGES,
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

    tokenizer_resolution = resolve_tokenizer_pipeline_task(
        tokenizer_pipeline_name=tokenizer_pipeline_name,
        clearml_project=settings.project_name,
        corpus=corpus,
        tokenizer_model_name=resolved_tokenizer_model_name,
    )

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
            "tokenizer_model_name": resolved_tokenizer_model_name,
            "tokenizer_pipeline_name": tokenizer_pipeline_name,
            "tokenizer_pipeline_controller_id": tokenizer_resolution.controller_id,
            "tokenizer_task_id": tokenizer_resolution.tokenizer_task_id,
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
        tokenizer_task_id=tokenizer_resolution.tokenizer_task_id,
        model_name=model_definition.name,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        text_column=resolved_text_column,
        streaming=streaming,
        train_ratio=train_ratio,
        split_seed=split_seed,
        evaluation_partition=evaluation_partition,
        training_limit=resolved_training_limit,
        evaluation_limit=resolved_evaluation_limit,
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
    click.echo(f"Tokenizer model: {resolved_tokenizer_model_name}")
    click.echo(f"Tokenizer pipeline controller task ID: {tokenizer_resolution.controller_id}")
    click.echo(f"Tokenizer stage task ID: {tokenizer_resolution.tokenizer_task_id}")
    click.echo(f"Pipeline controller task ID: {pipeline.task.id}")
    task_url = pipeline.task.get_output_log_web_page()
    if task_url:
        click.echo(f"Pipeline controller URL: {task_url}")
    if run_until_stage is not None:
        click.echo(f"Run until stage: {run_until_stage}")
    click.echo(f"Stage tasks: {MODEL_STAGE}, {EVALUATION_STAGE}, {QUERY_STAGE}")

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
            TRAINING_PIPELINE_STAGES,
            stage_names=TRAINING_PIPELINE_STAGES,
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
    tokenizer_task_id: str,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
    streaming: bool,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    training_limit: int | None,
    evaluation_limit: int | None,
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
        name=MODEL_STAGE,
        function=train_model_stage_entry,
        function_kwargs={
            "tokenizer_task_id": tokenizer_task_id,
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
