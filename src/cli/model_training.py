"""ClearML PipelineController DAG for model training experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from src.ml_core.cli.config import configured_command, load_defaults_from_sections
from src.pipelines.language_model.definition import (
    assert_pipeline_finished_successfully,
    build_pipeline_controller,
    configure_pipeline_control,
    connect_controller_experiment_parameters,
    pipeline_options,
    pipeline_resume_option,
    print_stage_task_ids,
    resume_pipeline_controller_stage,
)
from src.pipelines.language_model.model_training import (
    DEFAULT_TOKENIZER_TRAINING_NAME,
    EVALUATION_STAGE,
    MODEL_STAGE,
    MODEL_TRAINING_PIPELINE,
    QUERY_STAGE,
    add_pipeline_steps,
    resolve_tokenizer_training_task,
)
from src.pipelines.language_model.model_options import (
    MODEL_HYPERPARAMETER_NAMES,
    model_hyperparameters_from,
)
from src.pipelines.language_model.optuna import (
    DEFAULT_OPTUNA_DIRECTION,
    DEFAULT_OPTUNA_METRIC,
    SearchSpec,
    describe_search_space,
    load_objective_metric,
    parse_optuna_search_specs,
    sample_trial_parameters,
)
from src.corpora import normalization
from src.corpora import registry as corpora_registry
from src.ml_core.data.splits import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    PROJECT_PARTITIONS,
    VALIDATION_PARTITION,
)
from src.models.core import registry as model_registry
from src.ml_core.tracking import (
    assert_clearml_endpoints_reachable,
    clearml_options,
    clearml_settings,
    configure_clearml_config_file,
)


MODEL_TRAINING_CONFIG_SECTION = "model-training"
OPTUNA_CONFIG_SECTION = "optuna"
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

    values = _stage_config_values(candidates)
    if not values:
        return current_value
    return _consistent_stage_value(parameter_name, values)


def _stage_config_values(
    candidates: tuple[tuple[str, Mapping[str, object], str], ...],
) -> list[tuple[str, object]]:
    return [
        (section, defaults[key])
        for section, defaults, key in candidates
        if key in defaults
    ]


def _consistent_stage_value(
    parameter_name: str,
    values: Sequence[tuple[str, object]],
) -> object:
    first_value = values[0][1]
    conflicts = [
        (section, value)
        for section, value in values[1:]
        if value != first_value
    ]
    if not conflicts:
        return first_value

    formatted_values = ", ".join(
        f"[{section}] {parameter_name}={value!r}"
        for section, value in values
    )
    raise click.ClickException(
        f"Conflicting pipeline defaults for {parameter_name!r}: {formatted_values}. "
        "Set one shared value in [defaults] or [model-training], "
        "or make the stage sections match."
    )


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
        return _optional_int_default(stage_defaults["limit"], name="limit")
    return global_limit


def _optional_int_default(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise click.ClickException(f"Config default {name!r} must be an integer.")
    return value


def _parameter_is_explicit(ctx: click.Context | None, parameter_name: str) -> bool:
    if ctx is None or parameter_name not in ctx.params:
        return False
    source = ctx.get_parameter_source(parameter_name)
    return getattr(source, "name", None) in EXPLICIT_PARAMETER_SOURCES


def load_model_training_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections((TRAIN_CONFIG_SECTION,))
    evaluate_defaults = load_defaults_from_sections((EVALUATE_CONFIG_SECTION,))
    query_defaults = load_defaults_from_sections((QUERY_CONFIG_SECTION,))
    optuna_defaults = load_defaults_from_sections((OPTUNA_CONFIG_SECTION,))
    pipeline_defaults = load_defaults_from_sections((MODEL_TRAINING_CONFIG_SECTION,))

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
    defaults.update(optuna_defaults)
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
        first_value = _consistent_stage_value(parameter_name, values)
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
    "model-training",
    default_loader=load_model_training_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run model training, evaluation, and query as a ClearML Pipeline DAG.",
)
@pipeline_resume_option
@click.option(
    "--run-stage",
    type=click.Choice(MODEL_TRAINING_PIPELINE.stages),
    default=None,
    help=(
        "Resume an existing controller and run only this stage. "
        "If --pipeline-controller-id is omitted, the newest eligible run is selected."
    ),
)
@click.option(
    "--run-until-stage",
    type=click.Choice(MODEL_TRAINING_PIPELINE.stages),
    default=None,
    help="Create a new controller run and stop after this stage has run.",
)
@pipeline_options(default_name=MODEL_TRAINING_PIPELINE.default_name)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_registry.model_names()),
    default=model_registry.default_model_name(),
    show_default=True,
    help="Registered model to train, evaluate, and query.",
)
@click.option(
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name to resolve from tokenizer-training runs.",
)
@click.option(
    "--tokenizer-training-name",
    default=DEFAULT_TOKENIZER_TRAINING_NAME,
    show_default=True,
    help="ClearML tokenizer-training pipeline name to search for reusable tokenizer models.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpora_registry.corpus_names()),
    default=corpora_registry.default_corpus_name(),
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
    help=(
        "Primary project partition for unpartitioned summary metrics and Optuna "
        "objectives. The evaluation stage evaluates all project partitions."
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
    type=click.Choice(normalization.TEXT_NORMALIZATION_MODES),
    default=normalization.DEFAULT_TEXT_NORMALIZATION,
    show_default=True,
    help="Text normalization applied before model training.",
)
@click.option(
    "--optuna-trials",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Run this many Optuna trials. Zero disables hyperparameter optimization.",
)
@click.option(
    "--optuna-search",
    "optuna_search",
    multiple=True,
    help=(
        "Hyperparameter search spec. Repeatable. Examples: "
        "smoothing=float:1e-4:1.0:log, discount=float:0.1:0.95, "
        "top_k=int:1:10, model=categorical:bigram,trigram."
    ),
)
@click.option(
    "--optuna-metric",
    default=DEFAULT_OPTUNA_METRIC,
    show_default=True,
    help="Evaluation summary metric used as the Optuna objective.",
)
@click.option(
    "--optuna-direction",
    type=click.Choice(("minimize", "maximize")),
    default=DEFAULT_OPTUNA_DIRECTION,
    show_default=True,
    help="Whether Optuna should minimize or maximize the objective metric.",
)
@click.option(
    "--optuna-study-name",
    default=None,
    help="Optional Optuna study name. Required by some persistent storage backends.",
)
@click.option(
    "--optuna-storage",
    default=None,
    help="Optional Optuna storage URL, for example sqlite:///optuna.db.",
)
@click.option(
    "--optuna-load-if-exists/--optuna-no-load-if-exists",
    default=True,
    show_default=True,
    help="Reuse an existing named Optuna study when storage is configured.",
)
@click.option(
    "--optuna-timeout-seconds",
    type=click.IntRange(min=1),
    default=None,
    help="Optional maximum wall-clock time for the Optuna study.",
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
    tokenizer_training_name: str,
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
    optuna_trials: int,
    optuna_search: tuple[str, ...],
    optuna_metric: str,
    optuna_direction: str,
    optuna_study_name: str | None,
    optuna_storage: str | None,
    optuna_load_if_exists: bool,
    optuna_timeout_seconds: int | None,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    ctx = click.get_current_context(silent=True)
    pipeline_defaults = load_defaults_from_sections((MODEL_TRAINING_CONFIG_SECTION,))
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
    model_hyperparameters = model_hyperparameters_from(locals())
    model_hyperparameters = {
        name: _resolve_stage_default(
            ctx,
            parameter_name=name,
            current_value=model_hyperparameters[name],
            pipeline_defaults=pipeline_defaults,
            stage_defaults=train_defaults,
        )
        for name in MODEL_HYPERPARAMETER_NAMES
    }
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
    optuna_search_specs = parse_optuna_search_specs(optuna_search)
    optuna_enabled = optuna_trials > 0 or bool(optuna_search_specs)

    corpus_definition = corpora_registry.get_corpus(corpus)
    model_definition = model_registry.get_model(model_name)
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
            "Run tokenizer training first and use its tokenizer model name."
        )

    if run_stage is not None and run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")
    if pipeline_controller_id is not None and run_stage is None:
        raise click.ClickException("--pipeline-controller-id must be used with --run-stage.")
    if optuna_enabled:
        if optuna_trials <= 0:
            raise click.ClickException("--optuna-trials must be greater than zero when --optuna-search is set.")
        if not optuna_search_specs:
            raise click.ClickException("--optuna-trials requires at least one --optuna-search spec.")
        if run_stage is not None or run_until_stage is not None or pipeline_controller_id is not None:
            raise click.ClickException(
                "Optuna runs the full model-training pipeline for every trial; "
                "do not combine it with --run-stage, --run-until-stage, or --pipeline-controller-id."
            )
        if not wait:
            raise click.ClickException(
                "Optuna requires --wait so each trial can read its evaluation objective metric."
            )

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
            stage_dependencies=MODEL_TRAINING_PIPELINE.stage_dependencies,
            stage_names=MODEL_TRAINING_PIPELINE.stages,
        )
        return

    if optuna_enabled:
        _run_optuna_model_training(
            optuna_trials=optuna_trials,
            optuna_search_specs=optuna_search_specs,
            optuna_metric=optuna_metric,
            optuna_direction=optuna_direction,
            optuna_study_name=optuna_study_name,
            optuna_storage=optuna_storage,
            optuna_load_if_exists=optuna_load_if_exists,
            optuna_timeout_seconds=optuna_timeout_seconds,
            resolved_pipeline_name=resolved_pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_local=pipeline_local,
            controller_queue=controller_queue,
            execution_queue=execution_queue,
            wait=wait,
            add_run_number=add_run_number,
            tokenizer_training_name=tokenizer_training_name,
            model_name=model_definition.name,
            corpus=corpus,
            resolved_tokenizer_model_name=resolved_tokenizer_model_name,
            resolved_dataset_id=resolved_dataset_id,
            resolved_source_split=resolved_source_split,
            resolved_text_column=resolved_text_column,
            streaming=streaming,
            train_ratio=train_ratio,
            split_seed=split_seed,
            evaluation_partition=evaluation_partition,
            training_limit=resolved_training_limit,
            evaluation_limit=resolved_evaluation_limit,
            model_hyperparameters=model_hyperparameters,
            top_k=top_k,
            query_prompt=query_prompt,
            query_max_tokens=query_max_tokens,
            query_top_k=query_top_k,
            query_decoding=query_decoding,
            query_temperature=query_temperature,
            query_seed=query_seed,
            text_normalization=text_normalization,
            clearml_project=clearml_project,
            clearml_config_file=clearml_config_file,
            clearml_connectivity_check=clearml_connectivity_check,
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
        )
        return

    _run_model_training_pipeline(
        resolved_pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        pipeline_local=pipeline_local,
        controller_queue=controller_queue,
        execution_queue=execution_queue,
        wait=wait,
        add_run_number=add_run_number,
        run_until_stage=run_until_stage,
        tokenizer_training_name=tokenizer_training_name,
        model_name=model_definition.name,
        corpus=corpus,
        resolved_tokenizer_model_name=resolved_tokenizer_model_name,
        resolved_dataset_id=resolved_dataset_id,
        resolved_source_split=resolved_source_split,
        resolved_text_column=resolved_text_column,
        streaming=streaming,
        train_ratio=train_ratio,
        split_seed=split_seed,
        evaluation_partition=evaluation_partition,
        training_limit=resolved_training_limit,
        evaluation_limit=resolved_evaluation_limit,
        model_hyperparameters=model_hyperparameters,
        top_k=top_k,
        query_prompt=query_prompt,
        query_max_tokens=query_max_tokens,
        query_top_k=query_top_k,
        query_decoding=query_decoding,
        query_temperature=query_temperature,
        query_seed=query_seed,
        text_normalization=text_normalization,
        clearml_project=clearml_project,
        clearml_config_file=clearml_config_file,
        clearml_connectivity_check=clearml_connectivity_check,
        clearml_output_uri=clearml_output_uri,
        clearml_tags=clearml_tags,
    )


def _run_optuna_model_training(
    *,
    optuna_trials: int,
    optuna_search_specs: Sequence[SearchSpec],
    optuna_metric: str,
    optuna_direction: str,
    optuna_study_name: str | None,
    optuna_storage: str | None,
    optuna_load_if_exists: bool,
    optuna_timeout_seconds: int | None,
    resolved_pipeline_name: str,
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    tokenizer_training_name: str,
    model_name: str,
    corpus: str,
    resolved_tokenizer_model_name: str,
    resolved_dataset_id: str,
    resolved_source_split: str | None,
    resolved_text_column: str,
    streaming: bool,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    training_limit: int | None,
    evaluation_limit: int | None,
    model_hyperparameters: Mapping[str, object],
    top_k: int,
    query_prompt: str,
    query_max_tokens: int,
    query_top_k: int,
    query_decoding: str,
    query_temperature: float,
    query_seed: int | None,
    text_normalization: str,
    clearml_project: str,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    try:
        import optuna
    except ImportError as error:
        raise click.ClickException(
            "Optuna optimization requires the optuna Python package. "
            "Run `uv sync` before using --optuna-trials."
        ) from error

    study = optuna.create_study(
        study_name=optuna_study_name,
        storage=optuna_storage,
        direction=optuna_direction,
        load_if_exists=optuna_load_if_exists,
    )
    click.echo(f"Optuna study: {study.study_name}")
    click.echo(f"Optuna direction: {optuna_direction}")
    click.echo(f"Optuna objective metric: {optuna_metric}")
    click.echo(f"Optuna search space: {describe_search_space(optuna_search_specs)}")
    click.echo(f"Optuna trials: {optuna_trials}")

    def objective(trial: Any) -> float:
        sampled_parameters = sample_trial_parameters(trial, optuna_search_specs)
        trial_values = {
            "model_name": model_name,
            **model_hyperparameters,
            "top_k": top_k,
            "query_max_tokens": query_max_tokens,
            "query_top_k": query_top_k,
            "query_decoding": query_decoding,
            "query_temperature": query_temperature,
            "query_seed": query_seed,
        }
        trial_values.update(sampled_parameters)
        trial_tags = tuple(
            dict.fromkeys(
                (
                    *clearml_tags,
                    "optuna",
                    f"optuna-study-{study.study_name}",
                    f"optuna-trial-{trial.number}",
                )
            )
        )
        click.echo(
            f"Optuna trial {trial.number}: "
            + ", ".join(
                f"{name}={value!r}"
                for name, value in sorted(sampled_parameters.items())
            )
        )
        controller_id = _run_model_training_pipeline(
            resolved_pipeline_name=resolved_pipeline_name,
            pipeline_version=pipeline_version,
            pipeline_local=pipeline_local,
            controller_queue=controller_queue,
            execution_queue=execution_queue,
            wait=wait,
            add_run_number=add_run_number,
            run_until_stage=None,
            tokenizer_training_name=tokenizer_training_name,
            model_name=str(trial_values["model_name"]),
            corpus=corpus,
            resolved_tokenizer_model_name=resolved_tokenizer_model_name,
            resolved_dataset_id=resolved_dataset_id,
            resolved_source_split=resolved_source_split,
            resolved_text_column=resolved_text_column,
            streaming=streaming,
            train_ratio=train_ratio,
            split_seed=split_seed,
            evaluation_partition=evaluation_partition,
            training_limit=training_limit,
            evaluation_limit=evaluation_limit,
            model_hyperparameters=model_hyperparameters_from(trial_values),
            top_k=int(trial_values["top_k"]),
            query_prompt=query_prompt,
            query_max_tokens=int(trial_values["query_max_tokens"]),
            query_top_k=int(trial_values["query_top_k"]),
            query_decoding=str(trial_values["query_decoding"]),
            query_temperature=float(trial_values["query_temperature"]),
            query_seed=(
                int(trial_values["query_seed"])
                if trial_values["query_seed"] is not None
                else None
            ),
            text_normalization=text_normalization,
            clearml_project=clearml_project,
            clearml_config_file=clearml_config_file,
            clearml_connectivity_check=clearml_connectivity_check,
            clearml_output_uri=clearml_output_uri,
            clearml_tags=trial_tags,
            extra_controller_parameters={
                "optuna_study_name": study.study_name,
                "optuna_trial_number": trial.number,
                "optuna_metric": optuna_metric,
                "optuna_direction": optuna_direction,
                **{
                    f"optuna_{name}": value
                    for name, value in sampled_parameters.items()
                },
            },
        )
        objective_value = load_objective_metric(
            controller_id=controller_id,
            metric_name=optuna_metric,
            evaluation_partition=evaluation_partition,
        )
        trial.set_user_attr("pipeline_controller_id", controller_id)
        trial.set_user_attr("objective_metric", optuna_metric)
        click.echo(f"Optuna trial {trial.number} objective: {objective_value}")
        return objective_value

    study.optimize(
        objective,
        n_trials=optuna_trials,
        timeout=optuna_timeout_seconds,
    )
    try:
        best_trial = study.best_trial
    except ValueError:
        click.echo("Optuna study completed without a finished trial.")
        return

    click.echo(f"Optuna best trial: {best_trial.number}")
    click.echo(f"Optuna best value: {best_trial.value}")
    if best_trial.params:
        click.echo(
            "Optuna best parameters: "
            + ", ".join(
                f"{name}={value!r}"
                for name, value in sorted(best_trial.params.items())
            )
        )


def _run_model_training_pipeline(
    *,
    resolved_pipeline_name: str,
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    run_until_stage: str | None,
    tokenizer_training_name: str,
    model_name: str,
    corpus: str,
    resolved_tokenizer_model_name: str,
    resolved_dataset_id: str,
    resolved_source_split: str | None,
    resolved_text_column: str,
    streaming: bool,
    train_ratio: float,
    split_seed: int,
    evaluation_partition: str,
    training_limit: int | None,
    evaluation_limit: int | None,
    model_hyperparameters: Mapping[str, object],
    top_k: int,
    query_prompt: str,
    query_max_tokens: int,
    query_top_k: int,
    query_decoding: str,
    query_temperature: float,
    query_seed: int | None,
    text_normalization: str,
    clearml_project: str,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    extra_controller_parameters: Mapping[str, object] | None = None,
) -> str:
    model_definition = model_registry.get_model(model_name)
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

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

    tokenizer_resolution = resolve_tokenizer_training_task(
        tokenizer_training_name=tokenizer_training_name,
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
    controller_parameters: dict[str, object] = {
        "model": model_definition.name,
        "corpus": corpus,
        "tokenizer_model_name": resolved_tokenizer_model_name,
        "tokenizer_training_name": tokenizer_training_name,
        "tokenizer_training_controller_id": tokenizer_resolution.controller_id,
        "tokenizer_task_id": tokenizer_resolution.tokenizer_task_id,
        "dataset_id": resolved_dataset_id,
        "source_split": resolved_source_split or "",
        "text_column": resolved_text_column,
        "evaluation_partition": evaluation_partition,
    }
    if extra_controller_parameters:
        controller_parameters.update(extra_controller_parameters)
    connect_controller_experiment_parameters(pipeline.task, controller_parameters)
    add_pipeline_steps(
        pipeline,
        clearml_project=settings.project_name,
        clearml_output_uri=settings.output_uri,
        clearml_tags=settings.tags,
        clearml_config_file=resolved_config_file if pipeline_local else None,
        execution_queue=None if pipeline_local else execution_queue,
        tokenizer_task_id=tokenizer_resolution.tokenizer_task_id,
        tokenizer_model_name=resolved_tokenizer_model_name,
        model_name=model_definition.name,
        corpus=corpus,
        dataset_id=resolved_dataset_id,
        source_split=resolved_source_split,
        text_column=resolved_text_column,
        streaming=streaming,
        train_ratio=train_ratio,
        split_seed=split_seed,
        evaluation_partition=evaluation_partition,
        training_limit=training_limit,
        evaluation_limit=evaluation_limit,
        model_hyperparameters=model_hyperparameters,
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
            MODEL_TRAINING_PIPELINE.stages,
            stage_names=MODEL_TRAINING_PIPELINE.stages,
        )
        click.echo("ClearML pipeline run completed.")
    return str(pipeline.task.id)


if __name__ == "__main__":
    main()
