"""Model-training CLI orchestration."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import click

from src.cli import model_training_defaults as mt_defaults
from src.cli import model_training_runs
from src.ml_core import pipeline as core_pipeline
from src.ml_core.cli import config as cli_config
from src.models.core import registry as model_registry
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import optuna as lm_optuna
from src.corpora import registry as corpora_registry


@dataclasses.dataclass(frozen=True)
class CliArgs:
    """Raw Click arguments for the model-training command."""

    pipeline_name: str
    pipeline_version: str
    pipeline_local: bool
    controller_queue: str
    execution_queue: str | None
    wait: bool
    add_run_number: bool
    run_until_stage: str | None
    run_stage: str | None
    pipeline_controller_id: str | None
    model_name: str
    tokenizer_model_name: str | None
    tokenizer_training_name: str
    corpus: str
    dataset_id: str | None
    source_split: str | None
    train_ratio: float
    split_seed: int
    evaluation_partition: str
    text_column: str | None
    streaming: bool
    limit: int | None
    training_limit: int | None
    evaluation_limit: int | None
    smoothing: float
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    discount: float
    top_k: int
    query_prompt: str
    query_max_tokens: int
    query_top_k: int
    query_decoding: str
    query_temperature: float
    query_seed: int | None
    text_normalization: str
    optuna_trials: int
    optuna_search: tuple[str, ...]
    optuna_metric: str
    optuna_direction: str
    optuna_study_name: str | None
    optuna_storage: str | None
    optuna_load_if_exists: bool
    optuna_timeout_seconds: int | None
    clearml_project: str
    clearml_task_name: str | None
    clearml_config_file: Path | None
    clearml_connectivity_check: bool
    clearml_output_uri: str | None
    clearml_tags: tuple[str, ...]


def run(args: CliArgs) -> None:
    ctx = click.get_current_context(silent=True)
    pipeline_defaults = cli_config.load_defaults_from_sections(
        (mt_defaults.MODEL_TRAINING_CONFIG_SECTION,)
    )
    train_defaults = cli_config.load_defaults_from_sections(
        (mt_defaults.TRAIN_CONFIG_SECTION,)
    )
    evaluate_defaults = cli_config.load_defaults_from_sections(
        (mt_defaults.EVALUATE_CONFIG_SECTION,)
    )
    query_defaults = cli_config.load_defaults_from_sections(
        (mt_defaults.QUERY_CONFIG_SECTION,)
    )

    corpus = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="corpus",
        current_value=args.corpus,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "corpus"),
            ("evaluate", evaluate_defaults, "corpus"),
            ("query", query_defaults, "corpus"),
        ),
    )
    model_name = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="model_name",
        current_value=args.model_name,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "model_name"),
            ("evaluate", evaluate_defaults, "model_name"),
            ("query", query_defaults, "model_name"),
        ),
    )
    tokenizer_model_name = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="tokenizer_model_name",
        current_value=args.tokenizer_model_name,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    dataset_id = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="dataset_id",
        current_value=args.dataset_id,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "dataset_id"),
            ("evaluate", evaluate_defaults, "dataset_id"),
        ),
    )
    source_split = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="source_split",
        current_value=args.source_split,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "source_split"),
            ("evaluate", evaluate_defaults, "source_split"),
        ),
    )
    train_ratio = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="train_ratio",
        current_value=args.train_ratio,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "train_ratio"),
            ("evaluate", evaluate_defaults, "train_ratio"),
        ),
    )
    split_seed = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="split_seed",
        current_value=args.split_seed,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "split_seed"),
            ("evaluate", evaluate_defaults, "split_seed"),
        ),
    )
    text_column = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="text_column",
        current_value=args.text_column,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "text_column"),
            ("evaluate", evaluate_defaults, "text_column"),
        ),
    )
    streaming = mt_defaults.resolve_consistent_stage_default(
        ctx,
        parameter_name="streaming",
        current_value=args.streaming,
        pipeline_defaults=pipeline_defaults,
        candidates=(
            ("train", train_defaults, "streaming"),
            ("evaluate", evaluate_defaults, "streaming"),
        ),
    )
    evaluation_partition = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="evaluation_partition",
        current_value=args.evaluation_partition,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=evaluate_defaults,
    )
    model_hyperparameters = lm_model_options.model_hyperparameters_from(vars(args))
    model_hyperparameters = {
        name: mt_defaults.resolve_stage_default(
            ctx,
            parameter_name=name,
            current_value=model_hyperparameters[name],
            pipeline_defaults=pipeline_defaults,
            stage_defaults=train_defaults,
        )
        for name in lm_model_options.MODEL_HYPERPARAMETER_NAMES
    }
    top_k = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="top_k",
        current_value=args.top_k,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=evaluate_defaults,
    )
    query_prompt = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="query_prompt",
        current_value=args.query_prompt,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="prompt",
    )
    query_max_tokens = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="query_max_tokens",
        current_value=args.query_max_tokens,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="max_tokens",
    )
    query_top_k = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="query_top_k",
        current_value=args.query_top_k,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="top_k",
    )
    query_decoding = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="query_decoding",
        current_value=args.query_decoding,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="decoding",
    )
    query_temperature = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="query_temperature",
        current_value=args.query_temperature,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="temperature",
    )
    query_seed = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="query_seed",
        current_value=args.query_seed,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=query_defaults,
        stage_key="seed",
    )
    text_normalization = mt_defaults.resolve_stage_default(
        ctx,
        parameter_name="text_normalization",
        current_value=args.text_normalization,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
    )
    resolved_training_limit = mt_defaults.resolve_stage_limit(
        ctx,
        parameter_name="training_limit",
        current_value=args.training_limit,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=train_defaults,
        global_limit=args.limit,
    )
    resolved_evaluation_limit = mt_defaults.resolve_stage_limit(
        ctx,
        parameter_name="evaluation_limit",
        current_value=args.evaluation_limit,
        pipeline_defaults=pipeline_defaults,
        stage_defaults=evaluate_defaults,
        global_limit=args.limit,
    )
    optuna_search_specs = lm_optuna.parse_optuna_search_specs(args.optuna_search)
    optuna_enabled = args.optuna_trials > 0 or bool(optuna_search_specs)

    corpus_definition = corpora_registry.get_corpus(str(corpus))
    model_definition = model_registry.get_model(str(model_name))
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")
    if args.pipeline_local and not args.wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")

    resolved_pipeline_name = args.clearml_task_name or args.pipeline_name
    resolved_dataset_id = str(dataset_id or corpus_definition.dataset_id)
    resolved_source_split = source_split if source_split is not None else corpus_definition.split
    resolved_text_column = str(text_column or corpus_definition.text_column)
    resolved_tokenizer_model_name = str(tokenizer_model_name or "").strip()
    if not resolved_tokenizer_model_name:
        raise click.ClickException(
            "Training pipeline requires --tokenizer-model-name, or tokenizer_model_name in [train]. "
            "Run tokenizer training first and use its tokenizer model name."
        )

    if args.run_stage is not None and args.run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")
    if args.pipeline_controller_id is not None and args.run_stage is None:
        raise click.ClickException("--pipeline-controller-id must be used with --run-stage.")
    if optuna_enabled:
        if args.optuna_trials <= 0:
            raise click.ClickException("--optuna-trials must be greater than zero when --optuna-search is set.")
        if not optuna_search_specs:
            raise click.ClickException("--optuna-trials requires at least one --optuna-search spec.")
        if (
            args.run_stage is not None
            or args.run_until_stage is not None
            or args.pipeline_controller_id is not None
        ):
            raise click.ClickException(
                "Optuna runs the full model-training pipeline for every trial; "
                "do not combine it with --run-stage, --run-until-stage, or --pipeline-controller-id."
            )
        if not args.wait:
            raise click.ClickException(
                "Optuna requires --wait so each trial can read its evaluation objective metric."
            )

    parameter_filters = {
        "model": model_definition.name,
        "corpus": str(corpus),
        "tokenizer_model_name": resolved_tokenizer_model_name,
        "dataset_id": resolved_dataset_id,
        "source_split": resolved_source_split or "",
        "evaluation_partition": evaluation_partition,
    }
    if args.run_stage is not None or args.pipeline_controller_id is not None:
        if args.pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
                "Use --pipeline-queued when passing --run-stage or --pipeline-controller-id."
            )
        core_pipeline.resume_pipeline_controller_stage(
            stage_name=args.run_stage or lm_def.MODEL_STAGE,
            pipeline_controller_id=args.pipeline_controller_id,
            pipeline_name=resolved_pipeline_name,
            pipeline_version=args.pipeline_version,
            controller_queue=args.controller_queue,
            wait=args.wait,
            clearml_project=args.clearml_project,
            clearml_task_name=args.clearml_task_name,
            clearml_config_file=args.clearml_config_file,
            clearml_connectivity_check=args.clearml_connectivity_check,
            clearml_output_uri=args.clearml_output_uri,
            clearml_tags=args.clearml_tags,
            parameter_filters=parameter_filters,
            stage_dependencies=model_pipeline.MODEL_TRAINING_PIPELINE.stage_dependencies,
            stage_names=model_pipeline.MODEL_TRAINING_PIPELINE.stages,
        )
        return

    if optuna_enabled:
        model_training_runs.run_optuna_model_training(
            optuna_trials=args.optuna_trials,
            optuna_search_specs=optuna_search_specs,
            optuna_metric=args.optuna_metric,
            optuna_direction=args.optuna_direction,
            optuna_study_name=args.optuna_study_name,
            optuna_storage=args.optuna_storage,
            optuna_load_if_exists=args.optuna_load_if_exists,
            optuna_timeout_seconds=args.optuna_timeout_seconds,
            resolved_pipeline_name=resolved_pipeline_name,
            pipeline_version=args.pipeline_version,
            pipeline_local=args.pipeline_local,
            controller_queue=args.controller_queue,
            execution_queue=args.execution_queue,
            wait=args.wait,
            add_run_number=args.add_run_number,
            tokenizer_training_name=args.tokenizer_training_name,
            model_name=model_definition.name,
            corpus=str(corpus),
            resolved_tokenizer_model_name=resolved_tokenizer_model_name,
            resolved_dataset_id=resolved_dataset_id,
            resolved_source_split=resolved_source_split,
            resolved_text_column=resolved_text_column,
            streaming=bool(streaming),
            train_ratio=float(train_ratio),
            split_seed=int(split_seed),
            evaluation_partition=str(evaluation_partition),
            training_limit=resolved_training_limit,
            evaluation_limit=resolved_evaluation_limit,
            model_hyperparameters=model_hyperparameters,
            top_k=int(top_k),
            query_prompt=str(query_prompt),
            query_max_tokens=int(query_max_tokens),
            query_top_k=int(query_top_k),
            query_decoding=str(query_decoding),
            query_temperature=float(query_temperature),
            query_seed=int(query_seed) if query_seed is not None else None,
            text_normalization=str(text_normalization),
            clearml_project=args.clearml_project,
            clearml_config_file=args.clearml_config_file,
            clearml_connectivity_check=args.clearml_connectivity_check,
            clearml_output_uri=args.clearml_output_uri,
            clearml_tags=args.clearml_tags,
        )
        return

    model_training_runs.run_model_training_pipeline(
        resolved_pipeline_name=resolved_pipeline_name,
        pipeline_version=args.pipeline_version,
        pipeline_local=args.pipeline_local,
        controller_queue=args.controller_queue,
        execution_queue=args.execution_queue,
        wait=args.wait,
        add_run_number=args.add_run_number,
        run_until_stage=args.run_until_stage,
        tokenizer_training_name=args.tokenizer_training_name,
        model_name=model_definition.name,
        corpus=str(corpus),
        resolved_tokenizer_model_name=resolved_tokenizer_model_name,
        resolved_dataset_id=resolved_dataset_id,
        resolved_source_split=resolved_source_split,
        resolved_text_column=resolved_text_column,
        streaming=bool(streaming),
        train_ratio=float(train_ratio),
        split_seed=int(split_seed),
        evaluation_partition=str(evaluation_partition),
        training_limit=resolved_training_limit,
        evaluation_limit=resolved_evaluation_limit,
        model_hyperparameters=model_hyperparameters,
        top_k=int(top_k),
        query_prompt=str(query_prompt),
        query_max_tokens=int(query_max_tokens),
        query_top_k=int(query_top_k),
        query_decoding=str(query_decoding),
        query_temperature=float(query_temperature),
        query_seed=int(query_seed) if query_seed is not None else None,
        text_normalization=str(text_normalization),
        clearml_project=args.clearml_project,
        clearml_config_file=args.clearml_config_file,
        clearml_connectivity_check=args.clearml_connectivity_check,
        clearml_output_uri=args.clearml_output_uri,
        clearml_tags=args.clearml_tags,
    )
