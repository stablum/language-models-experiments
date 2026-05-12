"""Model-training CLI orchestration."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import model_training_defaults as mt_defaults
from src.cli import model_training_runs
from src.ml_core import cfg as core_cfg
from src.corpora import registry as corpora_registry
from src.models.core import registry as model_registry
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import optuna as lm_optuna


class CliArgs(core_cfg.BaseCfg):
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
    stage_defaults = mt_defaults.StageDefaults.load()
    all_stages = ("train", "evaluate", "query")
    data_stages = ("train", "evaluate")

    corpus = stage_defaults.resolve_shared(
        ctx,
        parameter_name="corpus",
        current_value=args.corpus,
        stages=all_stages,
    )
    model_name = stage_defaults.resolve_shared(
        ctx,
        parameter_name="model_name",
        current_value=args.model_name,
        stages=all_stages,
    )
    tokenizer_model_name = stage_defaults.resolve_stage(
        ctx,
        parameter_name="tokenizer_model_name",
        current_value=args.tokenizer_model_name,
        stage="train",
    )
    dataset_id = stage_defaults.resolve_shared(
        ctx,
        parameter_name="dataset_id",
        current_value=args.dataset_id,
        stages=data_stages,
    )
    source_split = stage_defaults.resolve_shared(
        ctx,
        parameter_name="source_split",
        current_value=args.source_split,
        stages=data_stages,
    )
    train_ratio = stage_defaults.resolve_shared(
        ctx,
        parameter_name="train_ratio",
        current_value=args.train_ratio,
        stages=data_stages,
    )
    split_seed = stage_defaults.resolve_shared(
        ctx,
        parameter_name="split_seed",
        current_value=args.split_seed,
        stages=data_stages,
    )
    text_column = stage_defaults.resolve_shared(
        ctx,
        parameter_name="text_column",
        current_value=args.text_column,
        stages=data_stages,
    )
    streaming = stage_defaults.resolve_shared(
        ctx,
        parameter_name="streaming",
        current_value=args.streaming,
        stages=data_stages,
    )
    evaluation_partition = stage_defaults.resolve_stage(
        ctx,
        parameter_name="evaluation_partition",
        current_value=args.evaluation_partition,
        stage="evaluate",
    )
    model_hyperparameters = lm_model_options.model_hyperparameters_from(
        args.model_dump()
    )
    model_hyperparameters = {
        name: stage_defaults.resolve_stage(
            ctx,
            parameter_name=name,
            current_value=model_hyperparameters[name],
            stage="train",
        )
        for name in lm_model_options.MODEL_HYPERPARAMETER_NAMES
    }
    top_k = stage_defaults.resolve_stage(
        ctx,
        parameter_name="top_k",
        current_value=args.top_k,
        stage="evaluate",
    )
    query_prompt = stage_defaults.resolve_stage(
        ctx,
        parameter_name="query_prompt",
        current_value=args.query_prompt,
        stage="query",
        stage_key="prompt",
    )
    query_max_tokens = stage_defaults.resolve_stage(
        ctx,
        parameter_name="query_max_tokens",
        current_value=args.query_max_tokens,
        stage="query",
        stage_key="max_tokens",
    )
    query_top_k = stage_defaults.resolve_stage(
        ctx,
        parameter_name="query_top_k",
        current_value=args.query_top_k,
        stage="query",
        stage_key="top_k",
    )
    query_decoding = stage_defaults.resolve_stage(
        ctx,
        parameter_name="query_decoding",
        current_value=args.query_decoding,
        stage="query",
        stage_key="decoding",
    )
    query_temperature = stage_defaults.resolve_stage(
        ctx,
        parameter_name="query_temperature",
        current_value=args.query_temperature,
        stage="query",
        stage_key="temperature",
    )
    query_seed = stage_defaults.resolve_stage(
        ctx,
        parameter_name="query_seed",
        current_value=args.query_seed,
        stage="query",
        stage_key="seed",
    )
    text_normalization = stage_defaults.resolve_stage(
        ctx,
        parameter_name="text_normalization",
        current_value=args.text_normalization,
        stage="train",
    )
    resolved_training_limit = stage_defaults.resolve_limit(
        ctx,
        parameter_name="training_limit",
        current_value=args.training_limit,
        stage="train",
        global_limit=args.limit,
    )
    resolved_evaluation_limit = stage_defaults.resolve_limit(
        ctx,
        parameter_name="evaluation_limit",
        current_value=args.evaluation_limit,
        stage="evaluate",
        global_limit=args.limit,
    )
    optuna_cfg = model_training_runs.OptunaCfg(
        trials=args.optuna_trials,
        search_specs=lm_optuna.parse_optuna_search_specs(args.optuna_search),
        metric=args.optuna_metric,
        direction=args.optuna_direction,
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=args.optuna_load_if_exists,
        timeout_seconds=args.optuna_timeout_seconds,
    )

    corpus_definition = corpora_registry.get_corpus(str(corpus))
    model_definition = model_registry.get_model(str(model_name))
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")
    if args.pipeline_local and not args.wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")

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
    run_spec = model_training_runs.RunSpec(
        pipeline=model_training_runs.PipelineRunCfg(
            name=args.pipeline_name,
            version=args.pipeline_version,
            local=args.pipeline_local,
            controller_queue=args.controller_queue,
            execution_queue=args.execution_queue,
            wait=args.wait,
            add_run_number=args.add_run_number,
            run_until_stage=args.run_until_stage,
        ),
        clearml=model_training_runs.ClearmlCfg(
            project=args.clearml_project,
            task_name=args.clearml_task_name,
            config_file=args.clearml_config_file,
            connectivity_check=args.clearml_connectivity_check,
            output_uri=args.clearml_output_uri,
            tags=args.clearml_tags,
        ),
        tokenizer_training_name=args.tokenizer_training_name,
        tokenizer_model_name=resolved_tokenizer_model_name,
        model=model_pipeline.ModelCfg(
            name=model_definition.name,
            hyperparameters=model_hyperparameters,
            limit=resolved_training_limit,
            text_normalization=str(text_normalization),
        ),
        data=model_pipeline.DataCfg(
            corpus=str(corpus),
            dataset_id=resolved_dataset_id,
            source_split=resolved_source_split,
            text_column=resolved_text_column,
            streaming=bool(streaming),
            train_ratio=float(train_ratio),
            split_seed=int(split_seed),
        ),
        evaluation=model_pipeline.EvaluationCfg(
            partition=str(evaluation_partition),
            limit=resolved_evaluation_limit,
            top_k=int(top_k),
        ),
        query=model_pipeline.QueryCfg(
            prompt=str(query_prompt),
            max_tokens=int(query_max_tokens),
            top_k=int(query_top_k),
            decoding=str(query_decoding),
            temperature=float(query_temperature),
            seed=int(query_seed) if query_seed is not None else None,
        ),
    )

    if optuna_cfg.enabled:
        if args.optuna_trials <= 0:
            raise click.ClickException("--optuna-trials must be greater than zero when --optuna-search is set.")
        if not optuna_cfg.search_specs:
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

    if args.run_stage is not None or args.pipeline_controller_id is not None:
        if args.pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
                "Use --pipeline-queued when passing --run-stage or --pipeline-controller-id."
            )
        model_training_runs.resume_model_training_stage(
            run_spec,
            stage_name=args.run_stage or lm_def.MODEL_STAGE,
            pipeline_controller_id=args.pipeline_controller_id,
        )
        return

    if optuna_cfg.enabled:
        model_training_runs.run_optuna_model_training(optuna_cfg, run_spec)
        return

    model_training_runs.run_model_training_pipeline(run_spec)
