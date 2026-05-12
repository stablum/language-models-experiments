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


ALL_STAGES = ("train", "evaluate", "query")
DATA_STAGES = ("train", "evaluate")
_CURRENT_VALUE_UNSET = object()


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


class ResolvedRunCfg(core_cfg.BaseCfg):
    """Stage cfg values after Click/config default resolution."""

    tokenizer_model_name: str
    model: model_pipeline.ModelCfg
    data: model_pipeline.DataCfg
    evaluation: model_pipeline.EvaluationCfg
    query: model_pipeline.QueryCfg


class DefaultResolver(core_cfg.BaseCfg):
    """Resolve Click values against stage-specific config defaults."""

    args: CliArgs
    stage_defaults: mt_defaults.StageDefaults
    ctx: click.Context | None

    def shared(self, parameter_name: str, *, stages: tuple[str, ...]) -> object:
        return self.stage_defaults.resolve_shared(
            self.ctx,
            parameter_name=parameter_name,
            current_value=getattr(self.args, parameter_name),
            stages=stages,
        )

    def stage(
        self,
        parameter_name: str,
        *,
        stage: str,
        stage_key: str | None = None,
        current_value: object = _CURRENT_VALUE_UNSET,
    ) -> object:
        value = (
            getattr(self.args, parameter_name)
            if current_value is _CURRENT_VALUE_UNSET
            else current_value
        )
        return self.stage_defaults.resolve_stage(
            self.ctx,
            parameter_name=parameter_name,
            current_value=value,
            stage=stage,
            stage_key=stage_key,
        )

    def limit(
        self,
        parameter_name: str,
        *,
        stage: str,
        current_value: int | None,
        global_limit: int | None,
    ) -> int | None:
        return self.stage_defaults.resolve_limit(
            self.ctx,
            parameter_name=parameter_name,
            current_value=current_value,
            stage=stage,
            global_limit=global_limit,
        )


def run(args: CliArgs) -> None:
    resolver = DefaultResolver(
        args=args,
        stage_defaults=mt_defaults.StageDefaults.load(),
        ctx=click.get_current_context(silent=True),
    )
    resolved_cfg = _resolve_run_cfg(resolver)
    optuna_cfg = _build_optuna_cfg(args)
    _validate_run_request(args, optuna_cfg)
    run_spec = _build_run_spec(args, resolved_cfg)

    if _should_resume_existing_stage(args):
        if args.pipeline_local:
            raise click.ClickException(
                "Existing PipelineController runs are resumed by re-enqueueing "
                "the controller task. "
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


def _resolve_run_cfg(resolver: DefaultResolver) -> ResolvedRunCfg:
    data_cfg = _resolve_data_cfg(resolver)
    model_name = str(resolver.shared("model_name", stages=ALL_STAGES))
    model_definition = model_registry.get_model(model_name)
    _require_model_pipeline_support(model_definition, model_name)

    return ResolvedRunCfg(
        tokenizer_model_name=_resolve_tokenizer_model_name(resolver),
        model=_resolve_model_cfg(
            resolver,
            model_name=model_definition.name,
        ),
        data=data_cfg,
        evaluation=_resolve_evaluation_cfg(resolver),
        query=_resolve_query_cfg(resolver),
    )


def _resolve_data_cfg(resolver: DefaultResolver) -> model_pipeline.DataCfg:
    corpus = str(resolver.shared("corpus", stages=ALL_STAGES))
    corpus_definition = corpora_registry.get_corpus(corpus)
    dataset_id = resolver.shared("dataset_id", stages=DATA_STAGES)
    source_split = resolver.shared("source_split", stages=DATA_STAGES)
    text_column = resolver.shared("text_column", stages=DATA_STAGES)

    return model_pipeline.DataCfg(
        corpus=corpus,
        dataset_id=str(dataset_id or corpus_definition.dataset_id),
        source_split=source_split if source_split is not None else corpus_definition.split,
        text_column=str(text_column or corpus_definition.text_column),
        streaming=bool(resolver.shared("streaming", stages=DATA_STAGES)),
        train_ratio=float(resolver.shared("train_ratio", stages=DATA_STAGES)),
        split_seed=int(resolver.shared("split_seed", stages=DATA_STAGES)),
    )


def _resolve_tokenizer_model_name(resolver: DefaultResolver) -> str:
    tokenizer_model_name = resolver.stage("tokenizer_model_name", stage="train")
    resolved = str(tokenizer_model_name or "").strip()
    if not resolved:
        raise click.ClickException(
            "Training pipeline requires --tokenizer-model-name, "
            "or tokenizer_model_name in [train]. "
            "Run tokenizer training first and use its tokenizer model name."
        )
    return resolved


def _resolve_model_cfg(
    resolver: DefaultResolver,
    *,
    model_name: str,
) -> model_pipeline.ModelCfg:
    text_normalization = resolver.stage("text_normalization", stage="train")
    return model_pipeline.ModelCfg(
        name=model_name,
        hyperparameters=_resolve_model_hyperparameters(resolver),
        limit=resolver.limit(
            "training_limit",
            stage="train",
            current_value=resolver.args.training_limit,
            global_limit=resolver.args.limit,
        ),
        text_normalization=str(text_normalization),
    )


def _resolve_model_hyperparameters(resolver: DefaultResolver) -> dict[str, object]:
    model_hyperparameters = lm_model_options.model_hyperparameters_from(
        resolver.args.model_dump()
    )
    return {
        name: resolver.stage(
            name,
            stage="train",
            current_value=model_hyperparameters[name],
        )
        for name in lm_model_options.MODEL_HYPERPARAMETER_NAMES
    }


def _resolve_evaluation_cfg(resolver: DefaultResolver) -> model_pipeline.EvaluationCfg:
    evaluation_partition = resolver.stage("evaluation_partition", stage="evaluate")
    top_k = resolver.stage("top_k", stage="evaluate")
    return model_pipeline.EvaluationCfg(
        partition=str(evaluation_partition),
        limit=resolver.limit(
            "evaluation_limit",
            stage="evaluate",
            current_value=resolver.args.evaluation_limit,
            global_limit=resolver.args.limit,
        ),
        top_k=int(top_k),
    )


def _resolve_query_cfg(resolver: DefaultResolver) -> model_pipeline.QueryCfg:
    query_seed = resolver.stage(
        "query_seed",
        stage="query",
        stage_key="seed",
    )
    return model_pipeline.QueryCfg(
        prompt=str(
            resolver.stage(
                "query_prompt",
                stage="query",
                stage_key="prompt",
            )
        ),
        max_tokens=int(
            resolver.stage(
                "query_max_tokens",
                stage="query",
                stage_key="max_tokens",
            )
        ),
        top_k=int(
            resolver.stage(
                "query_top_k",
                stage="query",
                stage_key="top_k",
            )
        ),
        decoding=str(
            resolver.stage(
                "query_decoding",
                stage="query",
                stage_key="decoding",
            )
        ),
        temperature=float(
            resolver.stage(
                "query_temperature",
                stage="query",
                stage_key="temperature",
            )
        ),
        seed=int(query_seed) if query_seed is not None else None,
    )


def _build_optuna_cfg(args: CliArgs) -> model_training_runs.OptunaCfg:
    return model_training_runs.OptunaCfg(
        trials=args.optuna_trials,
        search_specs=lm_optuna.parse_optuna_search_specs(args.optuna_search),
        metric=args.optuna_metric,
        direction=args.optuna_direction,
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=args.optuna_load_if_exists,
        timeout_seconds=args.optuna_timeout_seconds,
    )


def _build_run_spec(
    args: CliArgs,
    resolved_cfg: ResolvedRunCfg,
) -> model_training_runs.RunSpec:
    return model_training_runs.RunSpec(
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
        tokenizer_model_name=resolved_cfg.tokenizer_model_name,
        model=resolved_cfg.model,
        data=resolved_cfg.data,
        evaluation=resolved_cfg.evaluation,
        query=resolved_cfg.query,
    )


def _require_model_pipeline_support(
    model_definition: object,
    model_name: str,
) -> None:
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(f"Model does not support evaluation yet: {model_name}")
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")


def _validate_run_request(
    args: CliArgs,
    optuna_cfg: model_training_runs.OptunaCfg,
) -> None:
    if args.pipeline_local and not args.wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")
    if args.run_stage is not None and args.run_until_stage is not None:
        raise click.ClickException("--run-stage and --run-until-stage are mutually exclusive.")
    if args.pipeline_controller_id is not None and args.run_stage is None:
        raise click.ClickException("--pipeline-controller-id must be used with --run-stage.")

    if optuna_cfg.enabled:
        _validate_optuna_request(args, optuna_cfg)


def _validate_optuna_request(
    args: CliArgs,
    optuna_cfg: model_training_runs.OptunaCfg,
) -> None:
    if args.optuna_trials <= 0:
        raise click.ClickException(
            "--optuna-trials must be greater than zero when --optuna-search is set."
        )
    if not optuna_cfg.search_specs:
        raise click.ClickException("--optuna-trials requires at least one --optuna-search spec.")
    if _should_resume_existing_stage(args) or args.run_until_stage is not None:
        raise click.ClickException(
            "Optuna runs the full model-training pipeline for every trial; "
            "do not combine it with --run-stage, --run-until-stage, or --pipeline-controller-id."
        )
    if not args.wait:
        raise click.ClickException(
            "Optuna requires --wait so each trial can read its evaluation objective metric."
        )


def _should_resume_existing_stage(args: CliArgs) -> bool:
    return args.run_stage is not None or args.pipeline_controller_id is not None
