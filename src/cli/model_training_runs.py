"""Execution helpers for model-training pipeline runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from src.ml_core import cfg as core_cfg
from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.models.core import registry as model_registry
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import optuna as lm_optuna


class PipelineRunCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for ClearML PipelineController execution."""

    name: str
    version: str
    local: bool
    controller_queue: str
    execution_queue: str | None
    wait: bool
    add_run_number: bool
    run_until_stage: str | None = None


class ClearmlCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for ClearML task metadata and connectivity."""

    project: str
    task_name: str | None
    config_file: Path | None
    connectivity_check: bool
    output_uri: str | None
    tags: tuple[str, ...]


class RunSpec(core_cfg.BaseCfg):
    """Resolved model-training run specification."""

    pipeline: PipelineRunCfg
    clearml: ClearmlCfg
    tokenizer_training_name: str
    tokenizer_model_name: str
    model: model_pipeline.ModelCfg
    data: model_pipeline.DataCfg
    evaluation: model_pipeline.EvaluationCfg
    query: model_pipeline.QueryCfg

    @property
    def resolved_pipeline_name(self) -> str:
        return self.clearml.task_name or self.pipeline.name

    @property
    def parameter_filters(self) -> dict[str, object]:
        return {
            "model": self.model.name,
            "corpus": self.data.corpus,
            "tokenizer_model_name": self.tokenizer_model_name,
            "dataset_id": self.data.dataset_id,
            "source_split": self.data.source_split or "",
            "text_column": self.data.text_column,
            "streaming": self.data.streaming,
            "train_ratio": self.data.train_ratio,
            "split_seed": self.data.split_seed,
            "training_limit": self.model.limit,
            **self.model.hyperparameters,
            "text_normalization": self.model.text_normalization,
            "evaluation_partition": self.evaluation.partition,
            "evaluation_limit": self.evaluation.limit,
            "top_k": self.evaluation.top_k,
            "query_prompt": self.query.prompt,
            "query_max_tokens": self.query.max_tokens,
            "query_top_k": self.query.top_k,
            "query_decoding": self.query.decoding,
            "query_temperature": self.query.temperature,
            "query_seed": self.query.seed,
        }


class OptunaCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for Optuna optimization."""

    trials: int
    search_specs: Sequence[lm_optuna.SearchSpec]
    metric: str
    direction: str
    study_name: str | None
    storage: str | None
    load_if_exists: bool
    timeout_seconds: int | None

    @property
    def enabled(self) -> bool:
        return self.trials > 0 or bool(self.search_specs)


def resume_model_training_stage(
    run_spec: RunSpec,
    *,
    stage_name: str,
    pipeline_controller_id: str | None,
) -> None:
    core_pipeline.resume_pipeline_controller_stage(
        stage_name=stage_name,
        pipeline_controller_id=pipeline_controller_id,
        pipeline_name=run_spec.pipeline.name,
        pipeline_version=run_spec.pipeline.version,
        controller_queue=run_spec.pipeline.controller_queue,
        wait=run_spec.pipeline.wait,
        clearml_project=run_spec.clearml.project,
        clearml_task_name=run_spec.clearml.task_name,
        clearml_config_file=run_spec.clearml.config_file,
        clearml_connectivity_check=run_spec.clearml.connectivity_check,
        clearml_output_uri=run_spec.clearml.output_uri,
        clearml_tags=run_spec.clearml.tags,
        parameter_filters=run_spec.parameter_filters,
        stage_dependencies=model_pipeline.MODEL_TRAINING_PIPELINE.stage_dependencies,
        stage_names=model_pipeline.MODEL_TRAINING_PIPELINE.stages,
    )


def run_optuna_model_training(
    optuna_cfg: OptunaCfg,
    run_spec: RunSpec,
) -> None:
    try:
        import optuna
    except ImportError as error:
        raise click.ClickException(
            "Optuna optimization requires the optuna Python package. "
            "Run `uv sync` before using --optuna-trials."
        ) from error

    study = optuna.create_study(
        study_name=optuna_cfg.study_name,
        storage=optuna_cfg.storage,
        direction=optuna_cfg.direction,
        load_if_exists=optuna_cfg.load_if_exists,
    )
    click.echo(f"Optuna study: {study.study_name}")
    click.echo(f"Optuna direction: {optuna_cfg.direction}")
    click.echo(f"Optuna objective metric: {optuna_cfg.metric}")
    click.echo(
        f"Optuna search space: {lm_optuna.describe_search_space(optuna_cfg.search_specs)}"
    )
    click.echo(f"Optuna trials: {optuna_cfg.trials}")

    def objective(trial: Any) -> float:
        sampled_params = lm_optuna.sample_trial_parameters(
            trial,
            optuna_cfg.search_specs,
        )
        trial_values = _trial_values(run_spec)
        trial_values.update(sampled_params)
        trial_tags = _trial_tags(
            run_spec.clearml.tags,
            study_name=study.study_name,
            trial_number=trial.number,
        )
        click.echo(
            f"Optuna trial {trial.number}: "
            + ", ".join(
                f"{name}={value!r}"
                for name, value in sorted(sampled_params.items())
            )
        )
        controller_id = run_model_training_pipeline(
            _trial_run_spec(run_spec, trial_values=trial_values, trial_tags=trial_tags),
            extra_controller_parameters={
                "optuna_study_name": study.study_name,
                "optuna_trial_number": trial.number,
                "optuna_metric": optuna_cfg.metric,
                "optuna_direction": optuna_cfg.direction,
                **{
                    f"optuna_{name}": value
                    for name, value in sampled_params.items()
                },
            },
        )
        objective_value = lm_optuna.load_objective_metric(
            controller_id=controller_id,
            metric_name=optuna_cfg.metric,
            evaluation_partition=run_spec.evaluation.partition,
        )
        trial.set_user_attr("pipeline_controller_id", controller_id)
        trial.set_user_attr("objective_metric", optuna_cfg.metric)
        click.echo(f"Optuna trial {trial.number} objective: {objective_value}")
        return objective_value

    study.optimize(
        objective,
        n_trials=optuna_cfg.trials,
        timeout=optuna_cfg.timeout_seconds,
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


def run_model_training_pipeline(
    run_spec: RunSpec,
    *,
    extra_controller_parameters: Mapping[str, object] | None = None,
) -> str:
    model_definition = model_registry.get_model(run_spec.model.name)
    if model_definition.evaluate is None or model_definition.evaluation_items is None:
        raise click.ClickException(
            f"Model does not support evaluation yet: {run_spec.model.name}"
        )
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(
            f"Model does not support querying yet: {run_spec.model.name}"
        )

    settings = tracking.clearml_settings(
        project_name=run_spec.clearml.project,
        task_name=run_spec.resolved_pipeline_name,
        config_file=run_spec.clearml.config_file,
        connectivity_check=run_spec.clearml.connectivity_check,
        output_uri=run_spec.clearml.output_uri,
        tags=run_spec.clearml.tags,
    )
    resolved_config_file = tracking.configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        tracking.assert_clearml_endpoints_reachable(
            resolved_config_file,
            settings.output_uri,
        )

    tokenizer_resolution = lm_def.resolve_tokenizer_training_task(
        tokenizer_training_name=run_spec.tokenizer_training_name,
        clearml_project=settings.project_name,
        corpus=run_spec.data.corpus,
        tokenizer_model_name=run_spec.tokenizer_model_name,
    )

    pipeline = core_pipeline.build_pipeline_controller(
        pipeline_name=run_spec.resolved_pipeline_name,
        pipeline_version=run_spec.pipeline.version,
        clearml_project=settings.project_name,
        clearml_tags=settings.tags,
        clearml_output_uri=settings.output_uri,
        add_run_number=run_spec.pipeline.add_run_number,
    )
    lm_def.configure_pipeline_control(
        pipeline.task,
        run_stage=None,
        run_until_stage=run_spec.pipeline.run_until_stage,
        updated_by="pipeline-cli",
    )
    controller_params: dict[str, object] = {
        "model": model_definition.name,
        "corpus": run_spec.data.corpus,
        "tokenizer_model_name": run_spec.tokenizer_model_name,
        "tokenizer_training_name": run_spec.tokenizer_training_name,
        "tokenizer_training_controller_id": tokenizer_resolution.controller_id,
        "tokenizer_task_id": tokenizer_resolution.tokenizer_task_id,
        "dataset_id": run_spec.data.dataset_id,
        "source_split": run_spec.data.source_split or "",
        "text_column": run_spec.data.text_column,
        "streaming": run_spec.data.streaming,
        "train_ratio": run_spec.data.train_ratio,
        "split_seed": run_spec.data.split_seed,
        "training_limit": run_spec.model.limit,
        **run_spec.model.hyperparameters,
        "text_normalization": run_spec.model.text_normalization,
        "evaluation_partition": run_spec.evaluation.partition,
        "evaluation_limit": run_spec.evaluation.limit,
        "top_k": run_spec.evaluation.top_k,
        "query_prompt": run_spec.query.prompt,
        "query_max_tokens": run_spec.query.max_tokens,
        "query_top_k": run_spec.query.top_k,
        "query_decoding": run_spec.query.decoding,
        "query_temperature": run_spec.query.temperature,
        "query_seed": run_spec.query.seed,
    }
    if extra_controller_parameters:
        controller_params.update(extra_controller_parameters)
    core_pipeline.connect_controller_experiment_parameters(
        pipeline.task,
        controller_params,
    )
    model_pipeline.add_pipeline_steps(
        pipeline,
        execution=model_pipeline.ExecutionCfg(
            project_name=settings.project_name,
            output_uri=settings.output_uri,
            tags=settings.tags,
            config_file=resolved_config_file if run_spec.pipeline.local else None,
            queue=None if run_spec.pipeline.local else run_spec.pipeline.execution_queue,
        ),
        tokenizer=model_pipeline.TokenizerCfg(
            task_id=tokenizer_resolution.tokenizer_task_id,
            model_name=run_spec.tokenizer_model_name,
        ),
        model=run_spec.model.model_copy(update={"name": model_definition.name}),
        data=run_spec.data,
        evaluation=run_spec.evaluation,
        query=run_spec.query,
    )

    click.echo(f"ClearML pipeline: {settings.project_name}/{run_spec.resolved_pipeline_name}")
    click.echo(f"Pipeline version: {run_spec.pipeline.version}")
    click.echo(f"Tokenizer model: {run_spec.tokenizer_model_name}")
    click.echo(f"Tokenizer pipeline controller task ID: {tokenizer_resolution.controller_id}")
    click.echo(f"Tokenizer stage task ID: {tokenizer_resolution.tokenizer_task_id}")
    click.echo(f"Pipeline controller task ID: {pipeline.task.id}")
    task_url = pipeline.task.get_output_log_web_page()
    if task_url:
        click.echo(f"Pipeline controller URL: {task_url}")
    if run_spec.pipeline.run_until_stage is not None:
        click.echo(f"Run until stage: {run_spec.pipeline.run_until_stage}")
    click.echo(
        "Stage tasks: "
        f"{lm_def.MODEL_STAGE}, {lm_def.EVALUATION_STAGE}, {lm_def.QUERY_STAGE}"
    )

    if run_spec.pipeline.local:
        click.echo("Execution mode: local ClearML PipelineController")
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        click.echo(f"Execution mode: queued controller on {run_spec.pipeline.controller_queue}")
        if run_spec.pipeline.execution_queue is not None:
            click.echo(f"Step execution queue: {run_spec.pipeline.execution_queue}")
        pipeline.start(
            queue=run_spec.pipeline.controller_queue,
            wait=run_spec.pipeline.wait,
        )

    click.echo("ClearML pipeline submitted.")
    if run_spec.pipeline.wait:
        core_pipeline.assert_pipeline_finished_successfully(pipeline)
        core_pipeline.print_stage_task_ids(
            pipeline.task.id,
            model_pipeline.MODEL_TRAINING_PIPELINE.stages,
            stage_names=model_pipeline.MODEL_TRAINING_PIPELINE.stages,
        )
        click.echo("ClearML pipeline run completed.")
    return str(pipeline.task.id)


def _trial_values(run_spec: RunSpec) -> dict[str, object]:
    return {
        "model_name": run_spec.model.name,
        **run_spec.model.hyperparameters,
        "top_k": run_spec.evaluation.top_k,
        "query_max_tokens": run_spec.query.max_tokens,
        "query_top_k": run_spec.query.top_k,
        "query_decoding": run_spec.query.decoding,
        "query_temperature": run_spec.query.temperature,
        "query_seed": run_spec.query.seed,
    }


def _trial_tags(
    tags: tuple[str, ...],
    *,
    study_name: str,
    trial_number: int,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *tags,
                "optuna",
                f"optuna-study-{study_name}",
                f"optuna-trial-{trial_number}",
            )
        )
    )


def _trial_run_spec(
    run_spec: RunSpec,
    *,
    trial_values: Mapping[str, object],
    trial_tags: tuple[str, ...],
) -> RunSpec:
    query_seed = trial_values["query_seed"]
    return run_spec.model_copy(
        update={
            "clearml": run_spec.clearml.model_copy(update={"tags": trial_tags}),
            "model": run_spec.model.model_copy(
                update={
                    "name": str(trial_values["model_name"]),
                    "hyperparameters": lm_model_options.model_hyperparameters_from(
                        trial_values
                    ),
                }
            ),
            "evaluation": run_spec.evaluation.model_copy(
                update={"top_k": int(trial_values["top_k"])}
            ),
            "query": run_spec.query.model_copy(
                update={
                    "max_tokens": int(trial_values["query_max_tokens"]),
                    "top_k": int(trial_values["query_top_k"]),
                    "decoding": str(trial_values["query_decoding"]),
                    "temperature": float(trial_values["query_temperature"]),
                    "seed": int(query_seed) if query_seed is not None else None,
                }
            ),
        }
    )
