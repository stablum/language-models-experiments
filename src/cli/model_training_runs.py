"""Execution helpers for model-training pipeline runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click

from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking
from src.models.core import registry as model_registry
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options as lm_model_options
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import optuna as lm_optuna


def run_optuna_model_training(
    *,
    optuna_trials: int,
    optuna_search_specs: Sequence[lm_optuna.SearchSpec],
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
    click.echo(
        f"Optuna search space: {lm_optuna.describe_search_space(optuna_search_specs)}"
    )
    click.echo(f"Optuna trials: {optuna_trials}")

    def objective(trial: Any) -> float:
        sampled_parameters = lm_optuna.sample_trial_parameters(
            trial,
            optuna_search_specs,
        )
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
        controller_id = run_model_training_pipeline(
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
            model_hyperparameters=lm_model_options.model_hyperparameters_from(
                trial_values
            ),
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
        objective_value = lm_optuna.load_objective_metric(
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


def run_model_training_pipeline(
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

    settings = tracking.clearml_settings(
        project_name=clearml_project,
        task_name=resolved_pipeline_name,
        config_file=clearml_config_file,
        connectivity_check=clearml_connectivity_check,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
    )
    resolved_config_file = tracking.configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        tracking.assert_clearml_endpoints_reachable(
            resolved_config_file,
            settings.output_uri,
        )

    tokenizer_resolution = lm_def.resolve_tokenizer_training_task(
        tokenizer_training_name=tokenizer_training_name,
        clearml_project=settings.project_name,
        corpus=corpus,
        tokenizer_model_name=resolved_tokenizer_model_name,
    )

    pipeline = core_pipeline.build_pipeline_controller(
        pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        clearml_tags=settings.tags,
        clearml_output_uri=settings.output_uri,
        add_run_number=add_run_number,
    )
    lm_def.configure_pipeline_control(
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
    core_pipeline.connect_controller_experiment_parameters(
        pipeline.task,
        controller_parameters,
    )
    model_pipeline.add_pipeline_steps(
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
    click.echo(
        "Stage tasks: "
        f"{lm_def.MODEL_STAGE}, {lm_def.EVALUATION_STAGE}, {lm_def.QUERY_STAGE}"
    )

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
        core_pipeline.assert_pipeline_finished_successfully(pipeline)
        core_pipeline.print_stage_task_ids(
            pipeline.task.id,
            model_pipeline.MODEL_TRAINING_PIPELINE.stages,
            stage_names=model_pipeline.MODEL_TRAINING_PIPELINE.stages,
        )
        click.echo("ClearML pipeline run completed.")
    return str(pipeline.task.id)
