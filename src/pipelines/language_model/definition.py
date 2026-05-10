"""Shared language-model pipeline constants and controller lookup helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import click

from src.ml_core.pipeline import (
    ACTIVE_STATUSES,
    COMPLETED_STATUSES,
    DEFAULT_CONTROLLER_QUEUE,
    DEFAULT_PIPELINE_VERSION,
    FAILED_STATUSES,
    PIPELINE_CONTROL_MODE,
    PIPELINE_CONTROL_RUN_STAGE,
    PIPELINE_CONTROL_RUN_UNTIL_STAGE,
    PIPELINE_CONTROL_SECTION,
    PIPELINE_CONTROL_UPDATED_BY,
    PIPELINE_MODE_ALL,
    PIPELINE_MODE_RUN_STAGE,
    PIPELINE_MODE_RUN_UNTIL,
    TERMINAL_STATUSES,
    ControllerCandidate,
    PipelineControl,
    StageEligibility,
    StageTask,
    assert_controller_can_run_stage,
    assert_controller_task_succeeded,
    assert_pipeline_finished_successfully,
    build_pipeline_controller,
    configure_pipeline_control as configure_generic_pipeline_control,
    connect_controller_experiment_parameters,
    controller_parameters_match,
    list_pipeline_controller_candidates,
    make_stage_gate_callback,
    output_uri_value,
    pipeline_control_from_task,
    pipeline_options,
    pipeline_resume_option,
    pipeline_stage_eligibility,
    pipeline_stage_tasks,
    print_stage_task_ids,
    project_version,
    resolve_pipeline_controller_id,
    resume_pipeline_controller_stage,
    stage_allowed_by_control,
    validate_stage_selection,
    wait_for_controller_completion,
)
from src.ml_core.tracking import clearml_task


DEFAULT_MODEL_TRAINING_NAME = "model-training"
DEFAULT_TOKENIZER_TRAINING_NAME = "tokenizer-training"
DEFAULT_QUERY_NAME = "query"

TOKENIZER_STAGE = "train_tokenizer"
MODEL_STAGE = "train_model"
EVALUATION_STAGE = "evaluate"
QUERY_STAGE = "query"
ALL_PIPELINE_STAGES = (TOKENIZER_STAGE, MODEL_STAGE, EVALUATION_STAGE, QUERY_STAGE)
TOKENIZER_TRAINING_STAGES = (TOKENIZER_STAGE,)
MODEL_TRAINING_STAGES = (MODEL_STAGE, EVALUATION_STAGE, QUERY_STAGE)
QUERY_STAGES = (QUERY_STAGE,)
ALL_PIPELINE_STAGE_DEPENDENCIES = {
    TOKENIZER_STAGE: (),
    MODEL_STAGE: (TOKENIZER_STAGE,),
    EVALUATION_STAGE: (MODEL_STAGE,),
    QUERY_STAGE: (MODEL_STAGE,),
}
TOKENIZER_TRAINING_STAGE_DEPENDENCIES = {
    TOKENIZER_STAGE: (),
}
MODEL_TRAINING_STAGE_DEPENDENCIES = {
    MODEL_STAGE: (),
    EVALUATION_STAGE: (MODEL_STAGE,),
    QUERY_STAGE: (MODEL_STAGE,),
}
QUERY_STAGE_DEPENDENCIES = {
    QUERY_STAGE: (),
}
TOKENIZER_MODEL_ARTIFACT = "sentencepiece-model"

stage_gate_callback = make_stage_gate_callback(ALL_PIPELINE_STAGES)


@dataclass(frozen=True)
class PipelineDefinition:
    default_name: str
    stages: tuple[str, ...]
    stage_dependencies: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class TokenizerTrainingResolution:
    controller_id: str
    tokenizer_task_id: str
    tokenizer_model_name: str
    corpus: str


@dataclass(frozen=True)
class ModelTrainingResolution:
    controller_id: str
    model_task_id: str
    model_name: str
    tokenizer_model_name: str
    corpus: str


def configure_pipeline_control(
    task: object,
    *,
    run_stage: str | None,
    run_until_stage: str | None,
    updated_by: str,
    preserve_remote_control: bool = True,
) -> PipelineControl:
    return configure_generic_pipeline_control(
        task,
        run_stage=run_stage,
        run_until_stage=run_until_stage,
        updated_by=updated_by,
        stage_names=ALL_PIPELINE_STAGES,
        preserve_remote_control=preserve_remote_control,
    )


def resolve_tokenizer_training_task(
    *,
    tokenizer_training_name: str,
    clearml_project: str,
    corpus: str,
    tokenizer_model_name: str,
) -> TokenizerTrainingResolution:
    candidates = list_pipeline_controller_candidates(
        pipeline_name=tokenizer_training_name,
        pipeline_version=None,
        clearml_project=clearml_project,
    )
    parameter_filters = {
        "corpus": corpus,
        "tokenizer_model_name": tokenizer_model_name,
    }
    reasons: list[str] = []
    for candidate in candidates:
        if candidate.status not in COMPLETED_STATUSES:
            reasons.append(f"{candidate.id}: controller status is {candidate.status}")
            continue
        if not controller_parameters_match(candidate.id, parameter_filters):
            reasons.append(f"{candidate.id}: tokenizer parameters do not match")
            continue

        stage_tasks = pipeline_stage_tasks(
            candidate.id,
            stage_names=TOKENIZER_TRAINING_STAGES,
        )
        completed_tokenizer_tasks = [
            task
            for task in stage_tasks.get(TOKENIZER_STAGE, ())
            if task.status in COMPLETED_STATUSES
        ]
        if not completed_tokenizer_tasks:
            reasons.append(f"{candidate.id}: no completed {TOKENIZER_STAGE} stage task")
            continue

        for stage_task in completed_tokenizer_tasks:
            if _task_has_artifact(stage_task.id, TOKENIZER_MODEL_ARTIFACT):
                return TokenizerTrainingResolution(
                    controller_id=candidate.id,
                    tokenizer_task_id=stage_task.id,
                    tokenizer_model_name=tokenizer_model_name,
                    corpus=corpus,
                )
        reasons.append(
            f"{candidate.id}: completed {TOKENIZER_STAGE} task has no "
            f"{TOKENIZER_MODEL_ARTIFACT!r} artifact"
        )

    detail = ""
    if reasons:
        detail = " Checked candidates: " + "; ".join(reasons[:5])
    raise click.ClickException(
        "Could not find a completed tokenizer-training run for "
        f"corpus={corpus!r} and tokenizer_model_name={tokenizer_model_name!r}. "
        f"Run `python -m src.cli.tokenizer_training` first, or change "
        f"--tokenizer-training-name from {tokenizer_training_name!r}."
        f"{detail}"
    )


def resolve_model_training_task(
    *,
    pipeline_name: str,
    pipeline_version: str | None,
    clearml_project: str,
    model_name: str,
    tokenizer_model_name: str,
    corpus: str,
    pipeline_controller_id: str | None = None,
) -> ModelTrainingResolution:
    parameter_filters = {
        "model": model_name,
        "tokenizer_model_name": tokenizer_model_name,
        "corpus": corpus,
    }
    candidates = _model_training_candidates(
        pipeline_controller_id=pipeline_controller_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=clearml_project,
    )

    reasons: list[str] = []
    for candidate in candidates:
        if not controller_parameters_match(candidate.id, parameter_filters):
            reasons.append(f"{candidate.id}: model-training parameters do not match")
            continue

        stage_tasks = pipeline_stage_tasks(
            candidate.id,
            stage_names=MODEL_TRAINING_STAGES,
        )
        completed_model_tasks = [
            task
            for task in stage_tasks.get(MODEL_STAGE, ())
            if task.status in COMPLETED_STATUSES
        ]
        if not completed_model_tasks:
            reasons.append(f"{candidate.id}: no completed {MODEL_STAGE} stage task")
            continue

        return ModelTrainingResolution(
            controller_id=candidate.id,
            model_task_id=completed_model_tasks[0].id,
            model_name=model_name,
            tokenizer_model_name=tokenizer_model_name,
            corpus=corpus,
        )

    detail = ""
    if reasons:
        detail = " Checked candidates: " + "; ".join(reasons[:5])
    controller_hint = (
        f" controller {pipeline_controller_id!r}"
        if pipeline_controller_id is not None
        else f" pipeline {pipeline_name!r}"
    )
    raise click.ClickException(
        "Could not find a completed model-training run for "
        f"model={model_name!r}, corpus={corpus!r}, and "
        f"tokenizer_model_name={tokenizer_model_name!r} in{controller_hint}. "
        "Run model training first, pass --model-task-id, or pass --model-path."
        f"{detail}"
    )


def _model_training_candidates(
    *,
    pipeline_controller_id: str | None,
    pipeline_name: str,
    pipeline_version: str | None,
    clearml_project: str,
) -> tuple[ControllerCandidate, ...]:
    if pipeline_controller_id is None:
        return list_pipeline_controller_candidates(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            clearml_project=clearml_project,
        )

    task = clearml_task(pipeline_controller_id)
    return (
        ControllerCandidate(
            id=pipeline_controller_id,
            name=str(getattr(task, "name", "")),
            status=str(getattr(task, "status", "")),
            last_update=getattr(task, "last_update", None),
        ),
    )


def _task_has_artifact(task_id: str, artifact_name: str) -> bool:
    task = clearml_task(task_id)
    artifacts = getattr(task, "artifacts", {}) or {}
    return artifact_name in artifacts


__all__ = (
    "ACTIVE_STATUSES",
    "ALL_PIPELINE_STAGE_DEPENDENCIES",
    "ALL_PIPELINE_STAGES",
    "COMPLETED_STATUSES",
    "ControllerCandidate",
    "DEFAULT_CONTROLLER_QUEUE",
    "DEFAULT_MODEL_TRAINING_NAME",
    "DEFAULT_PIPELINE_VERSION",
    "DEFAULT_QUERY_NAME",
    "DEFAULT_TOKENIZER_TRAINING_NAME",
    "EVALUATION_STAGE",
    "FAILED_STATUSES",
    "MODEL_STAGE",
    "MODEL_TRAINING_STAGE_DEPENDENCIES",
    "MODEL_TRAINING_STAGES",
    "ModelTrainingResolution",
    "PIPELINE_CONTROL_MODE",
    "PIPELINE_CONTROL_RUN_STAGE",
    "PIPELINE_CONTROL_RUN_UNTIL_STAGE",
    "PIPELINE_CONTROL_SECTION",
    "PIPELINE_CONTROL_UPDATED_BY",
    "PIPELINE_MODE_ALL",
    "PIPELINE_MODE_RUN_STAGE",
    "PIPELINE_MODE_RUN_UNTIL",
    "PipelineControl",
    "PipelineDefinition",
    "QUERY_STAGE",
    "QUERY_STAGE_DEPENDENCIES",
    "QUERY_STAGES",
    "StageEligibility",
    "StageTask",
    "TERMINAL_STATUSES",
    "TOKENIZER_MODEL_ARTIFACT",
    "TOKENIZER_STAGE",
    "TOKENIZER_TRAINING_STAGE_DEPENDENCIES",
    "TOKENIZER_TRAINING_STAGES",
    "TokenizerTrainingResolution",
    "assert_controller_can_run_stage",
    "assert_controller_task_succeeded",
    "assert_pipeline_finished_successfully",
    "build_pipeline_controller",
    "configure_pipeline_control",
    "connect_controller_experiment_parameters",
    "controller_parameters_match",
    "list_pipeline_controller_candidates",
    "output_uri_value",
    "pipeline_control_from_task",
    "pipeline_options",
    "pipeline_resume_option",
    "pipeline_stage_eligibility",
    "pipeline_stage_tasks",
    "print_stage_task_ids",
    "project_version",
    "resolve_pipeline_controller_id",
    "resolve_model_training_task",
    "resolve_tokenizer_training_task",
    "resume_pipeline_controller_stage",
    "stage_allowed_by_control",
    "stage_gate_callback",
    "validate_stage_selection",
    "wait_for_controller_completion",
)
