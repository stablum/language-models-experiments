"""Shared language-model pipeline constants and controller lookup helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import click

from src.ml_core import pipeline as core_pipeline
from src.ml_core import tracking


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
PIPELINE_STAGE_TITLES = {
    TOKENIZER_STAGE: "Tokenizer training",
    MODEL_STAGE: "Model training",
    EVALUATION_STAGE: "Evaluation",
    QUERY_STAGE: "Query",
}
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

stage_gate_callback = core_pipeline.make_stage_gate_callback(ALL_PIPELINE_STAGES)


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


def pipeline_stage_title(
    pipeline_definition: PipelineDefinition,
    stage: str,
) -> tuple[int, int, str]:
    return (
        pipeline_definition.stages.index(stage) + 1,
        len(pipeline_definition.stages),
        PIPELINE_STAGE_TITLES.get(stage, stage.replace("_", " ").title()),
    )


def standalone_stage_title(stage: str) -> tuple[int, int, str]:
    return 1, 1, PIPELINE_STAGE_TITLES.get(stage, stage.replace("_", " ").title())


def model_output_name(*, tokenizer_model_name: str | None, model_name: str) -> str | None:
    if not tokenizer_model_name:
        return None
    return f"{tokenizer_model_name}-{model_name}"


def configure_pipeline_control(
    task: object,
    *,
    run_stage: str | None,
    run_until_stage: str | None,
    updated_by: str,
    preserve_remote_control: bool = True,
) -> core_pipeline.PipelineControl:
    return core_pipeline.configure_pipeline_control(
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
    candidates = core_pipeline.list_pipeline_controller_candidates(
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
        if candidate.status not in core_pipeline.COMPLETED_STATUSES:
            reasons.append(f"{candidate.id}: controller status is {candidate.status}")
            continue
        if not core_pipeline.controller_parameters_match(
            candidate.id,
            parameter_filters,
        ):
            reasons.append(f"{candidate.id}: tokenizer parameters do not match")
            continue

        stage_tasks = core_pipeline.pipeline_stage_tasks(
            candidate.id,
            stage_names=TOKENIZER_TRAINING_STAGES,
        )
        completed_tokenizer_tasks = [
            task
            for task in stage_tasks.get(TOKENIZER_STAGE, ())
            if task.status in core_pipeline.COMPLETED_STATUSES
        ]
        if not completed_tokenizer_tasks:
            reasons.append(f"{candidate.id}: no completed {TOKENIZER_STAGE} stage task")
            continue

        for stage_task in completed_tokenizer_tasks:
            if tracking.task_has_output_model(stage_task.id, tokenizer_model_name):
                return TokenizerTrainingResolution(
                    controller_id=candidate.id,
                    tokenizer_task_id=stage_task.id,
                    tokenizer_model_name=tokenizer_model_name,
                    corpus=corpus,
                )
        reasons.append(
            f"{candidate.id}: completed {TOKENIZER_STAGE} task has no "
            f"{tokenizer_model_name!r} output model"
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
    output_model_name = model_output_name(
        tokenizer_model_name=tokenizer_model_name,
        model_name=model_name,
    )
    for candidate in candidates:
        if not core_pipeline.controller_parameters_match(
            candidate.id,
            parameter_filters,
        ):
            reasons.append(f"{candidate.id}: model-training parameters do not match")
            continue

        stage_tasks = core_pipeline.pipeline_stage_tasks(
            candidate.id,
            stage_names=MODEL_TRAINING_STAGES,
        )
        completed_model_tasks = [
            task
            for task in stage_tasks.get(MODEL_STAGE, ())
            if task.status in core_pipeline.COMPLETED_STATUSES
        ]
        if not completed_model_tasks:
            reasons.append(f"{candidate.id}: no completed {MODEL_STAGE} stage task")
            continue

        for stage_task in completed_model_tasks:
            if tracking.task_has_output_model(stage_task.id, output_model_name):
                return ModelTrainingResolution(
                    controller_id=candidate.id,
                    model_task_id=stage_task.id,
                    model_name=model_name,
                    tokenizer_model_name=tokenizer_model_name,
                    corpus=corpus,
                )
        reasons.append(
            f"{candidate.id}: completed {MODEL_STAGE} task has no "
            f"{output_model_name!r} output model"
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
) -> tuple[core_pipeline.ControllerCandidate, ...]:
    if pipeline_controller_id is None:
        return core_pipeline.list_pipeline_controller_candidates(
            pipeline_name=pipeline_name,
            pipeline_version=pipeline_version,
            clearml_project=clearml_project,
        )

    task = tracking.clearml_task(pipeline_controller_id)
    return (
        core_pipeline.ControllerCandidate(
            id=pipeline_controller_id,
            name=str(getattr(task, "name", "")),
            status=str(getattr(task, "status", "")),
            last_update=getattr(task, "last_update", None),
        ),
    )


__all__ = (
    "ALL_PIPELINE_STAGE_DEPENDENCIES",
    "ALL_PIPELINE_STAGES",
    "DEFAULT_MODEL_TRAINING_NAME",
    "DEFAULT_QUERY_NAME",
    "DEFAULT_TOKENIZER_TRAINING_NAME",
    "EVALUATION_STAGE",
    "MODEL_STAGE",
    "MODEL_TRAINING_STAGE_DEPENDENCIES",
    "MODEL_TRAINING_STAGES",
    "ModelTrainingResolution",
    "PipelineDefinition",
    "PIPELINE_STAGE_TITLES",
    "QUERY_STAGE",
    "QUERY_STAGE_DEPENDENCIES",
    "QUERY_STAGES",
    "TOKENIZER_STAGE",
    "TOKENIZER_TRAINING_STAGE_DEPENDENCIES",
    "TOKENIZER_TRAINING_STAGES",
    "TokenizerTrainingResolution",
    "configure_pipeline_control",
    "model_output_name",
    "pipeline_stage_title",
    "resolve_model_training_task",
    "resolve_tokenizer_training_task",
    "standalone_stage_title",
    "stage_gate_callback",
)
