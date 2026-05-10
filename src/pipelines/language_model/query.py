"""Definition for repeatable language-model query ClearML pipelines."""

from __future__ import annotations

from pathlib import Path

from src.pipelines.language_model.definition import (
    DEFAULT_QUERY_NAME,
    QUERY_STAGE,
    QUERY_STAGE_DEPENDENCIES,
    QUERY_STAGES,
    PipelineDefinition,
    output_uri_value,
    resolve_model_training_task,
    stage_gate_callback,
)
from src.pipelines.language_model.stage_entries import query_stage_entry
from src.pipelines.language_model.stages import (
    pipeline_artifact_monitors,
    pipeline_metric_monitors,
)


QUERY_PIPELINE = PipelineDefinition(
    default_name=DEFAULT_QUERY_NAME,
    stages=QUERY_STAGES,
    stage_dependencies=QUERY_STAGE_DEPENDENCIES,
)


def add_pipeline_steps(
    pipeline: object,
    *,
    clearml_project: str,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    clearml_config_file: Path | None,
    execution_queue: str | None,
    source_pipeline_controller_id: str | None,
    model_task_id: str | None,
    model_path: Path | None,
    model_name: str,
    corpus: str,
    prompt: str,
    max_tokens: int,
    top_k: int,
    decoding: str,
    temperature: float,
    seed: int | None,
) -> None:
    artifact_monitors = pipeline_artifact_monitors()
    metric_monitors = pipeline_metric_monitors()
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
        name=QUERY_STAGE,
        function=query_stage_entry,
        function_kwargs={
            "model_task_id": model_task_id,
            "model_path": str(model_path) if model_path is not None else None,
            "source_pipeline_controller_id": source_pipeline_controller_id,
            "model_name": model_name,
            "corpus": corpus,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "decoding": decoding,
            "temperature": temperature,
            "seed": seed,
            "command": "src.cli.query",
            **common_step_kwargs,
        },
        task_name=QUERY_STAGE,
        task_type="inference",
        monitor_artifacts=artifact_monitors[QUERY_STAGE],
        monitor_metrics=metric_monitors[QUERY_STAGE],
        stage=QUERY_STAGE,
        **step_options,
    )


__all__ = (
    "DEFAULT_QUERY_NAME",
    "QUERY_PIPELINE",
    "QUERY_STAGE",
    "QUERY_STAGE_DEPENDENCIES",
    "QUERY_STAGES",
    "add_pipeline_steps",
    "resolve_model_training_task",
)
