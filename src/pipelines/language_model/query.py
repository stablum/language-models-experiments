"""Definition for repeatable language-model query ClearML pipelines."""

from __future__ import annotations

from pathlib import Path

from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import stage_entries
from src.pipelines.language_model import step_config


DEFAULT_QUERY_NAME = lm_def.DEFAULT_QUERY_NAME
QUERY_STAGE = lm_def.QUERY_STAGE
QUERY_STAGE_DEPENDENCIES = lm_def.QUERY_STAGE_DEPENDENCIES
QUERY_STAGES = lm_def.QUERY_STAGES
resolve_model_training_task = lm_def.resolve_model_training_task

QUERY_PIPELINE = lm_def.PipelineDefinition(
    default_name=lm_def.DEFAULT_QUERY_NAME,
    stages=lm_def.QUERY_STAGES,
    stage_dependencies=lm_def.QUERY_STAGE_DEPENDENCIES,
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
    cfg = step_config.StepCfg(
        project_name=clearml_project,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
        config_file=clearml_config_file,
        queue=execution_queue,
    )
    cfg.add(
        pipeline,
        name=lm_def.QUERY_STAGE,
        function=stage_entries.query_stage_entry,
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
        },
        task_type="inference",
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
