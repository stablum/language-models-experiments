"""Definition for repeatable language-model query ClearML pipelines."""

from __future__ import annotations

from pathlib import Path

from src.ml_core import cfg as core_cfg
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import stage_entries
from src.pipelines.language_model import step_config


QUERY_PIPELINE = lm_def.PipelineDefinition(
    default_name=lm_def.DEFAULT_QUERY_NAME,
    stages=lm_def.QUERY_STAGES,
    stage_dependencies=lm_def.QUERY_STAGE_DEPENDENCIES,
)


class ExecutionCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for ClearML execution of query steps."""

    project_name: str
    output_uri: str | None
    tags: tuple[str, ...]
    config_file: Path | None
    queue: str | None


class ModelSourceCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for the model artifact queried by the pipeline."""

    source_pipeline_controller_id: str | None
    model_task_id: str | None
    model_path: Path | None
    model_name: str
    tokenizer_model_name: str | None
    corpus: str


class QueryCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for text generation."""

    prompt: str
    max_tokens: int
    top_k: int
    decoding: str
    temperature: float
    seed: int | None


def add_pipeline_steps(
    pipeline: object,
    *,
    execution: ExecutionCfg,
    source: ModelSourceCfg,
    query: QueryCfg,
) -> None:
    cfg = step_config.StepCfg(
        pipeline_definition=QUERY_PIPELINE,
        project_name=execution.project_name,
        output_uri=execution.output_uri,
        tags=execution.tags,
        config_file=execution.config_file,
        queue=execution.queue,
    )
    cfg.add(
        pipeline,
        name=lm_def.QUERY_STAGE,
        function=stage_entries.query_stage_entry,
        function_kwargs={
            "model_task_id": source.model_task_id,
            "model_path": (
                str(source.model_path) if source.model_path is not None else None
            ),
            "source_pipeline_controller_id": source.source_pipeline_controller_id,
            "model_name": source.model_name,
            "tokenizer_model_name": source.tokenizer_model_name,
            "corpus": source.corpus,
            "prompt": query.prompt,
            "max_tokens": query.max_tokens,
            "top_k": query.top_k,
            "decoding": query.decoding,
            "temperature": query.temperature,
            "seed": query.seed,
            "command": "src.cli.query",
        },
        task_type="inference",
    )


__all__ = (
    "ExecutionCfg",
    "ModelSourceCfg",
    "QUERY_PIPELINE",
    "QueryCfg",
    "add_pipeline_steps",
)
