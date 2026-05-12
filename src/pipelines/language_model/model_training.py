"""Definition for the model-training ClearML pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.ml_core import cfg as core_cfg
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_options
from src.pipelines.language_model import stage_entries
from src.pipelines.language_model import step_config


MODEL_TRAINING_PIPELINE = lm_def.PipelineDefinition(
    default_name=lm_def.DEFAULT_MODEL_TRAINING_NAME,
    stages=lm_def.MODEL_TRAINING_STAGES,
    stage_dependencies=lm_def.MODEL_TRAINING_STAGE_DEPENDENCIES,
)

MODEL_TRAINING_PIPELINE_PARAMETERS = {
    "model_name": "Registered model implementation to train, evaluate, and query.",
    **model_options.MODEL_HYPERPARAMETER_DESCRIPTIONS,
    "text_normalization": "Text normalization applied before model training.",
}


class ExecutionCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for ClearML execution of pipeline steps."""

    project_name: str
    output_uri: str | None
    tags: tuple[str, ...]
    config_file: Path | None
    queue: str | None


class TokenizerCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for the tokenizer dependency."""

    task_id: str
    model_name: str


class DataCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for corpus loading and project splits."""

    corpus: str
    dataset_id: str
    source_split: str | None
    text_column: str
    streaming: bool
    train_ratio: float
    split_seed: int


class ModelCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for model training."""

    name: str
    hyperparameters: Mapping[str, object]
    limit: int | None
    text_normalization: str


class EvaluationCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for the evaluation stage."""

    partition: str
    limit: int | None
    top_k: int


class QueryCfg(core_cfg.BaseCfg):
    """Cfg (configuration) for the query stage."""

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
    tokenizer: TokenizerCfg,
    data: DataCfg,
    model: ModelCfg,
    evaluation: EvaluationCfg,
    query: QueryCfg,
) -> None:
    model_hyperparameters = model_options.model_hyperparameters_from(
        model.hyperparameters
    )
    pipeline_parameters = {
        "model_name": model.name,
        **model_hyperparameters,
        "text_normalization": model.text_normalization,
    }
    _add_pipeline_parameters(pipeline, pipeline_parameters)

    cfg = step_config.StepCfg(
        pipeline_definition=MODEL_TRAINING_PIPELINE,
        project_name=execution.project_name,
        output_uri=execution.output_uri,
        tags=execution.tags,
        config_file=execution.config_file,
        queue=execution.queue,
    )
    cfg.add(
        pipeline,
        name=lm_def.MODEL_STAGE,
        function=stage_entries.train_model_stage_entry,
        function_kwargs={
            "tokenizer_task_id": tokenizer.task_id,
            "tokenizer_model_name": tokenizer.model_name,
            "model_name": _pipeline_parameter_ref("model_name"),
            "corpus": data.corpus,
            "dataset_id": data.dataset_id,
            "source_split": data.source_split,
            "text_column": data.text_column,
            "streaming": data.streaming,
            "limit": model.limit,
            "train_ratio": data.train_ratio,
            "split_seed": data.split_seed,
            **_pipeline_parameter_refs(model_options.MODEL_HYPERPARAMETER_NAMES),
            "text_normalization": _pipeline_parameter_ref("text_normalization"),
        },
        task_type="training",
    )
    cfg.add(
        pipeline,
        name=lm_def.EVALUATION_STAGE,
        function=stage_entries.evaluate_stage_entry,
        function_kwargs={
            "model_task_id": f"${{{lm_def.MODEL_STAGE}.id}}",
            "model_name": _pipeline_parameter_ref("model_name"),
            "corpus": data.corpus,
            "dataset_id": data.dataset_id,
            "source_split": data.source_split,
            "text_column": data.text_column,
            "streaming": data.streaming,
            "limit": evaluation.limit,
            "train_ratio": data.train_ratio,
            "split_seed": data.split_seed,
            "evaluation_partition": evaluation.partition,
            "top_k": evaluation.top_k,
            "tokenizer_model_name": tokenizer.model_name,
        },
        task_type="testing",
        parents=[lm_def.MODEL_STAGE],
    )
    cfg.add(
        pipeline,
        name=lm_def.QUERY_STAGE,
        function=stage_entries.query_stage_entry,
        function_kwargs={
            "model_task_id": f"${{{lm_def.MODEL_STAGE}.id}}",
            "model_name": _pipeline_parameter_ref("model_name"),
            "corpus": data.corpus,
            "prompt": query.prompt,
            "max_tokens": query.max_tokens,
            "top_k": query.top_k,
            "decoding": query.decoding,
            "temperature": query.temperature,
            "seed": query.seed,
            "tokenizer_model_name": tokenizer.model_name,
        },
        task_type="inference",
        parents=[lm_def.MODEL_STAGE],
    )


def _add_pipeline_parameters(
    pipeline: object,
    parameters: dict[str, object],
) -> None:
    add_parameter = getattr(pipeline, "add_parameter", None)
    if not callable(add_parameter):
        return

    for name, value in parameters.items():
        param_type = type(value).__name__ if value is not None else None
        add_parameter(
            name=name,
            default=value,
            description=MODEL_TRAINING_PIPELINE_PARAMETERS[name],
            param_type=param_type,
        )


def _pipeline_parameter_ref(name: str) -> str:
    return f"${{pipeline.{name}}}"


def _pipeline_parameter_refs(names: tuple[str, ...]) -> dict[str, str]:
    return {name: _pipeline_parameter_ref(name) for name in names}


__all__ = (
    "DataCfg",
    "EvaluationCfg",
    "ExecutionCfg",
    "ModelCfg",
    "MODEL_TRAINING_PIPELINE",
    "QueryCfg",
    "TokenizerCfg",
    "add_pipeline_steps",
)
