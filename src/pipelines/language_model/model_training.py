"""Definition for the model-training ClearML pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.pipelines.language_model.definition import (
    DEFAULT_MODEL_TRAINING_NAME,
    DEFAULT_TOKENIZER_TRAINING_NAME,
    EVALUATION_STAGE,
    MODEL_STAGE,
    MODEL_TRAINING_STAGE_DEPENDENCIES,
    MODEL_TRAINING_STAGES,
    QUERY_STAGE,
    PipelineDefinition,
    output_uri_value,
    resolve_tokenizer_training_task,
    stage_gate_callback,
)
from src.pipelines.language_model.stage_entries import (
    evaluate_stage_entry,
    query_stage_entry,
    train_model_stage_entry,
)
from src.pipelines.language_model.model_options import (
    MODEL_HYPERPARAMETER_DESCRIPTIONS,
    MODEL_HYPERPARAMETER_NAMES,
    model_hyperparameters_from,
)
from src.pipelines.language_model.stages import (
    pipeline_artifact_monitors,
    pipeline_metric_monitors,
)


MODEL_TRAINING_PIPELINE = PipelineDefinition(
    default_name=DEFAULT_MODEL_TRAINING_NAME,
    stages=MODEL_TRAINING_STAGES,
    stage_dependencies=MODEL_TRAINING_STAGE_DEPENDENCIES,
)

MODEL_TRAINING_PIPELINE_PARAMETERS = {
    "model_name": "Registered model implementation to train, evaluate, and query.",
    **MODEL_HYPERPARAMETER_DESCRIPTIONS,
    "text_normalization": "Text normalization applied before model training.",
}


def add_pipeline_steps(
    pipeline: object,
    *,
    clearml_project: str,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
    clearml_config_file: Path | None,
    execution_queue: str | None,
    tokenizer_task_id: str,
    model_name: str,
    corpus: str,
    dataset_id: str,
    source_split: str | None,
    text_column: str,
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
) -> None:
    model_hyperparameters = model_hyperparameters_from(model_hyperparameters)
    pipeline_parameters = {
        "model_name": model_name,
        **model_hyperparameters,
        "text_normalization": text_normalization,
    }
    _add_pipeline_parameters(pipeline, pipeline_parameters)

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
        name=MODEL_STAGE,
        function=train_model_stage_entry,
        function_kwargs={
            "tokenizer_task_id": tokenizer_task_id,
            "model_name": _pipeline_parameter_ref("model_name"),
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": training_limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            **_pipeline_parameter_refs(MODEL_HYPERPARAMETER_NAMES),
            "text_normalization": _pipeline_parameter_ref("text_normalization"),
            **common_step_kwargs,
        },
        task_name=MODEL_STAGE,
        task_type="training",
        monitor_artifacts=artifact_monitors[MODEL_STAGE],
        monitor_metrics=metric_monitors[MODEL_STAGE],
        stage=MODEL_STAGE,
        **step_options,
    )
    pipeline.add_function_step(
        name=EVALUATION_STAGE,
        function=evaluate_stage_entry,
        function_kwargs={
            "model_task_id": f"${{{MODEL_STAGE}.id}}",
            "model_name": _pipeline_parameter_ref("model_name"),
            "corpus": corpus,
            "dataset_id": dataset_id,
            "source_split": source_split,
            "text_column": text_column,
            "streaming": streaming,
            "limit": evaluation_limit,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "evaluation_partition": evaluation_partition,
            "top_k": top_k,
            **common_step_kwargs,
        },
        parents=[MODEL_STAGE],
        task_name=EVALUATION_STAGE,
        task_type="testing",
        monitor_artifacts=artifact_monitors[EVALUATION_STAGE],
        monitor_metrics=metric_monitors[EVALUATION_STAGE],
        stage=EVALUATION_STAGE,
        **step_options,
    )
    pipeline.add_function_step(
        name=QUERY_STAGE,
        function=query_stage_entry,
        function_kwargs={
            "model_task_id": f"${{{MODEL_STAGE}.id}}",
            "model_name": _pipeline_parameter_ref("model_name"),
            "corpus": corpus,
            "prompt": query_prompt,
            "max_tokens": query_max_tokens,
            "top_k": query_top_k,
            "decoding": query_decoding,
            "temperature": query_temperature,
            "seed": query_seed,
            **common_step_kwargs,
        },
        parents=[MODEL_STAGE],
        task_name=QUERY_STAGE,
        task_type="inference",
        monitor_artifacts=artifact_monitors[QUERY_STAGE],
        monitor_metrics=metric_monitors[QUERY_STAGE],
        stage=QUERY_STAGE,
        **step_options,
    )


add_model_training_steps = add_pipeline_steps


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
    "DEFAULT_MODEL_TRAINING_NAME",
    "DEFAULT_TOKENIZER_TRAINING_NAME",
    "EVALUATION_STAGE",
    "MODEL_STAGE",
    "MODEL_TRAINING_PIPELINE",
    "MODEL_TRAINING_STAGE_DEPENDENCIES",
    "MODEL_TRAINING_STAGES",
    "QUERY_STAGE",
    "add_model_training_steps",
    "add_pipeline_steps",
    "resolve_tokenizer_training_task",
)
