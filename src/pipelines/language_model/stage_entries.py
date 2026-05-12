"""Small function-step wrappers used by ClearML PipelineController."""

from __future__ import annotations

from collections.abc import Callable

from src.pipelines.language_model import model_options as lm_model_options


def _run_stage_entry(step_fn: Callable[..., str], **kwargs: object) -> str:
    from src.ml_core.cli import output as cli_output

    with cli_output.timestamped_cli_output():
        return step_fn(**kwargs)


def train_tokenizer_stage_entry(**kwargs: object) -> str:
    from src.pipelines.language_model import stages

    return _run_stage_entry(stages.train_tokenizer_step, **kwargs)


def train_model_stage_entry(**kwargs: object) -> str:
    from src.pipelines.language_model import stages

    return _run_stage_entry(
        stages.train_model_pipeline_step,
        **_group_model_hyperparameters(kwargs),
    )


def evaluate_stage_entry(**kwargs: object) -> str:
    from src.pipelines.language_model import stages

    return _run_stage_entry(stages.evaluate_pipeline_step, **kwargs)


def query_stage_entry(**kwargs: object) -> str:
    from src.pipelines.language_model import query_stage

    return _run_stage_entry(query_stage.query_pipeline_step, **kwargs)


def _group_model_hyperparameters(kwargs: dict[str, object]) -> dict[str, object]:
    hyperparameters = {
        name: kwargs.pop(name)
        for name in lm_model_options.MODEL_HYPERPARAMETER_NAMES
        if name in kwargs
    }
    if hyperparameters:
        kwargs["model_hyperparameters"] = hyperparameters
    return kwargs
