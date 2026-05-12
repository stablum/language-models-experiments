"""Small function-step wrappers used by ClearML PipelineController."""

from __future__ import annotations


def _run_stage_entry(step_fn_name: str, **kwargs: object) -> str:
    from src.ml_core.cli import output as cli_output
    from src.pipelines.language_model import stages

    with cli_output.timestamped_cli_output():
        step_fn = getattr(stages, step_fn_name)
        return step_fn(**kwargs)


def train_tokenizer_stage_entry(**kwargs: object) -> str:
    return _run_stage_entry("train_tokenizer_step", **kwargs)


def train_model_stage_entry(**kwargs: object) -> str:
    return _run_stage_entry("train_model_pipeline_step", **kwargs)


def evaluate_stage_entry(**kwargs: object) -> str:
    return _run_stage_entry("evaluate_pipeline_step", **kwargs)


def query_stage_entry(**kwargs: object) -> str:
    return _run_stage_entry("query_pipeline_step", **kwargs)
