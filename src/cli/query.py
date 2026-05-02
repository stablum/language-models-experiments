"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.pipeline_common import (
    DEFAULT_PIPELINE_NAME,
    QUERY_STAGE,
    TRAINING_PIPELINE_STAGE_DEPENDENCIES,
    TRAINING_PIPELINE_STAGES,
    pipeline_options,
    pipeline_resume_option,
    resume_pipeline_controller_stage,
)
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import clearml_options


def load_query_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections(("train",))
    if "model_name" in train_defaults:
        defaults["model_name"] = train_defaults["model_name"]
    if "tokenizer_model_name" in train_defaults:
        defaults["tokenizer_model_name"] = train_defaults["tokenizer_model_name"]
    defaults.update(load_defaults_from_sections(("query",)))
    return defaults


@configured_command(
    "query",
    default_loader=load_query_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Query a registered language model.",
)
@pipeline_resume_option
@pipeline_options(default_name=DEFAULT_PIPELINE_NAME, default_local=False, default_wait=False)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_names()),
    default=DEFAULT_MODEL_NAME,
    show_default=True,
    help="Registered model to query.",
)
@click.option(
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name used by the training pipeline.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus used by the training pipeline.",
)
@click.option(
    "--model-task-id",
    default=None,
    help="Deprecated. Query resumes the model dependency from the pipeline controller.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Deprecated. Query resumes the model dependency from the pipeline controller.",
)
@click.option(
    "--prompt",
    default="",
    show_default=True,
    help="Text prefix to condition on.",
)
@click.option(
    "--max-tokens",
    type=click.IntRange(min=0),
    default=80,
    show_default=True,
    help="Maximum number of new tokens to generate.",
)
@click.option(
    "--top-k",
    type=click.IntRange(min=1),
    default=10,
    show_default=True,
    help="Number of likely next tokens to print for the prompt.",
)
@click.option(
    "--decoding",
    type=click.Choice(("sample", "most-probable")),
    default="sample",
    show_default=True,
    help="Generate by sampling or by choosing the most probable next token.",
)
@click.option(
    "--temperature",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Sampling temperature. Ignored for most-probable decoding.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible sampling.",
)
@clearml_options
def main(
    pipeline_name: str,
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    pipeline_controller_id: str | None,
    model_name: str,
    tokenizer_model_name: str | None,
    corpus: str,
    model_task_id: str | None,
    model_path: Path | None,
    prompt: str,
    max_tokens: int,
    top_k: int,
    decoding: str,
    temperature: float,
    seed: int | None,
    clearml_project: str,
    clearml_task_name: str | None,
    clearml_config_file: Path | None,
    clearml_connectivity_check: bool,
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    model_definition = get_model(model_name)
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")
    resolved_tokenizer_model_name = str(tokenizer_model_name or "").strip()
    if not resolved_tokenizer_model_name:
        raise click.ClickException(
            "Query requires --tokenizer-model-name, or tokenizer_model_name in [train]."
        )
    if model_task_id is not None or model_path is not None:
        raise click.ClickException(
            "Query now resumes the canonical ClearML pipeline DAG. "
            "Run train first in the same pipeline instead of passing --model-task-id or --model-path."
        )
    if pipeline_local:
        raise click.ClickException(
            "Existing PipelineController runs are resumed by re-enqueueing the controller task. "
            "Use --pipeline-queued for stage CLIs."
        )
    resume_pipeline_controller_stage(
        stage_name=QUERY_STAGE,
        pipeline_controller_id=pipeline_controller_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        controller_queue=controller_queue,
        wait=wait,
        clearml_project=clearml_project,
        clearml_task_name=clearml_task_name,
        clearml_config_file=clearml_config_file,
        clearml_connectivity_check=clearml_connectivity_check,
        clearml_output_uri=clearml_output_uri,
        clearml_tags=clearml_tags,
        parameter_filters={
            "model": model_definition.name,
            "tokenizer_model_name": resolved_tokenizer_model_name,
            "corpus": corpus,
        },
        stage_dependencies=TRAINING_PIPELINE_STAGE_DEPENDENCIES,
        stage_names=TRAINING_PIPELINE_STAGES,
    )


if __name__ == "__main__":
    main()
