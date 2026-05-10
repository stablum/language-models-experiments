"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import stage_resume
from src.ml_core.cli.config import configured_command, load_defaults_from_sections
from src.ml_core.cli.output import emit_stage_title
from src.pipelines.language_model.definition import resolve_model_training_task
from src.pipelines.language_model.model_training import MODEL_TRAINING_PIPELINE, QUERY_STAGE
from src.pipelines.language_model.stages import query_model_run
from src.corpora import registry as corpora_registry
from src.models.core import registry as model_registry
from src.ml_core.tracking import (
    assert_clearml_endpoints_reachable,
    clearml_options,
    clearml_settings,
    configure_clearml_config_file,
    start_clearml_run,
)


MODEL_TRAINING_CONFIG_SECTION = "model-training"


def load_query_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = stage_resume.load_stage_command_defaults("query")
    pipeline_defaults = load_defaults_from_sections((MODEL_TRAINING_CONFIG_SECTION,))
    for key in ("pipeline_name", "pipeline_version"):
        if key in pipeline_defaults:
            defaults[key] = pipeline_defaults[key]
    return defaults


@configured_command(
    "query",
    default_loader=load_query_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run a repeatable ClearML query task against a trained language model.",
)
@click.option(
    "--pipeline-controller-id",
    default=None,
    help=(
        "Existing model-training PipelineController task ID to query. "
        "If omitted, the newest matching completed train_model stage is selected."
    ),
)
@click.option(
    "--pipeline-version",
    default=None,
    help=(
        "Model-training pipeline version to search when --model-task-id is omitted. "
        "Omit to search all versions."
    ),
)
@click.option(
    "--pipeline-name",
    default=MODEL_TRAINING_PIPELINE.default_name,
    show_default=True,
    help="Model-training PipelineController name to search when --model-task-id is omitted.",
)
@click.option(
    "--model",
    "model_name",
    type=click.Choice(model_registry.model_names()),
    default=model_registry.default_model_name(),
    show_default=True,
    help="Registered model to query.",
)
@click.option(
    "--tokenizer-model-name",
    default=None,
    help="Registered tokenizer model name used by model training.",
)
@click.option(
    "--corpus",
    type=click.Choice(corpora_registry.corpus_names()),
    default=corpora_registry.default_corpus_name(),
    show_default=True,
    help="Registered corpus used by model training.",
)
@click.option(
    "--model-task-id",
    default=None,
    help="Completed train_model ClearML task ID to query. Overrides pipeline lookup.",
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Local trained model JSON to query. Overrides pipeline lookup.",
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
    pipeline_version: str | None,
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
    model_definition = model_registry.get_model(model_name)
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

    settings = clearml_settings(
        project_name=clearml_project,
        task_name=clearml_task_name,
        config_file=clearml_config_file,
        connectivity_check=clearml_connectivity_check,
        output_uri=clearml_output_uri,
        tags=clearml_tags,
    )
    resolved_config_file = configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        assert_clearml_endpoints_reachable(resolved_config_file, settings.output_uri)

    (
        resolved_model_task_id,
        resolved_model_path,
        source_controller_id,
    ) = _resolve_query_model_source(
        model_task_id=model_task_id,
        model_path=model_path,
        pipeline_controller_id=pipeline_controller_id,
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        model_name=model_definition.name,
        tokenizer_model_name=tokenizer_model_name,
        corpus=corpus,
    )

    emit_stage_title(1, 1, "Query")
    with start_clearml_run(
        clearml_settings(
            project_name=settings.project_name,
            task_name=settings.task_name,
            config_file=resolved_config_file,
            connectivity_check=False,
            output_uri=settings.output_uri,
            tags=settings.tags,
        ),
        default_task_name=f"query {model_definition.name} {corpus}",
        task_type="inference",
    ) as clearml_run:
        result = query_model_run(
            clearml_run,
            model_task_id=resolved_model_task_id,
            model_path=resolved_model_path,
            source_pipeline_controller_id=source_controller_id,
            model_name=model_definition.name,
            corpus=corpus,
            prompt=prompt,
            max_tokens=max_tokens,
            top_k=top_k,
            decoding=decoding,
            temperature=temperature,
            seed=seed,
            command="src.cli.query",
            stage=QUERY_STAGE,
        )
        for line in model_definition.query_lines(result):
            click.echo(line)

        if clearml_run.task_id is not None:
            click.echo(f"ClearML query task ID: {clearml_run.task_id}")
        if clearml_run.task_url:
            click.echo(f"ClearML query URL: {clearml_run.task_url}")

    if source_controller_id is not None:
        click.echo(f"Source pipeline controller task ID: {source_controller_id}")
    if resolved_model_task_id is not None:
        click.echo(f"Source model stage task ID: {resolved_model_task_id}")


def _resolve_query_model_source(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    pipeline_controller_id: str | None,
    pipeline_name: str,
    pipeline_version: str | None,
    clearml_project: str,
    model_name: str,
    tokenizer_model_name: str | None,
    corpus: str,
) -> tuple[str | None, Path | None, str | None]:
    if model_task_id is not None and model_path is not None:
        raise click.ClickException("Pass either --model-task-id or --model-path, not both.")
    if model_task_id is not None or model_path is not None:
        return model_task_id, model_path, None

    resolved_tokenizer_model_name = stage_resume.require_tokenizer_model_name(
        tokenizer_model_name,
        action="Query",
    )
    resolution = resolve_model_training_task(
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=clearml_project,
        model_name=model_name,
        tokenizer_model_name=resolved_tokenizer_model_name,
        corpus=corpus,
        pipeline_controller_id=pipeline_controller_id,
    )
    click.echo(f"Resolved model pipeline controller task ID: {resolution.controller_id}")
    click.echo(f"Resolved model stage task ID: {resolution.model_task_id}")
    return resolution.model_task_id, None, resolution.controller_id


if __name__ == "__main__":
    main()
