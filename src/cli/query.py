"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import stage_resume
from src.ml_core import pipeline as core_pipeline
from src.ml_core.cli.config import configured_command, load_defaults_from_sections
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import query as query_pipeline
from src.corpora import registry as corpora_registry
from src.models.core import registry as model_registry
from src.ml_core.tracking import (
    assert_clearml_endpoints_reachable,
    clearml_options,
    clearml_settings,
    configure_clearml_config_file,
)


MODEL_TRAINING_CONFIG_SECTION = "model-training"
QUERY_PIPELINE_CONFIG_SECTION = "query-pipeline"


def load_query_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = stage_resume.load_stage_command_defaults("query")
    defaults.update(load_defaults_from_sections((QUERY_PIPELINE_CONFIG_SECTION,)))

    model_training_defaults = load_defaults_from_sections((MODEL_TRAINING_CONFIG_SECTION,))
    if "pipeline_name" in model_training_defaults:
        defaults["model_training_name"] = model_training_defaults["pipeline_name"]
    if "pipeline_version" in model_training_defaults:
        defaults["model_training_version"] = model_training_defaults["pipeline_version"]
    return defaults


@configured_command(
    "query",
    default_loader=load_query_command_defaults,
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Run a repeatable single-stage ClearML query pipeline.",
)
@core_pipeline.pipeline_options(
    default_name=query_pipeline.QUERY_PIPELINE.default_name
)
@click.option(
    "--model-training-controller-id",
    "--source-pipeline-controller-id",
    "model_training_controller_id",
    default=None,
    help=(
        "Existing model-training PipelineController task ID to query. "
        "If omitted, the newest matching completed train_model stage is selected."
    ),
)
@click.option(
    "--model-training-version",
    default=None,
    help=(
        "Model-training pipeline version to search when --model-task-id is omitted. "
        "Omit to search all versions."
    ),
)
@click.option(
    "--model-training-name",
    default=model_pipeline.MODEL_TRAINING_PIPELINE.default_name,
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
    pipeline_version: str,
    pipeline_local: bool,
    controller_queue: str,
    execution_queue: str | None,
    wait: bool,
    add_run_number: bool,
    model_training_controller_id: str | None,
    model_training_version: str | None,
    model_training_name: str,
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
    if pipeline_local and not wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")
    if model_path is not None and not pipeline_local:
        raise click.ClickException(
            "--model-path is only supported with --pipeline-local. "
            "Use --model-task-id for queued query pipelines."
        )

    model_definition = model_registry.get_model(model_name)
    if model_definition.query is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

    resolved_pipeline_name = clearml_task_name or pipeline_name
    settings = clearml_settings(
        project_name=clearml_project,
        task_name=resolved_pipeline_name,
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
        model_training_controller_id=model_training_controller_id,
        model_training_name=model_training_name,
        model_training_version=model_training_version,
        clearml_project=settings.project_name,
        model_name=model_definition.name,
        tokenizer_model_name=tokenizer_model_name,
        corpus=corpus,
    )

    pipeline = core_pipeline.build_pipeline_controller(
        pipeline_name=resolved_pipeline_name,
        pipeline_version=pipeline_version,
        clearml_project=settings.project_name,
        clearml_tags=settings.tags,
        clearml_output_uri=settings.output_uri,
        add_run_number=add_run_number,
    )
    lm_def.configure_pipeline_control(
        pipeline.task,
        run_stage=None,
        run_until_stage=None,
        updated_by="query-pipeline-cli",
    )
    core_pipeline.connect_controller_experiment_parameters(
        pipeline.task,
        {
            "model": model_definition.name,
            "corpus": corpus,
            "tokenizer_model_name": tokenizer_model_name or "",
            "model_training_name": model_training_name,
            "model_training_version": model_training_version or "",
            "source_pipeline_controller_id": source_controller_id or "",
            "model_task_id": resolved_model_task_id or "",
            "model_path": resolved_model_path or "",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "decoding": decoding,
            "temperature": temperature,
            "seed": seed,
        },
    )
    query_pipeline.add_pipeline_steps(
        pipeline,
        clearml_project=settings.project_name,
        clearml_output_uri=settings.output_uri,
        clearml_tags=settings.tags,
        clearml_config_file=resolved_config_file if pipeline_local else None,
        execution_queue=None if pipeline_local else execution_queue,
        source_pipeline_controller_id=source_controller_id,
        model_task_id=resolved_model_task_id,
        model_path=resolved_model_path,
        model_name=model_definition.name,
        corpus=corpus,
        prompt=prompt,
        max_tokens=max_tokens,
        top_k=top_k,
        decoding=decoding,
        temperature=temperature,
        seed=seed,
    )

    click.echo(f"ClearML query pipeline: {settings.project_name}/{resolved_pipeline_name}")
    click.echo(f"Pipeline version: {pipeline_version}")
    if source_controller_id is not None:
        click.echo(f"Source pipeline controller task ID: {source_controller_id}")
    if resolved_model_task_id is not None:
        click.echo(f"Source model stage task ID: {resolved_model_task_id}")
    if resolved_model_path is not None:
        click.echo(f"Source model path: {resolved_model_path}")
    click.echo(f"Pipeline controller task ID: {pipeline.task.id}")
    task_url = pipeline.task.get_output_log_web_page()
    if task_url:
        click.echo(f"Pipeline controller URL: {task_url}")
    click.echo(f"Stage tasks: {lm_def.QUERY_STAGE}")

    if pipeline_local:
        click.echo("Execution mode: local ClearML PipelineController")
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        click.echo(f"Execution mode: queued controller on {controller_queue}")
        if execution_queue is not None:
            click.echo(f"Step execution queue: {execution_queue}")
        pipeline.start(queue=controller_queue, wait=wait)

    click.echo("ClearML query pipeline submitted.")
    if wait:
        core_pipeline.assert_pipeline_finished_successfully(pipeline)
        core_pipeline.print_stage_task_ids(
            pipeline.task.id,
            query_pipeline.QUERY_PIPELINE.stages,
            stage_names=query_pipeline.QUERY_PIPELINE.stages,
        )
        click.echo("ClearML query pipeline run completed.")


def _resolve_query_model_source(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    model_training_controller_id: str | None,
    model_training_name: str,
    model_training_version: str | None,
    clearml_project: str,
    model_name: str,
    tokenizer_model_name: str | None,
    corpus: str,
) -> tuple[str | None, Path | None, str | None]:
    if model_task_id is not None and model_path is not None:
        raise click.ClickException("Pass either --model-task-id or --model-path, not both.")
    if model_task_id is not None or model_path is not None:
        return model_task_id, model_path, model_training_controller_id

    resolved_tokenizer_model_name = stage_resume.require_tokenizer_model_name(
        tokenizer_model_name,
        action="Query",
    )
    resolution = lm_def.resolve_model_training_task(
        pipeline_name=model_training_name,
        pipeline_version=model_training_version,
        clearml_project=clearml_project,
        model_name=model_name,
        tokenizer_model_name=resolved_tokenizer_model_name,
        corpus=corpus,
        pipeline_controller_id=model_training_controller_id,
    )
    click.echo(f"Resolved model pipeline controller task ID: {resolution.controller_id}")
    click.echo(f"Resolved model stage task ID: {resolution.model_task_id}")
    return resolution.model_task_id, None, resolution.controller_id


if __name__ == "__main__":
    main()
