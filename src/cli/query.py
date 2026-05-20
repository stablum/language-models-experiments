"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.cli import options as cli_options
from src.cli import stage_resume
from src.ml_core import cfg as core_cfg
from src.ml_core import pipeline as core_pipeline
from src.ml_core import pipeline_tasks
from src.ml_core import tracking
from src.ml_core.cli import config as cli_config
from src.models.core import registry as model_registry
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import model_training as model_pipeline
from src.pipelines.language_model import query as query_pipeline


MODEL_TRAINING_CONFIG_SECTION = "model-training"
QUERY_PIPELINE_CONFIG_SECTION = "query-pipeline"


class QueryArgs(core_cfg.BaseCfg):
    """Raw Click arguments for the query-pipeline command."""

    pipeline_name: str
    pipeline_version: str
    pipeline_local: bool
    controller_queue: str
    execution_queue: str | None
    wait: bool
    add_run_number: bool
    model_training_controller_id: str | None
    model_training_version: str | None
    model_training_name: str
    model_name: str
    tokenizer_model_name: str | None
    corpus: str
    model_task_id: str | None
    model_path: Path | None
    prompt: str
    max_tokens: int
    top_k: int
    decoding: str
    temperature: float
    seed: int | None
    clearml_project: str
    clearml_task_name: str | None
    clearml_config_file: Path | None
    clearml_connectivity_check: bool
    clearml_output_uri: str | None
    clearml_tags: tuple[str, ...]


def load_query_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = stage_resume.load_stage_command_defaults("query")
    defaults.update(
        cli_config.load_defaults_from_sections((QUERY_PIPELINE_CONFIG_SECTION,))
    )

    model_training_defaults = cli_config.load_defaults_from_sections(
        (MODEL_TRAINING_CONFIG_SECTION,)
    )
    if "pipeline_name" in model_training_defaults:
        defaults["model_training_name"] = model_training_defaults["pipeline_name"]
    if "pipeline_version" in model_training_defaults:
        defaults["model_training_version"] = model_training_defaults["pipeline_version"]
    return defaults


@cli_config.configured_command(
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
@cli_options.model_option("Registered model to query.")
@cli_options.tokenizer_model_name_option
@cli_options.corpus_option
@click.option(
    "--model-task-id",
    default=None,
    help="Completed train_model ClearML task ID to query. Overrides pipeline lookup.",
)
@cli_options.model_path_option
@cli_options.query_generation_options
@tracking.clearml_options
def main(**kwargs: object) -> None:
    args = QueryArgs(**kwargs)
    if args.pipeline_local and not args.wait:
        raise click.ClickException("--no-wait is only supported with --pipeline-queued.")
    if args.model_path is not None and not args.pipeline_local:
        raise click.ClickException(
            "--model-path is only supported with --pipeline-local. "
            "Use --model-task-id for queued query pipelines."
        )

    model_definition = model_registry.get_model(args.model_name)
    if model_definition.query is None:
        raise click.ClickException(
            f"Model does not support querying yet: {args.model_name}"
        )

    resolved_pipeline_name = args.clearml_task_name or args.pipeline_name
    settings = tracking.clearml_settings(
        project_name=args.clearml_project,
        task_name=resolved_pipeline_name,
        config_file=args.clearml_config_file,
        connectivity_check=args.clearml_connectivity_check,
        output_uri=args.clearml_output_uri,
        tags=args.clearml_tags,
    )
    resolved_config_file = tracking.configure_clearml_config_file(settings.config_file)
    if settings.connectivity_check:
        tracking.assert_clearml_endpoints_reachable(
            resolved_config_file,
            settings.output_uri,
        )

    (
        resolved_model_task_id,
        resolved_model_path,
        source_controller_id,
        resolved_tokenizer_model_name,
    ) = _resolve_query_model_source(
        model_task_id=args.model_task_id,
        model_path=args.model_path,
        model_training_controller_id=args.model_training_controller_id,
        model_training_name=args.model_training_name,
        model_training_version=args.model_training_version,
        clearml_project=settings.project_name,
        model_name=model_definition.name,
        tokenizer_model_name=args.tokenizer_model_name,
        corpus=args.corpus,
    )

    pipeline = core_pipeline.build_pipeline_controller(
        pipeline_name=resolved_pipeline_name,
        pipeline_version=args.pipeline_version,
        clearml_project=settings.project_name,
        clearml_tags=settings.tags,
        clearml_output_uri=settings.output_uri,
        add_run_number=args.add_run_number,
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
            "corpus": args.corpus,
            "tokenizer_model_name": resolved_tokenizer_model_name or "",
            "model_training_name": args.model_training_name,
            "model_training_version": args.model_training_version or "",
            "source_pipeline_controller_id": source_controller_id or "",
            "model_task_id": resolved_model_task_id or "",
            "model_path": resolved_model_path or "",
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "top_k": args.top_k,
            "decoding": args.decoding,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    )
    query_pipeline.add_pipeline_steps(
        pipeline,
        execution=query_pipeline.ExecutionCfg(
            project_name=settings.project_name,
            output_uri=settings.output_uri,
            tags=settings.tags,
            config_file=resolved_config_file if args.pipeline_local else None,
            queue=None if args.pipeline_local else args.execution_queue,
        ),
        source=query_pipeline.ModelSourceCfg(
            source_pipeline_controller_id=source_controller_id,
            model_task_id=resolved_model_task_id,
            model_path=resolved_model_path,
            model_name=model_definition.name,
            tokenizer_model_name=resolved_tokenizer_model_name,
            corpus=args.corpus,
        ),
        query=query_pipeline.QueryCfg(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            top_k=args.top_k,
            decoding=args.decoding,
            temperature=args.temperature,
            seed=args.seed,
        ),
    )

    click.echo(f"ClearML query pipeline: {settings.project_name}/{resolved_pipeline_name}")
    click.echo(f"Pipeline version: {args.pipeline_version}")
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

    if args.pipeline_local:
        click.echo("Execution mode: local ClearML PipelineController")
        pipeline.start_locally(run_pipeline_steps_locally=True)
    else:
        click.echo(f"Execution mode: queued controller on {args.controller_queue}")
        if args.execution_queue is not None:
            click.echo(f"Step execution queue: {args.execution_queue}")
        pipeline.start(queue=args.controller_queue, wait=args.wait)

    click.echo("ClearML query pipeline submitted.")
    if args.wait:
        pipeline_tasks.assert_pipeline_finished_successfully(pipeline)
        pipeline_tasks.print_stage_task_ids(
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
) -> tuple[str | None, Path | None, str | None, str | None]:
    if model_task_id is not None and model_path is not None:
        raise click.ClickException("Pass either --model-task-id or --model-path, not both.")
    if model_task_id is not None or model_path is not None:
        return (
            model_task_id,
            model_path,
            model_training_controller_id,
            str(tokenizer_model_name or "").strip() or None,
        )

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
    return resolution.model_task_id, None, resolution.controller_id, resolved_tokenizer_model_name


if __name__ == "__main__":
    main()
