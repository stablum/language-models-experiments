"""ClearML query stage functions for language-model experiments."""

from __future__ import annotations

from pathlib import Path

import click

from src.ml_core import tracking
from src.ml_core.cli import staging
from src.ml_core.models import definition as model_def
from src.models.core import registry as model_registry
from src.pipelines.language_model import artifacts as lm_artifacts
from src.pipelines.language_model import definition as lm_def
from src.pipelines.language_model import stage_runtime


def query_pipeline_step(
    *,
    model_task_id: str | None,
    model_path: str | Path | None = None,
    source_pipeline_controller_id: str | None = None,
    model_name: str,
    corpus: str,
    prompt: str,
    max_tokens: int,
    top_k: int,
    decoding: str,
    temperature: float,
    seed: int | None,
    tokenizer_model_name: str | None = None,
    command: str = "src.cli.model_training",
    clearml_output_uri: str | None = None,
    clearml_tags: stage_runtime.ClearmlTags = None,
    clearml_config_file: str | None = None,
    pipeline_stage_index: int | None = None,
    pipeline_stage_total: int | None = None,
    pipeline_stage_title: str | None = None,
) -> str:
    """Query the trained model step artifact."""
    stage = lm_def.QUERY_STAGE
    clearml_run = stage_runtime.start_step(
        stage_runtime.StepRuntimeCfg(
            stage=stage,
            clearml_output_uri=clearml_output_uri,
            clearml_tags=clearml_tags,
            clearml_config_file=clearml_config_file,
            pipeline_stage_index=pipeline_stage_index,
            pipeline_stage_total=pipeline_stage_total,
            pipeline_stage_title=pipeline_stage_title,
        )
    )
    result = query_model_run(
        clearml_run,
        model_task_id=model_task_id,
        model_path=Path(model_path) if model_path is not None else None,
        source_pipeline_controller_id=source_pipeline_controller_id,
        model_name=model_name,
        corpus=corpus,
        prompt=prompt,
        max_tokens=max_tokens,
        top_k=top_k,
        decoding=decoding,
        temperature=temperature,
        seed=seed,
        tokenizer_model_name=tokenizer_model_name,
        command=command,
        stage=stage,
    )
    model_definition = model_registry.get_model(model_name)
    for line in _query_lines(model_definition, result):
        click.echo(line)
    return stage_runtime.require_task_id(clearml_run)


def query_model_run(
    clearml_run: tracking.ClearMLRun,
    *,
    model_task_id: str | None,
    model_path: Path | None,
    source_pipeline_controller_id: str | None,
    model_name: str,
    corpus: str,
    prompt: str,
    max_tokens: int,
    top_k: int,
    decoding: str,
    temperature: float,
    seed: int | None,
    command: str,
    tokenizer_model_name: str | None = None,
    stage: str = lm_def.QUERY_STAGE,
) -> object:
    """Query a trained model and store the standard query artifacts."""
    model_definition = model_registry.get_model(model_name)
    if model_definition.query is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

    with staging.temporary_staging_directory(prefix="lme-query-") as staging_dir:
        staged_model_path = lm_artifacts.stage_model_files(
            model_task_id=model_task_id,
            model_path=model_path,
            staging_dir=staging_dir,
            clearml_run=clearml_run,
            output_model_name=lm_def.model_output_name(
                tokenizer_model_name=tokenizer_model_name,
                model_name=model_definition.name,
            ),
        )
        query_options = {
            "corpus": corpus,
            "model_path": staged_model_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "top_k": top_k,
            "decoding": decoding,
            "temperature": temperature,
            "seed": seed,
        }
        if model_definition.validate_query_options is not None:
            try:
                model_definition.validate_query_options(query_options)
            except model_def.ModelOptionError as error:
                raise click.ClickException(str(error)) from error

        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": command,
                    "artifact_store": "clearml",
                },
                "Pipeline": {
                    "stage": stage,
                    "source_pipeline_controller_id": source_pipeline_controller_id,
                    "model_task_id": model_task_id,
                },
                "Data": {
                    "corpus": corpus,
                },
                "Model": {
                    "model": model_definition.name,
                    "tokenizer_model_name": tokenizer_model_name,
                    "model_task_id": model_task_id,
                    "model_path": model_path,
                    "model_file": staged_model_path.name,
                },
                "Query": {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "top_k": top_k,
                    "decoding": decoding,
                    "temperature": temperature,
                    "seed": seed,
                },
            }
        )

        result = model_definition.query(query_options)
        clearml_run.log_metrics("Query", lm_artifacts.query_metrics(result))
        clearml_run.report_debug_sample(
            title="Query",
            series="result",
            contents=lm_artifacts.query_debug_sample(result),
            file_extension="txt",
        )
        clearml_run.upload_artifact(
            "query-result",
            lm_artifacts.query_payload(result),
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        return result


def _query_lines(model_definition: object, result: object) -> tuple[str, ...]:
    query_lines = getattr(model_definition, "query_lines", None)
    return tuple(query_lines(result)) if query_lines is not None else ()
