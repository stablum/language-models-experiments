"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.cli.config import configured_command, load_defaults_from_sections
from src.cli.output import stage_title
from src.cli.pipeline_common import (
    DEFAULT_PIPELINE_NAME,
    QUERY_STAGE,
    pipeline_options,
    pipeline_resume_option,
    resume_pipeline_controller_stage,
)
from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import (
    clearml_options,
    clearml_settings,
    download_task_artifact,
    start_clearml_run,
)


def load_query_command_defaults(_config_section: str) -> dict[str, object]:
    defaults = load_defaults_from_sections(("defaults", "clearml"))
    train_defaults = load_defaults_from_sections(("train",))
    if "model_name" in train_defaults:
        defaults["model_name"] = train_defaults["model_name"]
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
    "--corpus",
    type=click.Choice(corpus_names()),
    default=DEFAULT_CORPUS_NAME,
    show_default=True,
    help="Registered corpus used to resolve the default model path.",
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
            "corpus": corpus,
        },
    )
    return

    click.echo(stage_title(1, 1, "Query"), color=True)
    click.echo(f"Model: {model_definition.name}")
    click.echo(f"Corpus: {corpus}")
    task_id: str | None = None
    task_url: str | None = None
    with (
        TemporaryDirectory(prefix="lme-query-") as staging_root,
        start_clearml_run(
            clearml_settings(
                project_name=clearml_project,
                task_name=clearml_task_name,
                config_file=clearml_config_file,
                connectivity_check=clearml_connectivity_check,
                output_uri=clearml_output_uri,
                tags=clearml_tags,
            ),
            default_task_name=f"query {model_definition.name} {corpus}",
            task_type="inference",
        ) as clearml_run,
    ):
        staged_model_path = stage_model_artifacts(
            model_task_id=model_task_id,
            model_path=model_path,
            staging_dir=Path(staging_root),
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
            model_definition.validate_query_options(query_options)

        task_id = clearml_run.task_id
        task_url = clearml_run.task_url
        clearml_run.connect_parameter_sections(
            {
                "Run": {
                    "command": "src.cli.query",
                    "artifact_store": "clearml",
                },
                "Data": {
                    "corpus": corpus,
                },
                "Model": {
                    "model": model_definition.name,
                    "model_task_id": model_task_id,
                },
                "Query": {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "top_k": top_k,
                    "decoding": decoding,
                    "temperature": temperature,
                    "seed": seed,
                },
                "Artifacts": {
                    "model_artifact": "trained-model-json",
                    "tokenizer_artifact": "input-tokenizer-model",
                    "model_artifact_file": staged_model_path.name,
                },
            }
        )

        result = model_definition.query(query_options)

        clearml_run.log_metrics("Query", query_metrics(result))
        clearml_run.upload_artifact(
            "query-result",
            query_payload(result),
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "queried-model",
            result.model_path,
            metadata={"model": model_definition.name, "corpus": corpus},
        )
        clearml_run.upload_artifact(
            "tokenizer-model",
            result.tokenizer_model,
            metadata={"model": model_definition.name, "corpus": corpus},
        )

    for line in model_definition.query_lines(result):
        click.echo(line)
    click.echo(f"ClearML task ID: {task_id}")
    if task_url is not None:
        click.echo(f"ClearML task URL: {task_url}")
    click.echo("Query artifact: query-result")


def stage_model_artifacts(
    *,
    model_task_id: str | None,
    model_path: Path | None,
    staging_dir: Path,
) -> Path:
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

    if model_task_id is not None:
        staged_model_path = download_task_artifact(
            task_id=model_task_id,
            artifact_name="trained-model-json",
            destination_dir=staging_dir,
        )
        download_task_artifact(
            task_id=model_task_id,
            artifact_name="input-tokenizer-model",
            destination_dir=staging_dir,
            filename=stored_tokenizer_filename(staged_model_path),
        )
        return staged_model_path

    return model_path


def validate_model_source(
    *,
    model_task_id: str | None,
    model_path: Path | None,
) -> None:
    if model_task_id is not None and model_path is not None:
        raise click.ClickException("Pass either --model-task-id or --model-path, not both.")

    if model_task_id is None and model_path is None:
        raise click.ClickException(
            "Querying now uses ClearML as the artifact store. Pass --model-task-id "
            "from src.cli.train, or pass --model-path to query a local model file."
        )


def query_metrics(result: object) -> dict[str, object]:
    next_predictions = getattr(result, "next_token_predictions", [])
    top_probability = next_predictions[0].probability if next_predictions else None
    return {
        "prompt_token_count": len(getattr(result, "prompt_token_ids", [])),
        "generated_token_count": len(getattr(result, "generated_token_ids", [])),
        "total_token_count": len(getattr(result, "token_ids", [])),
        "next_token_candidate_count": len(next_predictions),
        "top_next_token_probability": top_probability,
    }


def query_payload(result: object) -> dict[str, object]:
    return {
        "model_artifact_file": artifact_file(getattr(result, "model_path", None)),
        "tokenizer_artifact_file": artifact_file(getattr(result, "tokenizer_model", None)),
        "decoding": getattr(result, "decoding", None),
        "text_normalization": getattr(result, "text_normalization", None),
        "prompt": getattr(result, "prompt", None),
        "prompt_token_ids": getattr(result, "prompt_token_ids", None),
        "generated_token_ids": getattr(result, "generated_token_ids", None),
        "token_ids": getattr(result, "token_ids", None),
        "continuation_text": getattr(result, "continuation_text", None),
        "generated_text": getattr(result, "generated_text", None),
        "next_token_predictions": [
            {
                "token_id": prediction.token_id,
                "piece": prediction.piece,
                "count": prediction.count,
                "probability": prediction.probability,
            }
            for prediction in getattr(result, "next_token_predictions", [])
        ],
    }


def artifact_file(path: object) -> str | None:
    if path is None:
        return None
    return Path(path).name


def stored_tokenizer_filename(model_path: Path) -> str | None:
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping):
        return None

    tokenizer_model = payload.get("tokenizer_model")
    if tokenizer_model is None:
        return None
    return Path(str(tokenizer_model)).name


if __name__ == "__main__":
    main()
