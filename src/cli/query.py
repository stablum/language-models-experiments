"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import click

from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names
from src.tracking.clearml import (
    clearml_options,
    clearml_settings,
    download_task_artifact,
    start_clearml_run,
)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Query a registered language model.",
)
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
    help=(
        "ClearML task ID produced by src.cli.train. Downloads its trained-model-json "
        "and input-tokenizer-model artifacts for querying."
    ),
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Local trained model file to query.",
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
    clearml_output_uri: str | None,
    clearml_tags: tuple[str, ...],
) -> None:
    model_definition = get_model(model_name)
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")
    validate_model_source(model_task_id=model_task_id, model_path=model_path)

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
        clearml_run.connect_parameters(
            {
                "command": "src.cli.query",
                "artifact_store": "clearml",
                "model": model_definition.name,
                "model_task_id": model_task_id,
                "model_artifact": "trained-model-json",
                "tokenizer_artifact": "input-tokenizer-model",
                "model_artifact_file": staged_model_path.name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "top_k": top_k,
                "decoding": decoding,
                "temperature": temperature,
                "seed": seed,
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
        download_task_artifact(
            task_id=model_task_id,
            artifact_name="input-tokenizer-model",
            destination_dir=staging_dir,
        )
        return download_task_artifact(
            task_id=model_task_id,
            artifact_name="trained-model-json",
            destination_dir=staging_dir,
        )

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


if __name__ == "__main__":
    main()
