"""Generic Click CLI for querying registered language models."""

from __future__ import annotations

from pathlib import Path

import click

from src.corpora.registry import DEFAULT_CORPUS_NAME, corpus_names
from src.models.registry import DEFAULT_MODEL_NAME, get_model, model_names


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
    "--model-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a trained model file.",
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
    "--temperature",
    type=click.FloatRange(min=0.0),
    default=1.0,
    show_default=True,
    help="Sampling temperature. Use 0 for greedy decoding.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible sampling.",
)
def main(
    model_name: str,
    corpus: str,
    model_path: Path | None,
    prompt: str,
    max_tokens: int,
    top_k: int,
    temperature: float,
    seed: int | None,
) -> None:
    model_definition = get_model(model_name)
    if model_definition.query is None or model_definition.query_lines is None:
        raise click.ClickException(f"Model does not support querying yet: {model_name}")

    query_options = {
        "corpus": corpus,
        "model_path": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "top_k": top_k,
        "temperature": temperature,
        "seed": seed,
    }
    if model_definition.validate_query_options is not None:
        model_definition.validate_query_options(query_options)

    click.echo(f"Model: {model_definition.name}")
    click.echo(f"Corpus: {corpus}")
    for line in model_definition.query_lines(model_definition.query(query_options)):
        click.echo(line)


if __name__ == "__main__":
    main()
