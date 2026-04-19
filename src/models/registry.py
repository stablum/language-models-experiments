"""Registry for trainable language models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from src.models.bigram import BigramTrainingSummary, train_bigram_model


ModelOptions = Mapping[str, Any]
ModelTrainer = Callable[[Iterable[str], ModelOptions], Any]
ModelOptionValidator = Callable[[ModelOptions], None]
SummaryFormatter = Callable[[Any], list[tuple[str, str]]]


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    train: ModelTrainer
    validate_options: ModelOptionValidator
    summary_items: SummaryFormatter


DEFAULT_MODEL_NAME = "bigram"


def default_bigram_tokenizer_model(corpus: str) -> Path:
    return Path("artifacts", "tokenizers", f"{corpus}-sentencepiece-1000.model")


def default_bigram_output(corpus: str) -> Path:
    return Path("artifacts", "models", f"{corpus}-sentencepiece-bigram.json")


def resolve_bigram_tokenizer_model(options: ModelOptions) -> Path:
    return options["tokenizer_model"] or default_bigram_tokenizer_model(options["corpus"])


def resolve_bigram_output(options: ModelOptions) -> Path:
    return options["output"] or default_bigram_output(options["corpus"])


def validate_bigram_options(options: ModelOptions) -> None:
    tokenizer_model = resolve_bigram_tokenizer_model(options)
    if not tokenizer_model.exists():
        raise click.ClickException(
            f"Tokenizer model not found: {tokenizer_model}. "
            "Train it first with src.cli.train_sentencepiece."
        )


def train_bigram_from_options(
    texts: Iterable[str],
    options: ModelOptions,
) -> BigramTrainingSummary:
    return train_bigram_model(
        texts,
        tokenizer_model=resolve_bigram_tokenizer_model(options),
        output_path=resolve_bigram_output(options),
        smoothing=options["smoothing"],
    )


def format_bigram_summary(summary: BigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        ("Tokenizer", str(summary.tokenizer_model)),
        ("Bigram model", str(summary.output_path)),
        ("Vocabulary size", f"{summary.vocab_size:,}"),
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
        ("Transitions", f"{summary.transition_count:,}"),
    ]


MODELS = {
    DEFAULT_MODEL_NAME: ModelDefinition(
        name=DEFAULT_MODEL_NAME,
        train=train_bigram_from_options,
        validate_options=validate_bigram_options,
        summary_items=format_bigram_summary,
    )
}


def model_names() -> tuple[str, ...]:
    return tuple(MODELS)


def get_model(name: str) -> ModelDefinition:
    return MODELS[name]
