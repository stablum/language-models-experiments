"""Registry for trainable language models."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from src.models.bigram import (
    BigramPrediction,
    BigramQueryResult,
    BigramTrainingSummary,
    load_bigram_model,
    train_bigram_model,
)


ModelOptions = Mapping[str, Any]
ModelTrainer = Callable[[Iterable[str], ModelOptions], Any]
ModelQuery = Callable[[ModelOptions], Any]
ModelOptionValidator = Callable[[ModelOptions], None]
SummaryFormatter = Callable[[Any], list[tuple[str, str]]]
QueryFormatter = Callable[[Any], list[str]]


@dataclass(frozen=True)
class ModelDefinition:
    name: str
    train: ModelTrainer
    validate_options: ModelOptionValidator
    summary_items: SummaryFormatter
    query: ModelQuery | None = None
    validate_query_options: ModelOptionValidator | None = None
    query_lines: QueryFormatter | None = None


DEFAULT_MODEL_NAME = "bigram"


def default_bigram_tokenizer_model(corpus: str) -> Path:
    return Path("artifacts", "tokenizers", f"{corpus}-sentencepiece-1000.model")


def default_bigram_output(corpus: str) -> Path:
    return Path("artifacts", "models", f"{corpus}-sentencepiece-bigram.json")


def default_bigram_model(corpus: str) -> Path:
    return default_bigram_output(corpus)


def resolve_bigram_tokenizer_model(options: ModelOptions) -> Path:
    return options["tokenizer_model"] or default_bigram_tokenizer_model(options["corpus"])


def resolve_bigram_output(options: ModelOptions) -> Path:
    return options["output"] or default_bigram_output(options["corpus"])


def resolve_bigram_model(options: ModelOptions) -> Path:
    return options["model_path"] or default_bigram_model(options["corpus"])


def validate_bigram_options(options: ModelOptions) -> None:
    tokenizer_model = resolve_bigram_tokenizer_model(options)
    if not tokenizer_model.exists():
        raise click.ClickException(
            f"Tokenizer model not found: {tokenizer_model}. "
            "Train it first with src.cli.train_sentencepiece."
        )


def validate_bigram_query_options(options: ModelOptions) -> None:
    model_path = resolve_bigram_model(options)
    if not model_path.exists():
        raise click.ClickException(
            f"Bigram model not found: {model_path}. "
            "Train it first with src.cli.train."
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


def query_bigram_from_options(options: ModelOptions) -> BigramQueryResult:
    model = load_bigram_model(resolve_bigram_model(options))
    return model.query(
        prompt=options["prompt"],
        max_tokens=options["max_tokens"],
        top_k=options["top_k"],
        decoding=options["decoding"],
        temperature=options["temperature"],
        seed=options["seed"],
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


def format_bigram_query(result: BigramQueryResult) -> list[str]:
    continuation_label = (
        "Most probable continuation:"
        if result.decoding == "most-probable"
        else "Sampled continuation:"
    )
    lines = [
        f"Model file: {result.model_path}",
        f"Tokenizer: {result.tokenizer_model}",
        f"Decoding: {result.decoding}",
        f"Prompt tokens: {len(result.prompt_token_ids):,}",
        f"Generated tokens: {len(result.generated_token_ids):,}",
        continuation_label,
        format_console_text(result.continuation_text) if result.continuation_text else "(empty)",
        "Full text:",
        format_console_text(result.generated_text) if result.generated_text else "(empty)",
    ]

    if not result.next_token_predictions:
        return [*lines, "", "Top next tokens: none"]

    lines.extend(["", "Top next tokens:"])
    lines.extend(
        f"  {prediction.token_id:>4} {format_prediction_piece(prediction, result)} "
        f"count={prediction.count:,} p={prediction.probability:.4%}"
        for prediction in result.next_token_predictions
    )
    return lines


def format_prediction_piece(
    prediction: BigramPrediction,
    result: BigramQueryResult,
) -> str:
    special_label = special_token_label(prediction.token_id, result)
    if special_label is not None:
        return special_label
    return ascii(prediction.piece)


def special_token_label(token_id: int, result: BigramQueryResult) -> str | None:
    special_tokens = {
        result.bos_id: "[BOS]",
        result.eos_id: "[EOS]",
        result.unk_id: "[UNK]",
    }
    return special_tokens.get(token_id) if token_id >= 0 else None


def format_console_text(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


MODELS = {
    DEFAULT_MODEL_NAME: ModelDefinition(
        name=DEFAULT_MODEL_NAME,
        train=train_bigram_from_options,
        validate_options=validate_bigram_options,
        summary_items=format_bigram_summary,
        query=query_bigram_from_options,
        validate_query_options=validate_bigram_query_options,
        query_lines=format_bigram_query,
    )
}


def model_names() -> tuple[str, ...]:
    return tuple(MODELS)


def get_model(name: str) -> ModelDefinition:
    return MODELS[name]
