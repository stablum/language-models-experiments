"""Shared display formatting for n-gram model results."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def format_ngram_evaluation_metrics(summary: Any) -> list[tuple[str, str]]:
    return [
        ("Sequences", f"{summary.sequence_count:,}"),
        ("Tokens", f"{summary.token_count:,}"),
        ("Transitions", f"{summary.transition_count:,}"),
        (
            "Next-token accuracy",
            format_rate(summary.correct_next_token_count, summary.transition_count),
        ),
        (
            f"Top-{summary.top_k} accuracy",
            format_rate(summary.top_k_correct_next_token_count, summary.transition_count),
        ),
        (
            "Average NLL",
            format_metric(summary.average_negative_log_likelihood, suffix=" nats/token"),
        ),
        (
            "Cross entropy",
            format_metric(summary.cross_entropy_bits, suffix=" bits/token"),
        ),
        ("Perplexity", format_metric(summary.perplexity)),
        ("Zero-probability transitions", f"{summary.zero_probability_count:,}"),
    ]


def format_ngram_query(result: Any) -> list[str]:
    continuation_label = (
        "Most probable continuation:"
        if result.decoding == "most-probable"
        else "Sampled continuation:"
    )
    lines = [
        f"Model artifact file: {artifact_filename(result.model_path)}",
        f"Tokenizer artifact file: {artifact_filename(result.tokenizer_model)}",
        f"Text normalization: {result.text_normalization}",
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
    prediction: Any,
    result: Any,
) -> str:
    special_label = special_token_label(prediction.token_id, result)
    if special_label is not None:
        return special_label
    return ascii(prediction.piece.replace("\u2581", " "))


def artifact_filename(path: Path) -> str:
    return path.name


def special_token_label(token_id: int, result: Any) -> str | None:
    special_tokens = {
        result.bos_id: "[BOS]",
        result.eos_id: "[EOS]",
        result.unk_id: "[UNK]",
    }
    return special_tokens.get(token_id) if token_id >= 0 else None


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator:,}/{denominator:,} ({numerator / denominator:.2%})"


def format_metric(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return f"inf{suffix}"
    return f"{value:,.4f}{suffix}"


def format_interpolation_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> str:
    return (
        f"unigram={unigram_weight:.3f}, "
        f"bigram={bigram_weight:.3f}, "
        f"trigram={trigram_weight:.3f}"
    )


def format_console_text(text: str) -> str:
    return text.encode("ascii", errors="backslashreplace").decode("ascii")
