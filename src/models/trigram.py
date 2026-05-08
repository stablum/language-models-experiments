"""Interpolated token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import sentencepiece as spm

from src.corpora import normalization
from src.models import formatting, ngram, trigram_common
from src.ml_core.models.definition import ModelOptionError, ModelOptions


MODEL_NAME = "trigram"
MODEL_SUFFIX = "trigram"


class TrigramTrainingSummary(trigram_common.TrigramTrainingSummary):
    unigram_weight: float = 0.0
    bigram_weight: float = 0.0
    trigram_weight: float = 0.0


class TrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    unigram_weight: float = 0.0
    bigram_weight: float = 0.0
    trigram_weight: float = 0.0


class TrigramModel(trigram_common.BaseTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        TrigramEvaluationSummary
    )
    smoothing: float
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    unigram_counts: dict[int, int]
    unigram_total: int

    def evaluation_summary_fields(self) -> dict[str, object]:
        return {
            "unigram_weight": self.unigram_weight,
            "bigram_weight": self.bigram_weight,
            "trigram_weight": self.trigram_weight,
        }

    def context_probability(
        self,
        next_id: int,
        context: trigram_common.Context,
        counts: trigram_common.ResolvedTrigramContextCounts,
    ) -> float:
        return (
            self.unigram_weight * self.unigram_probability(next_id)
            + self.bigram_weight * self.conditional_probability(
                next_id,
                counts=counts.bigram_counts,
                total=counts.bigram_total,
            )
            + self.trigram_weight * self.conditional_probability(
                next_id,
                counts=counts.trigram_counts,
                total=counts.trigram_total,
            )
        )

    def unigram_probability(self, token_id: int) -> float:
        return ngram.additive_smoothed_probability(
            token_id,
            counts=self.unigram_counts,
            total=self.unigram_total,
            smoothing=self.smoothing,
            candidate_count=ngram.candidate_token_count(self.vocab_size, self.bos_id),
        )

    def conditional_probability(
        self,
        token_id: int,
        *,
        counts: dict[int, int],
        total: int,
    ) -> float:
        return ngram.additive_smoothed_probability(
            token_id,
            counts=counts,
            total=total,
            smoothing=self.smoothing,
            candidate_count=ngram.candidate_token_count(self.vocab_size, self.bos_id),
        )


def normalize_interpolation_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float, float]:
    total = unigram_weight + bigram_weight + trigram_weight
    if total <= 0:
        raise ValueError("At least one interpolation weight must be positive.")
    return unigram_weight / total, bigram_weight / total, trigram_weight / total


def load_trigram_model(model_path: Path) -> TrigramModel:
    data, tokenizer_model, processor, vocab_size = trigram_common.load_standard_trigram_payload(
        model_path,
        model_type="interpolated_trigram",
        label="an interpolated trigram model",
    )
    weights = data["interpolation_weights"]

    return TrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        **ngram.sentencepiece_model_fields(data, processor, vocab_size),
        smoothing=float(data["smoothing"]),
        unigram_weight=float(weights["unigram"]),
        bigram_weight=float(weights["bigram"]),
        trigram_weight=float(weights["trigram"]),
        unigram_counts=trigram_common.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigram_common.parse_bigram_transitions(data),
        trigram_transitions=trigram_common.parse_trigram_transitions(data),
    )


def train_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    smoothing: float = 0.1,
    unigram_weight: float = 0.1,
    bigram_weight: float = 0.3,
    trigram_weight: float = 0.6,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> TrigramTrainingSummary:
    normalized_weights = normalize_interpolation_weights(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
    )
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    summary = TrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        unigram_weight=normalized_weights[0],
        bigram_weight=normalized_weights[1],
        trigram_weight=normalized_weights[2],
        text_normalization=text_normalization,
    )
    counts = trigram_common.collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    trigram_common.apply_trigram_counts_to_summary(summary, counts)

    model = {
        **trigram_common.standard_trigram_model_payload(
            processor,
            model_type="interpolated_trigram",
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            vocab_size=summary.vocab_size,
            text_normalization=text_normalization,
            counts=counts,
        ),
        "smoothing": smoothing,
        "interpolation_weights": {
            "unigram": summary.unigram_weight,
            "bigram": summary.bigram_weight,
            "trigram": summary.trigram_weight,
        },
    }
    ngram.write_json_model_payload(output_path, model)

    return summary


def validate_interpolation_options(options: ModelOptions) -> None:
    try:
        normalize_interpolation_weights(
            unigram_weight=options["unigram_weight"],
            bigram_weight=options["bigram_weight"],
            trigram_weight=options["trigram_weight"],
        )
    except ValueError as error:
        raise ModelOptionError(str(error)) from error


def format_summary(summary: TrigramTrainingSummary) -> list[tuple[str, str]]:
    return [
        *trigram_common.base_training_summary_items(
            summary=summary,
            artifact_label="Trigram model artifact file",
        ),
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=summary.unigram_weight,
                bigram_weight=summary.bigram_weight,
                trigram_weight=summary.trigram_weight,
            ),
        ),
    ]


def format_evaluation(summary: TrigramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=summary.unigram_weight,
                bigram_weight=summary.bigram_weight,
                trigram_weight=summary.trigram_weight,
            ),
        ),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    name=MODEL_NAME,
    model_suffix=MODEL_SUFFIX,
    model_label="Trigram",
    train_model=train_trigram_model,
    summary_items=format_summary,
    load_model=load_trigram_model,
    evaluation_items=format_evaluation,
    training_option_names=(
        "smoothing",
        "unigram_weight",
        "bigram_weight",
        "trigram_weight",
    ),
    validate_training_options=validate_interpolation_options,
)
