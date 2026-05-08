"""Interpolated token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm

from src.corpora import normalization
from src.models import formatting, ngram, trigram_common
from src.ml_core.models.definition import ModelOptionError, ModelOptions


MODEL_NAME = "trigram"
MODEL_SUFFIX = "trigram"


class TrigramTrainingSummary(ngram.NgramPydanticModel):
    output_path: Path
    tokenizer_model: Path
    vocab_size: int = 0
    sequence_count: int = 0
    token_count: int = 0
    unigram_count: int = 0
    bigram_transition_count: int = 0
    trigram_transition_count: int = 0
    unigram_weight: float = 0.0
    bigram_weight: float = 0.0
    trigram_weight: float = 0.0
    text_normalization: str = "none"


class TrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    unigram_weight: float = 0.0
    bigram_weight: float = 0.0
    trigram_weight: float = 0.0


class TrigramModel(trigram_common.BaseTrigramModel):
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    smoothing: float
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    unigram_counts: dict[int, int]
    unigram_total: int
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[trigram_common.Context, tuple[tuple[int, int], ...]]
    text_normalization: str = "none"

    def evaluation_summary(
        self,
        *,
        top_k: int,
        text_normalization: str,
    ) -> TrigramEvaluationSummary:
        return TrigramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            text_normalization=text_normalization,
            unigram_weight=self.unigram_weight,
            bigram_weight=self.bigram_weight,
            trigram_weight=self.trigram_weight,
        )

    def transition_probability(
        self,
        next_id: int,
        context: trigram_common.Context,
        *,
        row: trigram_common.TrigramEvaluationRow | None = None,
        bigram_counts: dict[int, int] | None = None,
        trigram_counts: dict[int, int] | None = None,
        bigram_total: int | None = None,
        trigram_total: int | None = None,
    ) -> float:
        if next_id == self.bos_id:
            return 0.0

        counts = self.resolved_context_counts(
            context,
            row=row,
            bigram_counts=bigram_counts,
            trigram_counts=trigram_counts,
            bigram_total=bigram_total,
            trigram_total=trigram_total,
        )

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
        denominator = (
            self.unigram_total
            + self.smoothing * ngram.candidate_token_count(self.vocab_size, self.bos_id)
        )
        if denominator <= 0:
            return 0.0
        return (self.unigram_counts.get(token_id, 0) + self.smoothing) / denominator

    def conditional_probability(
        self,
        token_id: int,
        *,
        counts: dict[int, int],
        total: int,
    ) -> float:
        denominator = total + self.smoothing * ngram.candidate_token_count(
            self.vocab_size,
            self.bos_id,
        )
        if denominator <= 0:
            return 0.0
        return (counts.get(token_id, 0) + self.smoothing) / denominator


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
        vocab_size=vocab_size,
        smoothing=float(data["smoothing"]),
        unigram_weight=float(weights["unigram"]),
        bigram_weight=float(weights["bigram"]),
        trigram_weight=float(weights["trigram"]),
        bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]),
        unk_id=int(data["unk_id"]),
        pieces=ngram.load_pieces(data, processor, vocab_size),
        unigram_counts=trigram_common.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigram_common.parse_bigram_transitions(data),
        trigram_transitions=trigram_common.parse_trigram_transitions(data),
        text_normalization=str(data.get("text_normalization", "none")),
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


def train_from_options(
    texts: Iterable[str],
    options: ModelOptions,
) -> TrigramTrainingSummary:
    stored_tokenizer_model = options.get("stored_tokenizer_model")
    return train_trigram_model(
        texts,
        tokenizer_model=ngram.resolve_tokenizer_model(options),
        output_path=ngram.resolve_output(options, model_suffix=MODEL_SUFFIX),
        stored_tokenizer_model=Path(stored_tokenizer_model) if stored_tokenizer_model else None,
        smoothing=options["smoothing"],
        unigram_weight=options["unigram_weight"],
        bigram_weight=options["bigram_weight"],
        trigram_weight=options["trigram_weight"],
        text_normalization=options["text_normalization"],
    )


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


def format_query(result: trigram_common.TrigramQueryResult) -> list[str]:
    return formatting.format_ngram_query(result)


MODEL_DEFINITION = ngram.model_definition(
    name=MODEL_NAME,
    model_suffix=MODEL_SUFFIX,
    model_label="Trigram",
    train=train_from_options,
    summary_items=format_summary,
    load_model=load_trigram_model,
    query_lines=format_query,
    evaluation_items=format_evaluation,
    validate_training_options=validate_interpolation_options,
)
