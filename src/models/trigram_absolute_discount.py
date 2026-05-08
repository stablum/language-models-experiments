"""Absolute-discount token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import sentencepiece as spm

from src.corpora import normalization
from src.models import formatting, ngram, trigram_common
from src.ml_core.models.definition import ModelDefinition, ModelOptions


MODEL_NAME = "trigram-absolute-discount"
MODEL_SUFFIX = "trigram-absolute-discount"


class AbsoluteDiscountTrigramTrainingSummary(ngram.NgramPydanticModel):
    output_path: Path
    tokenizer_model: Path
    vocab_size: int = 0
    sequence_count: int = 0
    token_count: int = 0
    unigram_count: int = 0
    bigram_transition_count: int = 0
    trigram_transition_count: int = 0
    discount: float = 0.0
    text_normalization: str = "none"


class AbsoluteDiscountTrigramEvaluationSummary(ngram.NgramEvaluationSummary):
    discount: float = 0.0


class AbsoluteDiscountTrigramModel(trigram_common.BaseTrigramModel):
    model_path: Path
    tokenizer_model: Path
    processor: spm.SentencePieceProcessor
    vocab_size: int
    smoothing: float
    discount: float
    bos_id: int
    eos_id: int
    unk_id: int
    pieces: tuple[str, ...]
    bigram_transitions: dict[int, tuple[tuple[int, int], ...]]
    trigram_transitions: dict[trigram_common.Context, tuple[tuple[int, int], ...]]
    text_normalization: str = "none"

    def evaluation_summary(
        self,
        *,
        top_k: int,
        text_normalization: str,
    ) -> AbsoluteDiscountTrigramEvaluationSummary:
        return AbsoluteDiscountTrigramEvaluationSummary(
            model_path=self.model_path,
            tokenizer_model=self.tokenizer_model,
            top_k=top_k,
            text_normalization=text_normalization,
            discount=self.discount,
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

        return self.trigram_probability(
            next_id,
            bigram_counts=counts.bigram_counts,
            trigram_counts=counts.trigram_counts,
            bigram_total=counts.bigram_total,
            trigram_total=counts.trigram_total,
        )

    def trigram_probability(
        self,
        token_id: int,
        *,
        bigram_counts: dict[int, int],
        trigram_counts: dict[int, int],
        bigram_total: int,
        trigram_total: int,
    ) -> float:
        lower_order_probability = self.lower_order_probability(
            token_id,
            counts=bigram_counts,
            total=bigram_total,
        )
        if trigram_total <= 0:
            return lower_order_probability

        observed_count = trigram_counts.get(token_id, 0)
        discounted_probability = max(observed_count - self.discount, 0.0) / trigram_total
        backoff_weight = self.discount * len(trigram_counts) / trigram_total
        return discounted_probability + backoff_weight * lower_order_probability

    def lower_order_probability(
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


def load_absolute_discount_trigram_model(model_path: Path) -> AbsoluteDiscountTrigramModel:
    data, tokenizer_model, processor, vocab_size = trigram_common.load_standard_trigram_payload(
        model_path,
        model_type="absolute_discount_trigram",
        label="an absolute-discount trigram model",
    )

    return AbsoluteDiscountTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        vocab_size=vocab_size,
        smoothing=float(data["smoothing"]),
        discount=float(data["discount"]),
        bos_id=int(data["bos_id"]),
        eos_id=int(data["eos_id"]),
        unk_id=int(data["unk_id"]),
        pieces=ngram.load_pieces(data, processor, vocab_size),
        bigram_transitions=trigram_common.parse_bigram_transitions(data),
        trigram_transitions=trigram_common.parse_trigram_transitions(data),
        text_normalization=str(data.get("text_normalization", "none")),
    )


def train_absolute_discount_trigram_model(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    smoothing: float = 0.1,
    discount: float = 0.75,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> AbsoluteDiscountTrigramTrainingSummary:
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    summary = AbsoluteDiscountTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=processor.get_piece_size(),
        discount=discount,
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
            model_type="absolute_discount_trigram",
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            vocab_size=summary.vocab_size,
            text_normalization=text_normalization,
            counts=counts,
        ),
        "smoothing": smoothing,
        "discount": summary.discount,
    }
    ngram.write_json_model_payload(output_path, model)

    return summary


def default_tokenizer_model(corpus: str) -> Path:
    return ngram.default_tokenizer_model(corpus)


def default_output(corpus: str) -> Path:
    return ngram.default_ngram_output(corpus, MODEL_SUFFIX)


def default_model(corpus: str) -> Path:
    return default_output(corpus)


def resolve_tokenizer_model(options: ModelOptions) -> Path:
    return ngram.resolve_tokenizer_model(options)


def resolve_output(options: ModelOptions) -> Path:
    return ngram.resolve_output(options, model_suffix=MODEL_SUFFIX)


def resolve_model(options: ModelOptions) -> Path:
    return ngram.resolve_model(options, model_suffix=MODEL_SUFFIX)


def validate_options(options: ModelOptions) -> None:
    ngram.validate_tokenizer_model(options)


def validate_query_options(options: ModelOptions) -> None:
    ngram.validate_model_path(
        options,
        model_suffix=MODEL_SUFFIX,
        label="Absolute-discount trigram",
    )


def train_from_options(
    texts: Iterable[str],
    options: ModelOptions,
) -> AbsoluteDiscountTrigramTrainingSummary:
    stored_tokenizer_model = options.get("stored_tokenizer_model")
    return train_absolute_discount_trigram_model(
        texts,
        tokenizer_model=resolve_tokenizer_model(options),
        output_path=resolve_output(options),
        stored_tokenizer_model=Path(stored_tokenizer_model) if stored_tokenizer_model else None,
        smoothing=options["smoothing"],
        discount=options["discount"],
        text_normalization=options["text_normalization"],
    )


def query_from_options(options: ModelOptions) -> trigram_common.TrigramQueryResult:
    return ngram.query_from_options(
        options,
        load_model=load_absolute_discount_trigram_model,
        model_suffix=MODEL_SUFFIX,
    )


def evaluate_from_options(
    texts: Iterable[str],
    options: ModelOptions,
) -> AbsoluteDiscountTrigramEvaluationSummary:
    return ngram.evaluate_from_options(
        texts,
        options,
        load_model=load_absolute_discount_trigram_model,
        model_suffix=MODEL_SUFFIX,
    )


def format_summary(
    summary: AbsoluteDiscountTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigram_common.base_training_summary_items(
            summary=summary,
            artifact_label="Absolute-discount trigram model artifact file",
        ),
        ("Discount", f"{summary.discount:.3f}"),
    ]


def format_evaluation(
    summary: AbsoluteDiscountTrigramEvaluationSummary,
) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        ("Discount", f"{summary.discount:.3f}"),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]


def format_query(result: trigram_common.TrigramQueryResult) -> list[str]:
    return formatting.format_ngram_query(result)


MODEL_DEFINITION = ModelDefinition(
    name=MODEL_NAME,
    train=train_from_options,
    validate_options=validate_options,
    summary_items=format_summary,
    query=query_from_options,
    validate_query_options=validate_query_options,
    query_lines=format_query,
    evaluate=evaluate_from_options,
    validate_evaluation_options=validate_query_options,
    evaluation_items=format_evaluation,
)
