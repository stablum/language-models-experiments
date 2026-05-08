"""Absolute-discount token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import sentencepiece as spm

from src.corpora import normalization
from src.models.core import ngram
from src.models.core import trigrams


_SCHEMA_TYPE = "absolute_discount_trigram"


class AbsoluteDiscountTrigramTrainingSummary(trigrams.TrigramTrainingSummary):
    discount: float = 0.0


class AbsoluteDiscountTrigramModel(trigrams.DiscountedTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        trigrams.DiscountedTrigramEvaluationSummary
    )
    smoothing: float

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
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
        return ngram.additive_smoothed_probability(
            token_id,
            counts=counts,
            total=total,
            smoothing=self.smoothing,
            candidate_count=ngram.candidate_token_count(self.vocab_size, self.bos_id),
        )


def load_absolute_discount_trigram_model(model_path: Path) -> AbsoluteDiscountTrigramModel:
    data, tokenizer_model, processor, vocab_size = trigrams.load_standard_trigram_payload(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return AbsoluteDiscountTrigramModel(
        model_path=model_path,
        tokenizer_model=tokenizer_model,
        processor=processor,
        **ngram.sentencepiece_model_fields(data, processor, vocab_size),
        smoothing=float(data["smoothing"]),
        discount=float(data["discount"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
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
    counts = trigrams.collect_trigram_counts(
        texts,
        processor,
        text_normalization=text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)

    model = {
        **trigrams.standard_trigram_model_payload(
            processor,
            model_type=_SCHEMA_TYPE,
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


def format_summary(
    summary: AbsoluteDiscountTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Absolute-discount trigram model artifact file",
        ),
        trigrams.discount_item(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_absolute_discount_trigram_model,
    summary_items=format_summary,
    load_model=load_absolute_discount_trigram_model,
    evaluation_items=trigrams.discounted_evaluation_items,
    training_option_names=("smoothing", "discount"),
)
