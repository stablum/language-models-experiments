"""Absolute-discount token-level autoregressive trigram model.

For a history h, absolute discounting uses
``max(c(h, w) - D, 0) / c(h) + lambda(h) * P_lower(w)``. Here
``lambda(h) = D * T(h) / c(h)``, with T(h) the number of observed next-token
types in the row. This model backs off from trigram rows to additively-smoothed
bigram rows.

Notation in comments uses ``h = (u, v)`` for the trigram history, ``w`` for the
candidate next token, ``D`` for the discount, and ``k`` for add-k smoothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

from src.corpora import normalization
from src.models.core import ngram
from src.models.core import trigrams


_SCHEMA_TYPE = "absolute_discount_trigram"


class AbsoluteDiscountTrigramTrainingSummary(trigrams.TrigramTrainingSummary):
    discount: float = 0.0  # D, the absolute discount.


class AbsoluteDiscountTrigramModel(trigrams.DiscountedTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        trigrams.DiscountedTrigramEvaluationSummary
    )
    smoothing: float  # k, the lower-order add-k pseudo-count.

    def context_probability(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        return self.trigram_probability(next_id, counts)

    def trigram_probability(
        self,
        token_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        # token_id is w. counts.trigram_counts[w] is c(h, w), and
        # counts.trigram_total is c(h) for h = (u, v).
        # Absolute discounting removes D mass from every observed trigram type.
        # The helper redistributes the total removed mass through this lower
        # order bigram probability.
        lower_order_probability = self.lower_order_probability(
            token_id,
            counts=counts.bigram_counts,
            total=counts.bigram_total,
        )
        return ngram.discounted_interpolation_probability(
            token_id,
            counts=counts.trigram_counts,
            total=counts.trigram_total,
            discount=self.discount,
            lower_order_probability=lower_order_probability,
        )

    def lower_order_probability(
        self,
        token_id: int,
        *,
        counts: dict[int, int],
        total: int,
    ) -> float:
        # The lower-order history is h = v. Return add-k P_k(w | v).
        # Unlike Kneser-Ney, this model backs off to ordinary bigram counts.
        # Additive smoothing gives every candidate next token a non-zero floor.
        return ngram.additive_smoothed_probability(
            token_id,
            counts=counts,
            total=total,
            smoothing=self.smoothing,
            candidate_count=self.candidate_count,
        )


def load(model_path: Path) -> AbsoluteDiscountTrigramModel:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return AbsoluteDiscountTrigramModel(
        **model_fields,
        smoothing=float(data["smoothing"]),
        discount=float(data["discount"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def train(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    smoothing: float = 0.1,
    discount: float = 0.75,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> AbsoluteDiscountTrigramTrainingSummary:
    artifacts = trigrams.collect_training_artifacts(
        texts,
        tokenizer_model=tokenizer_model,
        text_normalization=text_normalization,
    )
    summary = AbsoluteDiscountTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=artifacts.tokenizer.vocab_size,
        discount=discount,
        text_normalization=text_normalization,
    )
    # Training stores raw trigram and bigram counts; discounting and additive
    # smoothing are applied lazily when probabilities are queried.
    trigrams.apply_trigram_counts_to_summary(summary, artifacts.counts)

    model = {
        **trigrams.standard_trigram_model_payload(
            artifacts.tokenizer,
            model_type=_SCHEMA_TYPE,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            text_normalization=text_normalization,
            counts=artifacts.counts,
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
            artifact_label="Absolute-discount trigram model file",
        ),
        trigrams.discount_item(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train,
    summary_items=format_summary,
    load_model=load,
    evaluation_items=trigrams.discounted_evaluation_items,
    training_option_names=("smoothing", "discount"),
)
