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

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from src.models.core import ngram
from src.models.core import trigrams


CONTEXT_LENGTH = trigrams.CONTEXT_LENGTH  # len(h), inherited trigram history size.


class TrainingSummary(trigrams.TrigramTrainingSummary):
    """Store absolute-discount hyperparams with trigram count metadata."""

    smoothing: float = 0.0  # k, the lower-order add-k pseudo-count.
    discount: float = 0.0  # D, the absolute discount.


class Model(trigrams.DiscountedTrigramModel):
    evaluation_summary_type: ClassVar[type[ngram.NgramEvaluationSummary]] = (
        trigrams.DiscountedTrigramEvaluationSummary
    )
    smoothing: float  # k, the lower-order add-k pseudo-count.

    def context_prob(
        self,
        next_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        return self._trigram_prob(next_id, counts)

    def _trigram_prob(
        self,
        token_id: int,
        counts: trigrams.ResolvedTrigramContextCounts,
    ) -> float:
        # token_id is w. counts.trigram_counts[w] is c(h, w), and
        # counts.trigram_tot is c(h) for h = (u, v).
        # Absolute discounting removes D mass from every observed trigram type.
        # The helper redistributes the total removed mass through this lower
        # order bigram probability.
        lower_prob = self._lower_order_prob(
            token_id,
            counts=counts.bigram_counts,
            tot=counts.bigram_tot,
        )
        return ngram.discounted_interp_prob(
            token_id,
            counts=counts.trigram_counts,
            tot=counts.trigram_tot,
            discount=self.discount,
            lower_prob=lower_prob,
        )

    def _lower_order_prob(
        self,
        token_id: int,
        *,
        counts: dict[int, int],
        tot: int,
    ) -> float:
        # The lower-order history is h = v. Return add-k P_k(w | v).
        # Unlike Kneser-Ney, this model backs off to ordinary bigram counts.
        # Additive smoothing gives every candidate next token a non-zero floor.
        return ngram.add_k_prob(
            token_id,
            counts=counts,
            tot=tot,
            smoothing=self.smoothing,
            cand_count=self.cand_count,
        )


def load(model_path: Path) -> Model:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        module_name=__name__,
    )

    return Model(
        **model_fields,
        smoothing=float(data["smoothing"]),
        discount=float(data["discount"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def fit(
    tok_seqs: Iterable[Sequence[int]],
    *,
    token_space: ngram.TokenSpace,
    smoothing: float = 0.1,
    discount: float = 0.75,
) -> ngram.TrainingResult[TrainingSummary]:
    """Fit absolute-discount trigram counts from token ID sequences."""
    def payload(
        _counts: trigrams.TrigramCounts,
        summary: TrainingSummary,
    ) -> dict[str, object]:
        # Training stores raw counts; smoothing/discounting are applied lazily.
        return {"smoothing": summary.smoothing, "discount": summary.discount}

    return trigrams.fit_counted_trigram_model(
        tok_seqs,
        token_space=token_space,
        summary_type=TrainingSummary,
        summary_fields={"smoothing": smoothing, "discount": discount},
        extra_payload=payload,
    )
