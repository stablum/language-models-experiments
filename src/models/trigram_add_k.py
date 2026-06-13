"""Interpolated add-k token-level autoregressive trigram model.

Each row estimate is ``P_k(w | h) = (c(h, w) + k) / (c(h) + k |V|)``.
The final trigram probability linearly interpolates unigram, bigram, and
trigram rows with ``lambda_1``, ``lambda_2``, and ``lambda_3``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from src.models.core import ngram, trigram_interpolation as interp, trigrams


CONTEXT_LENGTH = trigrams.CONTEXT_LENGTH  # len(h), inherited trigram history size.


class TrainingSummary(trigrams.InterpolatedTrigramTrainingSummary):
    """Store add-k smoothing together with interpolation training metadata."""

    smoothing: float = 0.0  # k, the additive smoothing pseudo-count.


class Model(trigrams.InterpolatedTrigramModel):
    smoothing: float  # k, the additive smoothing pseudo-count.
    unigram_counts: dict[int, int]  # c(w), unigram counts.
    unigram_tot: int  # tot = N = sum_w c(w), the unigram normalizer.

    def unigram_prob(self, token_id: int) -> float:
        # token_id is w. Return P_k(w) with the empty history h.
        return ngram.add_k_prob(
            token_id,
            counts=self.unigram_counts,
            tot=self.unigram_tot,
            smoothing=self.smoothing,
            cand_count=self.cand_count,
        )

    def conditional_prob(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        tot: int,
    ) -> float:
        # For h = v or h = (u, v), counts[token_id] is c(h, w).
        return ngram.add_k_prob(
            token_id,
            counts=counts,
            tot=tot,
            smoothing=self.smoothing,
            cand_count=self.cand_count,
        )


def load(model_path: Path) -> Model:
    return interp.load_interpolated_trigram_model(
        Model,
        model_path,
        module_name=__name__,
        extra_fields=lambda data: {"smoothing": float(data["smoothing"])},
    )


def fit(
    tok_seqs: Iterable[Sequence[int]],
    *,
    token_space: ngram.TokenSpace,
    smoothing: float = 0.1,
    unigram_weight: float = interp.DEFAULT_UNIGRAM_WEIGHT,
    bigram_weight: float = interp.DEFAULT_BIGRAM_WEIGHT,
    trigram_weight: float = interp.DEFAULT_TRIGRAM_WEIGHT,
    beta_2: float | None = None,
    beta_3: float | None = None,
) -> ngram.TrainingResult[TrainingSummary]:
    """Fit interpolated add-k trigram counts from token ID sequences."""
    # lambda_i are stored as weights; beta_i are an equivalent recursive form.
    interp_params = interp.resolve_params(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
        beta_2=beta_2,
        beta_3=beta_3,
    )
    return interp.fit_interpolated_trigram_model(
        tok_seqs,
        token_space=token_space,
        params=interp_params,
        summary_type=TrainingSummary,
        summary_fields={"smoothing": smoothing},
        extra_model_payload={"smoothing": smoothing},
    )
