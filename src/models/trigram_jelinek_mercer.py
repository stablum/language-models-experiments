"""Fixed-lambda Jelinek-Mercer token-level autoregressive trigram model.

For history ``h = (u, v)`` and next token ``w``, the interpolation is
``lambda_3 P_ML(w | u, v) + lambda_2 P_ML(w | v) + lambda_1 P_ML(w)``.
The optional ``beta_2`` and ``beta_3`` params are the recursive backoff weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from src.models.core import ngram, trigram_interpolation as interp, trigrams


CONTEXT_LENGTH = trigrams.CONTEXT_LENGTH  # len(h), inherited trigram history size.


class Model(trigrams.InterpolatedTrigramModel):
    unigram_counts: dict[int, int]  # c(w), unigram counts.
    unigram_tot: int  # tot = N = sum_w c(w), the unigram normalizer.

    def unigram_prob(self, token_id: int) -> float:
        # token_id is w. Return P_ML(w) = c(w) / N.
        return ngram.ml_prob(
            token_id,
            counts=self.unigram_counts,
            tot=self.unigram_tot,
        )

    def conditional_prob(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        tot: int,
    ) -> float:
        # For h = v or h = (u, v), return P_ML(w | h) = c(h, w) / c(h).
        return ngram.ml_prob(
            token_id,
            counts=counts,
            tot=tot,
        )


def load(model_path: Path) -> Model:
    return interp.load_interpolated_trigram_model(
        Model,
        model_path,
        module_name=__name__,
    )


def fit(
    tok_seqs: Iterable[Sequence[int]],
    *,
    unigram_weight: float = interp.DEFAULT_UNIGRAM_WEIGHT,
    bigram_weight: float = interp.DEFAULT_BIGRAM_WEIGHT,
    trigram_weight: float = interp.DEFAULT_TRIGRAM_WEIGHT,
    beta_2: float | None = None,
    beta_3: float | None = None,
) -> ngram.TrainingResult[trigrams.InterpolatedTrigramTrainingSummary]:
    """Fit Jelinek-Mercer trigram counts from token ID sequences."""
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
        params=interp_params,
        summary_type=trigrams.InterpolatedTrigramTrainingSummary,
    )
