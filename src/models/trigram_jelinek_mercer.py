"""Fixed-lambda Jelinek-Mercer token-level autoregressive trigram model.

For history ``h = (u, v)`` and next token ``w``, the interpolation is
``lambda_3 P_ML(w | u, v) + lambda_2 P_ML(w | v) + lambda_1 P_ML(w)``.
The optional ``beta_2`` and ``beta_3`` params are the recursive backoff weights.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram, trigram_interpolation as interp, trigrams


_SCHEMA_TYPE = "jelinek_mercer_trigram"


class JelinekMercerTrigramModel(trigrams.InterpolatedTrigramModel):
    unigram_counts: dict[int, int]  # c(w), unigram counts.
    unigram_total: int  # N = sum_w c(w), the unigram normalizer.

    def unigram_probability(self, token_id: int) -> float:
        # token_id is w. Return P_ML(w) = c(w) / N.
        return ngram.maximum_likelihood_probability(
            token_id,
            counts=self.unigram_counts,
            total=self.unigram_total,
        )

    def conditional_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
    ) -> float:
        # For h = v or h = (u, v), return P_ML(w | h) = c(h, w) / c(h).
        return ngram.maximum_likelihood_probability(
            token_id,
            counts=counts,
            total=total,
        )


def load(model_path: Path) -> JelinekMercerTrigramModel:
    return interp.load_interpolated_trigram_model(
        JelinekMercerTrigramModel,
        model_path,
        model_type=_SCHEMA_TYPE,
    )


def train(
    texts: Iterable[str],
    *,
    tokenizer_model: Path,
    output_path: Path,
    stored_tokenizer_model: Path | None = None,
    unigram_weight: float = interp.DEFAULT_UNIGRAM_WEIGHT,
    bigram_weight: float = interp.DEFAULT_BIGRAM_WEIGHT,
    trigram_weight: float = interp.DEFAULT_TRIGRAM_WEIGHT,
    beta_2: float | None = None,
    beta_3: float | None = None,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> trigrams.InterpolatedTrigramTrainingSummary:
    # lambda_i are stored as weights; beta_i are an equivalent recursive form.
    interpolation = interp.resolve_params(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
        beta_2=beta_2,
        beta_3=beta_3,
    )
    return interp.train_interpolated_trigram_model(
        texts,
        interp.InterpolatedTrainingSpec(
            model_type=_SCHEMA_TYPE,
            output_path=output_path,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
            text_normalization=text_normalization,
            params=interpolation,
        ),
    )


def format_summary(
    summary: trigrams.InterpolatedTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Jelinek-Mercer trigram model file",
        ),
        *interp.items(summary),
    ]
