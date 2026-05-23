"""Interpolated add-k token-level autoregressive trigram model.

Each row estimate is ``P_k(w | h) = (c(h, w) + k) / (c(h) + k |V|)``.
The final trigram probability linearly interpolates unigram, bigram, and
trigram rows with ``lambda_1``, ``lambda_2``, and ``lambda_3``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram, trigram_interpolation as interp, trigrams
from src.tokenizers import core as tok_core


class Model(trigrams.InterpolatedTrigramModel):
    smoothing: float  # k, the additive smoothing pseudo-count.
    unigram_counts: dict[int, int]  # c(w), unigram counts.
    unigram_total: int  # N = sum_w c(w), the unigram normalizer.

    def unigram_probability(self, token_id: int) -> float:
        # token_id is w. Return P_k(w) with the empty history h.
        return ngram.additive_smoothed_probability(
            token_id,
            counts=self.unigram_counts,
            total=self.unigram_total,
            smoothing=self.smoothing,
            candidate_count=self.candidate_count,
        )

    def conditional_probability(
        self,
        token_id: int,
        *,
        counts: Mapping[int, int],
        total: int,
    ) -> float:
        # For h = v or h = (u, v), counts[token_id] is c(h, w).
        return ngram.additive_smoothed_probability(
            token_id,
            counts=counts,
            total=total,
            smoothing=self.smoothing,
            candidate_count=self.candidate_count,
        )


def load(model_path: Path) -> Model:
    return interp.load_interpolated_trigram_model(
        Model,
        model_path,
        module_name=__name__,
        extra_fields=lambda data: {"smoothing": float(data["smoothing"])},
    )


def train(
    texts: Iterable[str],
    *,
    tokenizer: tok_core.TokenizerCodec,
    smoothing: float = 0.1,
    unigram_weight: float = interp.DEFAULT_UNIGRAM_WEIGHT,
    bigram_weight: float = interp.DEFAULT_BIGRAM_WEIGHT,
    trigram_weight: float = interp.DEFAULT_TRIGRAM_WEIGHT,
    beta_2: float | None = None,
    beta_3: float | None = None,
    text_normalization: normalization.TextNormalization = normalization.DEFAULT_TEXT_NORMALIZATION,
) -> ngram.TrainingResult[trigrams.InterpolatedTrigramTrainingSummary]:
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
        tokenizer,
        text_normalization=text_normalization,
        params=interpolation,
        extra_model_payload={"smoothing": smoothing},
    )


def format_summary(
    summary: trigrams.InterpolatedTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Interpolated add-k trigram model file",
        ),
        *interp.items(summary),
    ]
