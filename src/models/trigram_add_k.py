"""Interpolated add-k token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram, trigrams
from src.tokenizers import core as tok_core


_SCHEMA_TYPE = "interpolated_add_k_trigram"


class AddKTrigramModel(trigrams.InterpolatedTrigramModel):
    smoothing: float
    unigram_counts: dict[int, int]
    unigram_total: int

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
        counts: Mapping[int, int],
        total: int,
    ) -> float:
        return ngram.additive_smoothed_probability(
            token_id,
            counts=counts,
            total=total,
            smoothing=self.smoothing,
            candidate_count=ngram.candidate_token_count(self.vocab_size, self.bos_id),
        )


def load_add_k_trigram_model(model_path: Path) -> AddKTrigramModel:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        model_type=_SCHEMA_TYPE,
    )
    weights = data["interpolation_weights"]

    return AddKTrigramModel(
        **model_fields,
        smoothing=float(data["smoothing"]),
        unigram_weight=float(weights["unigram"]),
        bigram_weight=float(weights["bigram"]),
        trigram_weight=float(weights["trigram"]),
        unigram_counts=trigrams.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def train_add_k_trigram_model(
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
) -> trigrams.InterpolatedTrigramTrainingSummary:
    normalized_weights = trigrams.normalize_interpolation_weights(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
    )
    tokenizer = tok_core.load_tokenizer(tokenizer_model)
    summary = trigrams.InterpolatedTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=tokenizer.vocab_size,
        unigram_weight=normalized_weights[0],
        bigram_weight=normalized_weights[1],
        trigram_weight=normalized_weights[2],
        text_normalization=text_normalization,
    )
    counts = trigrams.collect_trigram_counts(
        texts,
        tokenizer,
        text_normalization=text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)

    model = {
        **trigrams.standard_trigram_model_payload(
            tokenizer,
            model_type=_SCHEMA_TYPE,
            tokenizer_model=tokenizer_model,
            stored_tokenizer_model=stored_tokenizer_model,
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


def format_summary(
    summary: trigrams.InterpolatedTrigramTrainingSummary,
) -> list[tuple[str, str]]:
    return [
        *trigrams.base_training_summary_items(
            summary=summary,
            artifact_label="Interpolated add-k trigram model file",
        ),
        trigrams.interpolation_weight_item(summary),
    ]


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_add_k_trigram_model,
    summary_items=format_summary,
    load_model=load_add_k_trigram_model,
    evaluation_items=trigrams.interpolated_evaluation_items,
    training_option_names=(
        "smoothing",
        "unigram_weight",
        "bigram_weight",
        "trigram_weight",
    ),
    validate_training_options=trigrams.validate_interpolation_options,
)
