"""Fixed-lambda Jelinek-Mercer token-level autoregressive trigram model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from src.corpora import normalization
from src.models.core import ngram, trigram_interpolation as interp, trigrams
from src.tokenizers import core as tok_core


_SCHEMA_TYPE = "jelinek_mercer_trigram"


class JelinekMercerTrigramModel(trigrams.InterpolatedTrigramModel):
    unigram_counts: dict[int, int]
    unigram_total: int

    def unigram_probability(self, token_id: int) -> float:
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
        return ngram.maximum_likelihood_probability(
            token_id,
            counts=counts,
            total=total,
        )


def load_jelinek_mercer_trigram_model(model_path: Path) -> JelinekMercerTrigramModel:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        model_type=_SCHEMA_TYPE,
    )

    return JelinekMercerTrigramModel(
        **model_fields,
        **interp.parse_fields(data),
        unigram_counts=trigrams.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def train_jelinek_mercer_trigram_model(
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
    interpolation = interp.resolve_params(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
        beta_2=beta_2,
        beta_3=beta_3,
    )
    tokenizer = tok_core.load_tokenizer(tokenizer_model)
    summary = trigrams.InterpolatedTrigramTrainingSummary(
        output_path=output_path,
        tokenizer_model=tokenizer_model,
        vocab_size=tokenizer.vocab_size,
        unigram_weight=interpolation.unigram_weight,
        bigram_weight=interpolation.bigram_weight,
        trigram_weight=interpolation.trigram_weight,
        beta_2=interpolation.beta_2,
        beta_3=interpolation.beta_3,
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
        **interp.payload(summary),
    }
    ngram.write_json_model_payload(output_path, model)

    return summary


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


MODEL_DEFINITION = ngram.model_definition(
    module_name=__name__,
    train_model=train_jelinek_mercer_trigram_model,
    summary_items=format_summary,
    load_model=load_jelinek_mercer_trigram_model,
    evaluation_items=interp.evaluation_items,
    training_option_names=(
        "unigram_weight",
        "bigram_weight",
        "trigram_weight",
        "beta_2",
        "beta_3",
    ),
    validate_training_options=interp.validate_options,
)
