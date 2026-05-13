"""Interpolation parameter helpers for trigram models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

import pydantic

from src.corpora import normalization
from src.ml_core.models import definition as model_def
from src.models.core import formatting, ngram, trigrams
from src.tokenizers import core as tok_core


DEFAULT_UNIGRAM_WEIGHT = 0.1
DEFAULT_BIGRAM_WEIGHT = 0.3
DEFAULT_TRIGRAM_WEIGHT = 0.6


class InterpolationSummary(Protocol):
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    beta_2: float | None
    beta_3: float | None


class InterpolationParams(ngram.FrozenNgramModel):
    unigram_weight: float
    bigram_weight: float
    trigram_weight: float
    beta_2: float | None = None
    beta_3: float | None = None


class InterpolatedTrainingSpec(ngram.FrozenNgramModel):
    model_type: str
    tokenizer_model: Path
    output_path: Path
    stored_tokenizer_model: Path | None = None
    params: InterpolationParams
    text_normalization: normalization.TextNormalization
    extra_model_payload: Mapping[str, object] = pydantic.Field(default_factory=dict)


def normalize_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float, float]:
    total = unigram_weight + bigram_weight + trigram_weight
    if total <= 0:
        raise ValueError("At least one interpolation weight must be positive.")
    return unigram_weight / total, bigram_weight / total, trigram_weight / total


def weights_from_betas(
    *,
    beta_2: float,
    beta_3: float,
) -> tuple[float, float, float]:
    validate_beta("beta_2", beta_2)
    validate_beta("beta_3", beta_3)
    return (1 - beta_3) * (1 - beta_2), (1 - beta_3) * beta_2, beta_3


def betas_from_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float]:
    lower_weight = unigram_weight + bigram_weight
    # If lambda_3 is 1, the lower-order branch is unused; beta_2 is arbitrary.
    beta_2 = bigram_weight / lower_weight if lower_weight > 0 else 0.0
    return beta_2, trigram_weight


def validate_beta(name: str, value: float) -> None:
    if value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def resolve_params(
    *,
    unigram_weight: float = DEFAULT_UNIGRAM_WEIGHT,
    bigram_weight: float = DEFAULT_BIGRAM_WEIGHT,
    trigram_weight: float = DEFAULT_TRIGRAM_WEIGHT,
    beta_2: float | None = None,
    beta_3: float | None = None,
) -> InterpolationParams:
    if (beta_2 is None) != (beta_3 is None):
        raise ValueError("Set both beta_2 and beta_3, or neither.")

    if beta_2 is not None and beta_3 is not None:
        weights = weights_from_betas(beta_2=beta_2, beta_3=beta_3)
        return InterpolationParams(
            unigram_weight=weights[0],
            bigram_weight=weights[1],
            trigram_weight=weights[2],
            beta_2=beta_2,
            beta_3=beta_3,
        )

    weights = normalize_weights(
        unigram_weight=unigram_weight,
        bigram_weight=bigram_weight,
        trigram_weight=trigram_weight,
    )
    betas = betas_from_weights(
        unigram_weight=weights[0],
        bigram_weight=weights[1],
        trigram_weight=weights[2],
    )
    return InterpolationParams(
        unigram_weight=weights[0],
        bigram_weight=weights[1],
        trigram_weight=weights[2],
        beta_2=betas[0],
        beta_3=betas[1],
    )


def train_interpolated_trigram_model(
    texts: Iterable[str],
    spec: InterpolatedTrainingSpec,
) -> trigrams.InterpolatedTrigramTrainingSummary:
    tokenizer = tok_core.load_tokenizer(spec.tokenizer_model)
    summary = trigrams.InterpolatedTrigramTrainingSummary(
        output_path=spec.output_path,
        tokenizer_model=spec.tokenizer_model,
        vocab_size=tokenizer.vocab_size,
        unigram_weight=spec.params.unigram_weight,
        bigram_weight=spec.params.bigram_weight,
        trigram_weight=spec.params.trigram_weight,
        beta_2=spec.params.beta_2,
        beta_3=spec.params.beta_3,
        text_normalization=spec.text_normalization,
    )
    counts = trigrams.collect_trigram_counts(
        texts,
        tokenizer,
        text_normalization=spec.text_normalization,
    )
    trigrams.apply_trigram_counts_to_summary(summary, counts)

    model = {
        **trigrams.standard_trigram_model_payload(
            tokenizer,
            model_type=spec.model_type,
            tokenizer_model=spec.tokenizer_model,
            stored_tokenizer_model=spec.stored_tokenizer_model,
            text_normalization=spec.text_normalization,
            counts=counts,
        ),
        **spec.extra_model_payload,
        **payload(summary),
    }
    ngram.write_json_model_payload(spec.output_path, model)

    return summary


def validate_options(options: model_def.ModelOptions) -> None:
    try:
        resolve_params(
            unigram_weight=float(options.get("unigram_weight", DEFAULT_UNIGRAM_WEIGHT)),
            bigram_weight=float(options.get("bigram_weight", DEFAULT_BIGRAM_WEIGHT)),
            trigram_weight=float(options.get("trigram_weight", DEFAULT_TRIGRAM_WEIGHT)),
            beta_2=optional_float(options.get("beta_2")),
            beta_3=optional_float(options.get("beta_3")),
        )
    except ValueError as error:
        raise model_def.ModelOptionError(str(error)) from error


def optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def payload(params: InterpolationParams | InterpolationSummary) -> dict[str, object]:
    beta_2, beta_3 = betas(params)
    return {
        "interpolation_weights": {
            "unigram": params.unigram_weight,
            "bigram": params.bigram_weight,
            "trigram": params.trigram_weight,
        },
        "interpolation_betas": {
            "beta_2": beta_2,
            "beta_3": beta_3,
        },
    }


def betas(params: InterpolationParams | InterpolationSummary) -> tuple[float, float]:
    if params.beta_2 is not None and params.beta_3 is not None:
        return params.beta_2, params.beta_3
    return betas_from_weights(
        unigram_weight=params.unigram_weight,
        bigram_weight=params.bigram_weight,
        trigram_weight=params.trigram_weight,
    )


def parse_fields(data: dict[str, object]) -> dict[str, object]:
    weights = data["interpolation_weights"]
    fields = {
        "unigram_weight": float(weights["unigram"]),
        "bigram_weight": float(weights["bigram"]),
        "trigram_weight": float(weights["trigram"]),
    }
    beta_data = data.get("interpolation_betas")
    if isinstance(beta_data, Mapping):
        fields["beta_2"] = optional_float(beta_data.get("beta_2"))
        fields["beta_3"] = optional_float(beta_data.get("beta_3"))
    else:
        beta_2, beta_3 = betas_from_weights(
            unigram_weight=fields["unigram_weight"],
            bigram_weight=fields["bigram_weight"],
            trigram_weight=fields["trigram_weight"],
        )
        fields["beta_2"] = beta_2
        fields["beta_3"] = beta_3
    return fields


def weight_item(summary: InterpolationSummary) -> tuple[str, str]:
    return (
        "Interpolation weights",
        formatting.format_interpolation_weights(
            unigram_weight=summary.unigram_weight,
            bigram_weight=summary.bigram_weight,
            trigram_weight=summary.trigram_weight,
        ),
    )


def items(summary: InterpolationSummary) -> list[tuple[str, str]]:
    beta_2, beta_3 = betas(summary)
    return [
        weight_item(summary),
        ("Interpolation betas", f"beta_2={beta_2:.3f}, beta_3={beta_3:.3f}"),
    ]


def evaluation_items(summary: ngram.NgramEvaluationSummary) -> list[tuple[str, str]]:
    return [
        *ngram.base_evaluation_items(summary),
        *items(summary),
        *formatting.format_ngram_evaluation_metrics(summary),
    ]
