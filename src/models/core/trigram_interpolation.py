"""Interpolation parameter helpers for trigram models.

The trigram interpolation is
``lambda_1 P_1(w) + lambda_2 P_2(w | v) + lambda_3 P_3(w | u, v)``.
The recursive form uses ``beta_3 = lambda_3`` and
``beta_2 = lambda_2 / (lambda_1 + lambda_2)`` when lower-order mass is nonzero.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol, TypeVar

from src.corpora import normalization
from src.ml_core.models import definition as model_def
from src.models.core import formatting, ngram, trigrams
from src.tokenizers import core as tok_core


DEFAULT_UNIGRAM_WEIGHT = 0.1  # lambda_1.
DEFAULT_BIGRAM_WEIGHT = 0.3  # lambda_2.
DEFAULT_TRIGRAM_WEIGHT = 0.6  # lambda_3.


class InterpolationSummary(Protocol):
    unigram_weight: float  # lambda_1.
    bigram_weight: float  # lambda_2.
    trigram_weight: float  # lambda_3.
    beta_2: float | None  # beta_2, lower-order bigram share.
    beta_3: float | None  # beta_3, trigram share.


class InterpolationParams(ngram.FrozenNgramModel):
    unigram_weight: float  # lambda_1.
    bigram_weight: float  # lambda_2.
    trigram_weight: float  # lambda_3.
    beta_2: float | None = None  # beta_2, lower-order bigram share.
    beta_3: float | None = None  # beta_3, trigram share.


InterpolatedModelT = TypeVar(
    "InterpolatedModelT",
    bound=trigrams.InterpolatedTrigramModel,
)
ExtraFieldsFn = Callable[[dict[str, object]], Mapping[str, object]]


def load_interpolated_trigram_model(
    model_cls: type[InterpolatedModelT],
    model_path: Path,
    *,
    module_name: str,
    extra_fields: ExtraFieldsFn | None = None,
) -> InterpolatedModelT:
    data, model_fields = trigrams.load_standard_trigram_model_fields(
        model_path,
        module_name=module_name,
    )
    resolved_extra_fields = dict(extra_fields(data)) if extra_fields else {}

    return model_cls(
        **model_fields,
        **parse_fields(data),
        **resolved_extra_fields,
        unigram_counts=trigrams.parse_unigram_counts(data),
        unigram_total=int(data["unigram_count"]),
        bigram_transitions=trigrams.parse_bigram_transitions(data),
        trigram_transitions=trigrams.parse_trigram_transitions(data),
    )


def normalize_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float, float]:
    total = unigram_weight + bigram_weight + trigram_weight  # sum_i lambda_i.
    if total <= 0:
        raise ValueError("At least one interpolation weight must be positive.")
    # Return normalized lambda_1, lambda_2, lambda_3.
    return unigram_weight / total, bigram_weight / total, trigram_weight / total


def weights_from_betas(
    *,
    beta_2: float,
    beta_3: float,
) -> tuple[float, float, float]:
    validate_beta("beta_2", beta_2)
    validate_beta("beta_3", beta_3)
    # Recursive interpolation:
    # P = beta_3 P_3 + (1 - beta_3) [beta_2 P_2 + (1 - beta_2) P_1].
    return (1 - beta_3) * (1 - beta_2), (1 - beta_3) * beta_2, beta_3


def betas_from_weights(
    *,
    unigram_weight: float,
    bigram_weight: float,
    trigram_weight: float,
) -> tuple[float, float]:
    lower_weight = unigram_weight + bigram_weight  # lambda_1 + lambda_2.
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
    tokenizer: tok_core.TokenizerCodec,
    *,
    params: InterpolationParams,
    text_normalization: normalization.TextNormalization,
    extra_model_payload: Mapping[str, object] | None = None,
) -> ngram.TrainingResult[trigrams.InterpolatedTrigramTrainingSummary]:
    counts = trigrams.collect_trigram_counts(
        texts,
        tokenizer=tokenizer,
        text_normalization=text_normalization,
    )
    summary = trigrams.InterpolatedTrigramTrainingSummary(
        vocab_size=tokenizer.vocab_size,
        **trigrams.trigram_summary_fields(counts),
        unigram_weight=params.unigram_weight,
        bigram_weight=params.bigram_weight,
        trigram_weight=params.trigram_weight,
        beta_2=params.beta_2,
        beta_3=params.beta_3,
        text_normalization=text_normalization,
    )

    model = {
        **trigrams.trigram_counts_payload(counts),
        **dict(extra_model_payload or {}),
        **payload(summary),
    }

    return ngram.TrainingResult[trigrams.InterpolatedTrigramTrainingSummary](
        summary=summary,
        payload=model,
    )


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


def items(summary: InterpolationSummary) -> list[tuple[str, str]]:
    beta_2, beta_3 = betas(summary)
    return [
        (
            "Interpolation weights",
            formatting.format_interpolation_weights(
                unigram_weight=summary.unigram_weight,
                bigram_weight=summary.bigram_weight,
                trigram_weight=summary.trigram_weight,
            ),
        ),
        ("Interpolation betas", f"beta_2={beta_2:.3f}, beta_3={beta_3:.3f}"),
    ]
