"""Shared model option metadata for language-model pipelines."""

from __future__ import annotations

from collections.abc import Mapping


MODEL_HYPERPARAMETER_DESCRIPTIONS = {
    "smoothing": "Add-k smoothing value for models that use it.",
    "unigram_weight": "Interpolation weight for unigram probabilities in models that use it.",
    "bigram_weight": "Interpolation weight for bigram probabilities in models that use it.",
    "trigram_weight": "Interpolation weight for trigram probabilities in models that use it.",
    "discount": "Absolute discount value for models that use it.",
}
MODEL_HYPERPARAMETER_NAMES = tuple(MODEL_HYPERPARAMETER_DESCRIPTIONS)


def model_hyperparameters_from(values: Mapping[str, object]) -> dict[str, object]:
    """Extract supported model hyperparameters from a larger value mapping."""
    return {
        name: values[name]
        for name in MODEL_HYPERPARAMETER_NAMES
        if name in values
    }


def merge_model_hyperparameters(
    grouped: Mapping[str, object] | None,
    legacy_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Merge grouped model hyperparameters with legacy top-level kwargs."""
    hyperparameters = dict(grouped or {})
    hyperparameters.update(model_hyperparameters_from(legacy_kwargs))
    return hyperparameters


__all__ = (
    "MODEL_HYPERPARAMETER_DESCRIPTIONS",
    "MODEL_HYPERPARAMETER_NAMES",
    "merge_model_hyperparameters",
    "model_hyperparameters_from",
)
