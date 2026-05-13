"""Shared model option metadata for language-model pipelines."""

from __future__ import annotations

from collections.abc import Mapping


MODEL_HYPERPARAMETER_DESCRIPTIONS = {
    "smoothing": "Add-k smoothing value for models that use it.",
    "unigram_weight": "Interpolation weight for unigram probabilities in models that use it.",
    "bigram_weight": "Interpolation weight for bigram probabilities in models that use it.",
    "trigram_weight": "Interpolation weight for trigram probabilities in models that use it.",
    "beta_2": "Recursive interpolation beta for the bigram-vs-unigram branch.",
    "beta_3": "Recursive interpolation beta for the trigram-vs-lower-order branch.",
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


__all__ = (
    "MODEL_HYPERPARAMETER_DESCRIPTIONS",
    "MODEL_HYPERPARAMETER_NAMES",
    "model_hyperparameters_from",
)
