"""Shared model option metadata for language-model pipelines."""

from __future__ import annotations

from collections.abc import Mapping


MODEL_HYPERPARAMETER_DESCRIPTIONS = {
    "smoothing": "Add-k smoothing value for models that use it.",
    "unigram_weight": "Interpolation weight for unigram probabilities in models that use it.",
    "bigram_weight": "Interpolation weight for bigram probabilities in models that use it.",
    "trigram_weight": "Interpolation weight for trigram probabilities in models that use it.",
    "beta_2": (
        "Recursive interpolation beta for the bigram-vs-unigram branch. "
        "Set with --beta-3 to derive interpolation weights."
    ),
    "beta_3": (
        "Recursive interpolation beta for the trigram-vs-lower-order branch. "
        "Set with --beta-2 to derive interpolation weights."
    ),
    "discount": "Absolute discount value for models that use it.",
}
MODEL_HYPERPARAMETER_DEFAULTS = {
    "smoothing": 0.1,
    "unigram_weight": 0.1,
    "bigram_weight": 0.3,
    "trigram_weight": 0.6,
    "beta_2": None,
    "beta_3": None,
    "discount": 0.75,
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
    "MODEL_HYPERPARAMETER_DEFAULTS",
    "MODEL_HYPERPARAMETER_DESCRIPTIONS",
    "MODEL_HYPERPARAMETER_NAMES",
    "model_hyperparameters_from",
)
