"""Registry for trainable language models."""

from __future__ import annotations

from src.ml_core.models.definition import ModelDefinition
from src.models import bigram, trigram, trigram_absolute_discount, trigram_kneser_ney


MODELS = {
    model.name: model
    for model in (
        bigram.MODEL_DEFINITION,
        trigram.MODEL_DEFINITION,
        trigram_absolute_discount.MODEL_DEFINITION,
        trigram_kneser_ney.MODEL_DEFINITION,
    )
}


def default_model_name() -> str:
    return next(iter(MODELS))


def model_names() -> tuple[str, ...]:
    return tuple(MODELS)


def get_model(name: str) -> ModelDefinition:
    return MODELS[name]
