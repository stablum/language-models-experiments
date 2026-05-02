"""Registry for trainable language models."""

from __future__ import annotations

from src.models.bigram import MODEL_DEFINITION as BIGRAM_MODEL
from src.models.definition import ModelDefinition
from src.models.trigram import MODEL_DEFINITION as TRIGRAM_MODEL
from src.models.trigram_absolute_discount import (
    MODEL_DEFINITION as ABSOLUTE_DISCOUNT_TRIGRAM_MODEL,
)
from src.models.trigram_kneser_ney import MODEL_DEFINITION as KNESER_NEY_TRIGRAM_MODEL


DEFAULT_MODEL_NAME = BIGRAM_MODEL.name
TRIGRAM_MODEL_NAME = TRIGRAM_MODEL.name
ABSOLUTE_DISCOUNT_TRIGRAM_MODEL_NAME = ABSOLUTE_DISCOUNT_TRIGRAM_MODEL.name
KNESER_NEY_TRIGRAM_MODEL_NAME = KNESER_NEY_TRIGRAM_MODEL.name


MODELS = {
    model.name: model
    for model in (
        BIGRAM_MODEL,
        TRIGRAM_MODEL,
        ABSOLUTE_DISCOUNT_TRIGRAM_MODEL,
        KNESER_NEY_TRIGRAM_MODEL,
    )
}


def model_names() -> tuple[str, ...]:
    return tuple(MODELS)


def get_model(name: str) -> ModelDefinition:
    return MODELS[name]
