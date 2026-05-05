"""Compatibility imports for language-model pipeline task definitions."""

from src.pipelines.language_model.model_training import (
    add_model_training_steps,
    add_pipeline_steps,
)
from src.pipelines.language_model.tokenizer_training import add_tokenizer_training_step


__all__ = (
    "add_model_training_steps",
    "add_pipeline_steps",
    "add_tokenizer_training_step",
)
