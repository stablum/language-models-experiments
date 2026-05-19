"""Registry for trainable language models."""

from __future__ import annotations

import importlib
import pkgutil

from src.ml_core.models import definition as model_def
import src.models as model_pkg
from src.models.core import model_modules


def iter_model_definitions() -> tuple[model_def.ModelDefinition, ...]:
    definitions: list[model_def.ModelDefinition] = []
    module_infos = sorted(
        pkgutil.iter_modules(model_pkg.__path__),
        key=lambda module_info: module_info.name,
    )
    for module_info in module_infos:
        if module_info.ispkg or module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{model_pkg.__name__}.{module_info.name}")
        definition = model_modules.model_definition_from_module(module)
        if definition is not None:
            definitions.append(definition)

    return tuple(definitions)


MODELS = {definition.name: definition for definition in iter_model_definitions()}


def default_model_name() -> str:
    return next(iter(MODELS))


def model_names() -> tuple[str, ...]:
    return tuple(MODELS)


def get_model(name: str) -> model_def.ModelDefinition:
    return MODELS[name]
