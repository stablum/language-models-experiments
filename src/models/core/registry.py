"""Registry for trainable language models."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from pathlib import Path

from src.ml_core.models import definition as model_def
import src.models as model_pkg
from src.models.core import model_modules


def model_source_path(module_name: str) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def iter_model_definitions() -> tuple[model_def.ModelDefinition, ...]:
    definitions: list[model_def.ModelDefinition] = []
    module_infos = sorted(
        pkgutil.iter_modules(model_pkg.__path__),
        key=lambda module_info: module_info.name,
    )
    for module_info in module_infos:
        if module_info.ispkg or module_info.name.startswith("_"):
            continue

        module_name = f"{model_pkg.__name__}.{module_info.name}"
        module_path = model_source_path(module_name)
        if module_path is not None and not model_modules.registry_enabled(
            module_path,
            module_name=module_name,
        ):
            continue

        module = importlib.import_module(module_name)
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
