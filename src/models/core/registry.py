"""Registry for trainable language-model modules."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from pathlib import Path

import src.models as model_pkg
from src.models.core import model_modules


def model_source_path(module_name: str) -> Path | None:
    """Return the Python source path for a model module, when available."""
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def iter_registered_models() -> tuple[model_modules.RegisteredModel, ...]:
    """Import discoverable modules and yield those matching the model contract."""
    models: list[model_modules.RegisteredModel] = []
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
        model = model_modules.registered_model_from_module(module)
        if model is not None:
            models.append(model)

    return tuple(models)


MODELS = {model.name: model for model in iter_registered_models()}


def default_model_name() -> str:
    """Return the first registered model name for CLI defaults."""
    return next(iter(MODELS))


def model_names() -> tuple[str, ...]:
    """Return registered model names in discovery order."""
    return tuple(MODELS)


def get_model(name: str) -> model_modules.RegisteredModel:
    """Return the registered model matching a CLI name."""
    return MODELS[name]
