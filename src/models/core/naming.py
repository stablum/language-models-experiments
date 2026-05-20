"""Naming helpers shared by model discovery and serialized artifacts."""

from __future__ import annotations


def module_leaf(module_name: str) -> str:
    return module_name.rsplit(".", maxsplit=1)[-1]


def model_type_from_module(module_name: str) -> str:
    return module_leaf(module_name)


def registered_name_from_module(module_name: str) -> str:
    return module_leaf(module_name).replace("_", "-")


def label_from_registered_name(name: str) -> str:
    return name.replace("-", " ").capitalize()


def schema_label(model_type: str) -> str:
    words = model_type.replace("_", " ")
    article = "an" if words[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {words} model"
