"""Small JSON file IO helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON file is not an object: {path}")
    return dict(payload)


def maybe_read_mapping(path: Path) -> dict[str, Any] | None:
    try:
        return read_mapping(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None


def write_json(
    path: Path,
    payload: object,
    *,
    sort_keys: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )
