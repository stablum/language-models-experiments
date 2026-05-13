"""Shared Pydantic cfg (configuration) models."""

from __future__ import annotations

import pydantic


class BaseCfg(pydantic.BaseModel):
    """Base cfg (configuration) model for grouped runtime options."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class FrozenBaseCfg(BaseCfg):
    """Immutable cfg (configuration) model for small value objects."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)


__all__ = ("BaseCfg", "FrozenBaseCfg")
