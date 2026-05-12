"""Shared Pydantic cfg (configuration) models."""

from __future__ import annotations

import pydantic


class BaseCfg(pydantic.BaseModel):
    """Base cfg (configuration) model for grouped runtime options."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


__all__ = ("BaseCfg",)
