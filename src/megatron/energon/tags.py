# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from abc import ABCMeta
from typing import Any, Dict, Optional


def resolve_tags(
    tags: Optional[Dict[str, Any]],
    subflavors: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the canonical and legacy names for dataset tags."""
    if tags is not None and subflavors is not None:
        raise ValueError("Cannot set both 'tags' and 'subflavors'.")
    return tags if tags is not None else subflavors


def canonicalize_tag_kwargs(kwargs: dict) -> dict:
    """Return keyword arguments using the canonical sample tag field."""
    if "__subflavors__" not in kwargs:
        return kwargs
    if "__tags__" in kwargs:
        raise ValueError("Cannot set both '__tags__' and '__subflavors__'.")
    kwargs = kwargs.copy()
    kwargs["__tags__"] = kwargs.pop("__subflavors__")
    return kwargs


class _TagAliasMeta(ABCMeta):
    """Translate legacy constructor keywords before generated dataclass initializers run."""

    @property
    def __signature__(cls) -> inspect.Signature:
        """Expose the generated initializer signature instead of this metaclass's ``__call__``."""
        signature = inspect.signature(cls.__init__)
        return signature.replace(parameters=tuple(signature.parameters.values())[1:])

    def __call__(cls, *args, **kwargs):
        for legacy_name, canonical_name in cls.__config_aliases__.items():
            if legacy_name not in kwargs:
                continue
            if canonical_name in kwargs:
                raise ValueError(f"Cannot set both {canonical_name!r} and {legacy_name!r}.")
            kwargs[canonical_name] = kwargs.pop(legacy_name)
        return super().__call__(*args, **kwargs)


class TagsAlias(metaclass=_TagAliasMeta):
    """Compatibility alias for objects exposing ``tags``."""

    __slots__ = ()
    __config_aliases__ = {"subflavors": "tags"}

    @property
    def subflavors(self) -> Optional[Dict[str, Any]]:
        return self.tags

    @subflavors.setter
    def subflavors(self, value: Optional[Dict[str, Any]]) -> None:
        self.tags = value


class SampleTagsAlias(metaclass=_TagAliasMeta):
    """Compatibility alias for samples and batches exposing ``__tags__``."""

    __slots__ = ()
    __config_aliases__ = {"__subflavors__": "__tags__"}

    @property
    def __subflavors__(self):
        return self.__tags__

    @__subflavors__.setter
    def __subflavors__(self, value) -> None:
        self.__tags__ = value


class HasTagsAlias(metaclass=_TagAliasMeta):
    """Compatibility alias for cooker tag selectors."""

    __slots__ = ()
    __config_aliases__ = {"has_subflavors": "has_tags"}

    @property
    def has_subflavors(self) -> Optional[dict]:
        return self.has_tags

    @has_subflavors.setter
    def has_subflavors(self, value: Optional[dict]) -> None:
        self.has_tags = value
