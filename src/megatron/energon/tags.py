# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from abc import ABCMeta
from typing import Any, Dict, Optional

from megatron.energon.deprecation import warn_deprecated


def _warn_legacy_name(legacy_name: str, canonical_name: str, stacklevel: int = 1) -> None:
    """Emit the deprecation warning for a legacy tag name.

    Args:
        legacy_name: The deprecated name that was used.
        canonical_name: The name to use instead.
        stacklevel: Like :func:`warnings.warn`, relative to the caller of this function.
    """
    warn_deprecated(
        f"{legacy_name!r} is deprecated and will be removed in a future release, "
        f"use {canonical_name!r} instead.",
        stacklevel=stacklevel + 2,
    )


def resolve_tags(
    tags: Optional[Dict[str, Any]],
    subflavors: Optional[Dict[str, Any]],
    *,
    stacklevel: int = 2,
) -> Optional[Dict[str, Any]]:
    """Resolve the canonical and legacy names for dataset tags."""
    if subflavors is None:
        return tags
    if tags is not None:
        raise ValueError("Cannot set both 'tags' and 'subflavors'.")
    _warn_legacy_name("subflavors", "tags", stacklevel + 1)
    return subflavors


def canonicalize_tag_kwargs(kwargs: dict, *, stacklevel: int = 2) -> dict:
    """Return keyword arguments using the canonical sample tag field."""
    if "__subflavors__" not in kwargs:
        return kwargs
    if "__tags__" in kwargs:
        raise ValueError("Cannot set both '__tags__' and '__subflavors__'.")
    _warn_legacy_name("__subflavors__", "__tags__", stacklevel + 1)
    kwargs = kwargs.copy()
    kwargs["__tags__"] = kwargs.pop("__subflavors__")
    return kwargs


def apply_config_aliases(cls: Any, kwargs: Any, *, stacklevel: int = 2) -> Any:
    """Rename legacy keyword arguments of ``cls`` (declared in ``__config_aliases__``) to their
    canonical names.

    Args:
        cls: The class to be constructed.
        kwargs: The constructor keyword arguments or raw config. Non-dicts are returned as is.
        stacklevel: Like :func:`warnings.warn`, relative to the caller of this function.

    Returns:
        The keyword arguments with canonical names. The input is not modified.
    """
    if not isinstance(kwargs, dict):
        return kwargs
    aliases = getattr(cls, "__config_aliases__", None)
    if not aliases:
        return kwargs
    for legacy_name, canonical_name in aliases.items():
        if legacy_name not in kwargs:
            continue
        if canonical_name in kwargs:
            raise ValueError(f"Cannot set both {canonical_name!r} and {legacy_name!r}.")
        _warn_legacy_name(legacy_name, canonical_name, stacklevel + 1)
        kwargs = kwargs.copy()
        kwargs[canonical_name] = kwargs.pop(legacy_name)
    return kwargs


class _TagAliasMeta(ABCMeta):
    """Translate legacy constructor keywords before generated dataclass initializers run."""

    @property
    def __signature__(cls) -> inspect.Signature:
        """Expose the generated initializer signature instead of this metaclass's ``__call__``."""
        signature = inspect.signature(cls.__init__)
        return signature.replace(parameters=tuple(signature.parameters.values())[1:])

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **apply_config_aliases(cls, kwargs))


class TagsAlias(metaclass=_TagAliasMeta):
    """Compatibility alias for objects exposing ``tags``."""

    __slots__ = ()
    __config_aliases__ = {"subflavors": "tags"}

    @property
    def subflavors(self) -> Optional[Dict[str, Any]]:
        _warn_legacy_name("subflavors", "tags", 2)
        return self.tags

    @subflavors.setter
    def subflavors(self, value: Optional[Dict[str, Any]]) -> None:
        _warn_legacy_name("subflavors", "tags", 2)
        self.tags = value


class SampleTagsAlias(metaclass=_TagAliasMeta):
    """Compatibility alias for samples and batches exposing ``__tags__``."""

    __slots__ = ()
    __config_aliases__ = {"__subflavors__": "__tags__"}

    @property
    def __subflavors__(self):
        _warn_legacy_name("__subflavors__", "__tags__", 2)
        return self.__tags__

    @__subflavors__.setter
    def __subflavors__(self, value) -> None:
        _warn_legacy_name("__subflavors__", "__tags__", 2)
        self.__tags__ = value


class HasTagsAlias(metaclass=_TagAliasMeta):
    """Compatibility alias for cooker tag selectors."""

    __slots__ = ()
    __config_aliases__ = {"has_subflavors": "has_tags"}

    @property
    def has_subflavors(self) -> Optional[dict]:
        _warn_legacy_name("has_subflavors", "has_tags", 2)
        return self.has_tags

    @has_subflavors.setter
    def has_subflavors(self, value: Optional[dict]) -> None:
        _warn_legacy_name("has_subflavors", "has_tags", 2)
        self.has_tags = value
