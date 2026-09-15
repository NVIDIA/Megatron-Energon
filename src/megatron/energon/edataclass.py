# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from typing_extensions import dataclass_transform


# We define an alias for `@dataclass(slots=True, kw_only=True)`,
# because we want to use this combination almost everywhere.
# dataclass_transform has no standardized slots_default (PEP 681); slots is runtime-only.
@dataclass_transform(kw_only_default=True)
def edataclass(cls):
    """
    A decorator equivalent to `@dataclass(slots=True, kw_only=True)`.

    If you need more options, use `dataclass` directly.
    E.g.: `@dataclass(slots=True, kw_only=True, eq=False)`.
    """
    return dataclass(kw_only=True, slots=True)(cls)
