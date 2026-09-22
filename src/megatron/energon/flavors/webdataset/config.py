# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re

from megatron.energon.deprecated import DEPRECATED_ATTRS, deprecated_dir, deprecated_getattr

split_name_re = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")
skip_meta_re = re.compile(r"__[^/]*__($|/)")

__all__ = ["skip_meta_re", "split_name_re", *DEPRECATED_ATTRS[__name__]]


def __getattr__(name: str):
    return deprecated_getattr(__name__, name)


def __dir__():
    return deprecated_dir(__name__, ["skip_meta_re", "split_name_re"])
