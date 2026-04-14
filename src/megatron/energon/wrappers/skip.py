# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings


class SkipSample(Exception):
    """DEPRECATED, USE `megatron.energon.SkipSample` INSTEAD.
    Exception to raise in the map_fn to skip a sample."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SkipSample is deprecated and will be removed in a future version. Use megatron.energon.SkipSample instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
