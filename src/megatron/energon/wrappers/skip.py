# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.deprecation import warn_deprecated
from megatron.energon.errors import SkipSample as _SkipSample


class SkipSample(_SkipSample):
    """Deprecated alias for :class:`megatron.energon.SkipSample`."""

    def __init__(self, *args, **kwargs):
        warn_deprecated(
            "megatron.energon.wrappers.SkipSample is deprecated and will be removed in a future "
            "version. Use megatron.energon.SkipSample instead.",
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
