# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


class FlexState(dict):
    def __reduce__(self):
        # dict subclasses normally pickle via SETITEMS, which torch's weights_only
        # unpickler only allows for exact dict/OrderedDict/Counter types (not subclasses).
        # Route through a plain dict argument so the round trip works under
        # torch.load(..., weights_only=True) once FlexState is a registered safe global.
        return (FlexState, (dict(self),))
