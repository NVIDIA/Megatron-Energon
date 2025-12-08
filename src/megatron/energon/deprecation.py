# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from functools import wraps


def warn_deprecated(reason, stacklevel=2):
    warnings.warn(reason, FutureWarning, stacklevel=stacklevel)


def deprecated(reason):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warn_deprecated(f"{func.__name__} is deprecated: {reason}", stacklevel=3)
            return func(*args, **kwargs)

        return wrapper

    return decorator
