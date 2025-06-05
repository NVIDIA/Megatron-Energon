# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import itertools
import warnings
from functools import wraps
from typing import Any, Type, TypeVar, Union

import numpy as np
import torch


def compact_str(
    value: Union[dict, list, str, int, bool, None],
    depth: int = 3,
    max_items: int = 10,
    max_str_len: int = 50,
) -> str:
    """
    Compact representation of a value as a string.

    Args:
        value: The value to compact
        depth: The maximum depth to compact
        max_items: The maximum number of items to show in a list or dict
        max_str_len: The maximum string length to show

    Returns: The printable string
    """
    if isinstance(value, dict):
        if depth <= 0:
            return "{...}"
        return (
            "{"
            + ", ".join(
                (
                    f"{k}: {v!r}"
                    if isinstance(k, str) and k.startswith("__")
                    else f"{k}: {compact_str(v, depth - 1, max_items, max_str_len)}"
                )
                for k, v in itertools.islice(value.items(), max_items)
            )
            + "}"
        )
    elif isinstance(value, list):
        if depth <= 0:
            return "[...]"
        return (
            "["
            + ", ".join(
                compact_str(v, depth - 1, max_items, max_str_len) for v in value[:max_items]
            )
            + "]"
        )
    elif isinstance(value, tuple):
        if depth <= 0:
            return "(...)"
        return (
            "("
            + ", ".join(
                compact_str(v, depth - 1, max_items, max_str_len) for v in value[:max_items]
            )
            + ")"
        )
    elif isinstance(value, str):
        if len(value) > max_str_len:
            return repr(value[:max_str_len] + "...")
        return repr(value)
    elif isinstance(value, torch.Tensor):
        return f"Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})"
    elif isinstance(value, np.ndarray):
        return f"np.ndarray(shape={value.shape}, dtype={value.dtype})"
    elif dataclasses.is_dataclass(value):
        return f"{value.__class__.__name__}({', '.join(f'{field.name}={compact_str(getattr(value, field.name))}' for field in dataclasses.fields(value))})"
    else:
        return compact_str(repr(value), depth, max_items, max_str_len)


T = TypeVar("T")


class SampleException(ValueError):
    @classmethod
    def from_sample_key(cls: Type[T], sample_key: str) -> T:
        return cls(f"Sample {sample_key} failed")

    @classmethod
    def from_sample(cls: Type[T], sample: Any, message: str = "") -> T:
        if message:
            message = f": {message}"
        return cls(f"Sample {compact_str(sample)} failed{message}")


class FatalSampleError(SampleException):
    # This will not be handled by the error handler
    pass


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


SYSTEM_EXCEPTIONS = (
    SystemError,
    SyntaxError,
    ImportError,
    StopIteration,
    StopAsyncIteration,
    MemoryError,
    RecursionError,
    ReferenceError,
    NameError,
    UnboundLocalError,
    FatalSampleError,
)
