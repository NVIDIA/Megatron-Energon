# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Utilities for inspecting and debugging samples."""

import dataclasses
import itertools
from typing import Any

import numpy as np
import torch


def default_get_batch_keys(batch: Any) -> list[str] | None:
    """Extract sample keys from a batch using common heuristics.

    This function attempts to extract keys from samples by checking for common
    key attributes/fields in the following order:
    - __key__ or __keys__ attributes
    - __key__ or __keys__ dict keys
    - "keys" dict key

    Args:
        batch: A sample or batch to extract keys from. If a list, uses the first element.

    Returns:
        List of string keys if found, None otherwise.
    """
    if isinstance(batch, list):
        batch = batch[0]
    if (
        hasattr(batch, "__key__")
        and isinstance(batch.__key__, list)
        and all(isinstance(k, str) for k in batch.__key__)
    ):
        return batch.__key__
    elif (
        hasattr(batch, "__keys__")
        and isinstance(batch.__keys__, list)
        and all(isinstance(k, str) for k in batch.__keys__)
    ):
        return batch.__keys__
    elif (
        isinstance(batch, dict)
        and "__key__" in batch
        and all(isinstance(k, str) for k in batch["__key__"])
    ):
        return batch["__key__"]
    elif (
        isinstance(batch, dict)
        and "__keys__" in batch
        and all(isinstance(k, str) for k in batch["__keys__"])
    ):
        return batch["__keys__"]
    elif (
        isinstance(batch, dict)
        and "keys" in batch
        and all(isinstance(k, str) for k in batch["keys"])
    ):
        return batch["keys"]
    return None


def format_sample_compact(
    sample: Any,
    depth: int = 3,
    max_items: int = 10,
    max_str_len: int = 50,
) -> str:
    """Create a compact single-line string representation of a sample.

    Designed for inline use in error messages and logs. For detailed
    multi-line debugging output, use format_sample_detailed().

    Args:
        sample: The sample to represent as a string.
        depth: Maximum nesting depth to show before truncation.
        max_items: Maximum number of items to show in collections.
        max_str_len: Maximum string length before truncation.

    Returns:
        A compact single-line string representation.

    Example:
        >>> format_sample_compact({"key": "value", "count": 42})
        "{'key': 'value', 'count': 42}"
    """
    if isinstance(sample, dict):
        if depth <= 0:
            return "{...}"
        return (
            "{"
            + ", ".join(
                (
                    f"{k}: {v!r}"
                    if isinstance(k, str) and k.startswith("__")
                    else f"{k}: {format_sample_compact(v, depth - 1, max_items, max_str_len)}"
                )
                for k, v in itertools.islice(sample.items(), max_items)
            )
            + "}"
        )
    elif isinstance(sample, list):
        if depth <= 0:
            return "[...]"
        return (
            "["
            + ", ".join(
                format_sample_compact(v, depth - 1, max_items, max_str_len)
                for v in sample[:max_items]
            )
            + "]"
        )
    elif isinstance(sample, tuple):
        if depth <= 0:
            return "(...)"
        return (
            "("
            + ", ".join(
                format_sample_compact(v, depth - 1, max_items, max_str_len)
                for v in sample[:max_items]
            )
            + ")"
        )
    elif isinstance(sample, str):
        if len(sample) > max_str_len:
            return repr(sample[:max_str_len] + "...")
        return repr(sample)
    elif isinstance(sample, torch.Tensor):
        return f"Tensor(shape={sample.shape}, dtype={sample.dtype}, device={sample.device})"
    elif isinstance(sample, np.ndarray):
        return f"np.ndarray(shape={sample.shape}, dtype={sample.dtype})"
    elif dataclasses.is_dataclass(sample):
        return f"{type(sample).__name__}({', '.join(f'{field.name}={format_sample_compact(getattr(sample, field.name), depth, max_items, max_str_len)}' for field in dataclasses.fields(sample))})"
    else:
        repr_str = repr(sample)
        return (
            format_sample_compact(repr_str, depth, max_items, max_str_len)
            if not isinstance(sample, str)
            else repr_str
        )


def format_sample_detailed(sample: Any, indent: str = "") -> str:
    """Create a detailed multi-line string representation of a sample for debugging.

    Produces human-readable representations with proper indentation and detailed
    information about tensors, arrays, and nested structures. For compact inline
    representations in error messages, use format_sample_compact().

    Args:
        sample: The sample to represent as a string.
        indent: Current indentation level (used internally for recursion).

    Returns:
        A formatted multi-line string representation with detailed information.

    Example:
        >>> print(format_sample_detailed({"image": torch.zeros(3, 224, 224), "label": 5}))
         - image: Tensor(shape=(3, 224, 224), dtype=torch.float32, ...)
         - label: 5
    """
    if isinstance(sample, dict):
        result = []
        for _, (key, value) in zip(range(25), sample.items()):
            result.append(f"{indent} - {key}: {format_sample_detailed(value, indent + '  ')}")
        if len(sample) > 25:
            result.append(f"{indent} - ... (and {len(sample) - 25} more items)")
        return "\n".join(result)
    elif isinstance(sample, str):
        if len(sample) > 1000:
            sample = f"{sample[:1000]}... (and {len(sample) - 1000} more characters)"
        if "\n" in sample:
            # represent as """ string if it contains newlines:
            return '"""' + sample.replace("\n", "\n   " + indent) + '"""'
        return repr(sample)
    elif isinstance(sample, (int, float, bool, type(None))):
        return repr(sample)
    elif isinstance(sample, (list, tuple)):
        if all(isinstance(value, (str, int, float, bool, type(None))) for value in sample):
            return f"[{', '.join(repr(value) for value in sample)}]"
        result = []
        for _, value in zip(range(10), sample):
            result.append(f"{indent} - {format_sample_detailed(value, indent + '   ')}")
        if len(sample) > 10:
            result.append(f"{indent} - ... (and {len(sample) - 10} more items)")
        return "\n".join(result)
    elif isinstance(sample, torch.Tensor):
        try:
            min_val = sample.min().item()
            max_val = sample.max().item()
            values_repr = ""
            # flatten tensor, get first and last 3 values if possible
            numel = sample.numel()
            flat = sample.flatten()
            n_show = 3
            if numel == 0:
                values_repr = "values=[]"
            elif numel <= n_show * 2:
                shown = ", ".join(repr(v.item()) for v in flat)
                values_repr = f"values=[{shown}]"
            else:
                first_vals = ", ".join(repr(v.item()) for v in flat[:n_show])
                last_vals = ", ".join(repr(v.item()) for v in flat[-n_show:])
                values_repr = f"values=[{first_vals}, ..., {last_vals}]"
            return (
                f"Tensor(shape={sample.shape}, dtype={sample.dtype}, device={sample.device}, "
                f"min={min_val}, max={max_val}, {values_repr})"
            )
        except (RuntimeError, ValueError):
            # Handle empty tensors or non-numeric dtypes
            return f"Tensor(shape={sample.shape}, dtype={sample.dtype}, device={sample.device})"
    elif isinstance(sample, np.ndarray):
        try:
            min_val = sample.min()
            max_val = sample.max()
            values_repr = ""
            flat = sample.ravel()
            n_show = 3
            numel = flat.size
            if numel == 0:
                values_repr = "values=[]"
            elif numel <= n_show * 2:
                shown = ", ".join(repr(x) for x in flat)
                values_repr = f"values=[{shown}]"
            else:
                first_vals = ", ".join(repr(x) for x in flat[:n_show])
                last_vals = ", ".join(repr(x) for x in flat[-n_show:])
                values_repr = f"values=[{first_vals}, ..., {last_vals}]"
            return (
                f"np.ndarray(shape={sample.shape}, dtype={sample.dtype}, "
                f"min={min_val}, max={max_val}, {values_repr})"
            )
        except (ValueError, TypeError):
            # Handle empty arrays or non-numeric dtypes
            return f"np.ndarray(shape={sample.shape}, dtype={sample.dtype})"
    elif dataclasses.is_dataclass(sample):
        result = [f"{indent}{type(sample).__name__}("]
        for field in dataclasses.fields(sample):
            result.append(
                f"{indent}  {field.name}={format_sample_detailed(getattr(sample, field.name), indent + '  ')}"
            )
        result.append(f"{indent})")
        return "\n".join(result)
    else:
        repr_str = repr(sample)
        return repr_str[:200] + "..." if len(repr_str) > 200 else repr_str
