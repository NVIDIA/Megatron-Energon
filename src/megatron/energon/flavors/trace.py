# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypeVar, Union

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.tracing import AsyncContext

T_dataset = TypeVar("T_dataset", bound=SavableDataset)
T_call = TypeVar("T_call", bound=Callable)
T_sample = TypeVar("T_sample")


def _flatten_str_list(keys: Any) -> Iterator[Optional[str]]:
    """Flatten a list of keys into a list of strings."""
    if isinstance(keys, str):
        yield keys
    elif isinstance(keys, (list, tuple)):
        for key in keys:
            yield from _flatten_str_list(key)
    else:
        yield None


def _flatten_str_list_or_none(keys: Any) -> Optional[List[str]]:
    """Flatten a list of keys into a list of strings. If this cannot be fetched, return None."""
    keys = list(_flatten_str_list(keys))
    if any(k is None for k in keys):
        return None
    return keys


def default_get_keys(batch: Any) -> Optional[List[str]]:
    """Default get_keys, which has some heuristics to find the sample keys."""
    if isinstance(batch, list):
        all_keys = []
        for b in batch:
            k = default_get_keys(b)
            if k is None:
                return None
            all_keys.extend(k)
        return all_keys
    if hasattr(batch, "__key__"):
        return _flatten_str_list_or_none(batch.__key__)
    elif hasattr(batch, "__keys__"):
        return _flatten_str_list_or_none(batch.__keys__)
    elif isinstance(batch, dict):
        if "__key__" in batch:
            return _flatten_str_list_or_none(batch["__key__"])
        elif "__keys__" in batch:
            return _flatten_str_list_or_none(batch["__keys__"])
        elif "keys" in batch:
            return _flatten_str_list_or_none(batch["keys"])
    return None


class TraceIter:
    last_args: Dict[str, Any] = {}

    def __init__(
        self,
        outer_self: T_dataset,
        name: str,
        trace_span: AsyncContext,
        call_args: Dict[str, Union[str, Callable[[T_dataset], Any]]],
    ):
        self.outer_self = outer_self
        self.name = name
        self.trace_span = trace_span
        self.call_args = call_args

    def sample_exception(
        self, exception: Exception, samples: Union[T_sample, Sequence[T_sample]]
    ) -> None:
        self.trace_span.instant(
            f"{self.name}.error/skip",
            args={
                "exception": f"{type(exception).__name__}: {str(exception)}",
                "sample_keys": default_get_keys(samples),
                **{
                    arg_name: arg_value(self.outer_self) if callable(arg_value) else arg_value
                    for arg_name, arg_value in self.call_args.items()
                },
            },
            level=1,
        )

    def skip_sample(self, samples: Sequence[T_sample]) -> None:
        self.trace_span.instant(
            f"{self.name}.skip",
            args={
                "sample_keys": default_get_keys(samples),
            },
            level=1,
        )

    def sample(
        self, sample: Union[T_sample, Sequence[T_sample]], args: Dict[str, Any] = {}
    ) -> None:
        self.last_args["sample_keys"] = default_get_keys(sample)
        self.last_args.update(args)

    def wrap_fn(self, fn: T_call) -> T_call:
        fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", "<unknown>"))

        @functools.wraps(fn)
        def wrapped_fn(*args, **kwargs):
            with self.trace_span.span(
                f"{self.name}.{fn_name}.call",
                args={
                    arg_name: arg_value(self.outer_self) if callable(arg_value) else arg_value
                    for arg_name, arg_value in self.call_args.items()
                },
                level=2,
            ):
                return fn(*args, **kwargs)

        return wrapped_fn

    def wrap_inner(self, call_args: Callable[..., Dict[str, Any]] = lambda *args, **kwargs: {}):
        def decorator(fn):
            fn_name = getattr(fn, "__qualname__", getattr(fn, "__name__", "<unknown>"))

            @functools.wraps(fn)
            def wrapped_inner_gen(*args, **kwargs):
                with self.trace_span.span(
                    f"{self.name}.{fn_name}.__iter__", args=call_args(*args, **kwargs), level=2
                ):
                    return fn(*args, **kwargs)

            return wrapped_inner_gen

        return decorator


def trace_iter(
    name: Callable[[T_dataset], str] = lambda ds: type(ds).__name__,
    call_args: Dict[str, Union[str, Callable[[T_dataset], Any]]] = {},
    next_args: Dict[str, Union[str, Callable[[T_dataset], Any]]] = {},
) -> Callable[
    [Callable[[T_dataset, TraceIter], Iterator[T_sample]]],
    Callable[[T_dataset], Iterator[T_sample]],
]:
    """Decorator for SavableDataset.__iter__ to trace the iteration using the worker config."""

    def decorator(
        iter_fn: Callable[[T_dataset, TraceIter], Iterator[T_sample]],
    ) -> Callable[[T_dataset], Iterator[T_sample]]:
        @functools.wraps(iter_fn)
        def wrapper(self: T_dataset) -> Iterator[T_sample]:
            trace_span = self.worker_config.worker_trace_span()
            span_name = name(self)
            trace_iter = TraceIter(self, span_name, trace_span, call_args)
            with (
                trace_span.span(
                    f"{span_name}.__iter__",
                    args={
                        arg_name: arg_value(self) if callable(arg_value) else arg_value
                        for arg_name, arg_value in call_args.items()
                    },
                    level=1,
                ),
                self.worker_config.worker_trace_writer().generator(
                    f"{span_name}.__iter__.next",
                    level=2,
                ) as trace_gen,
            ):
                for sample in trace_span.iterable(
                    iter_fn(self, trace_iter),
                    name=f"{span_name}.__iter__.loop",
                    level=2,
                ):
                    with trace_gen.yield_(
                        last_args={
                            **{
                                arg_name: arg_value(self) if callable(arg_value) else arg_value
                                for arg_name, arg_value in next_args.items()
                            },
                            **trace_iter.last_args,
                        },
                    ):
                        trace_iter.last_args.clear()
                        yield sample

        return wrapper

    return decorator
