# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar, Union, overload

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.source_info import SourceInfo

T_sample = TypeVar("T_sample", bound=Sample, covariant=True)

F = TypeVar("F", bound=Callable[..., Sample])


@overload
def cooker(
    fn: None = None,
) -> Callable[[F], F]: ...


@overload
def cooker(
    *,
    need_cache: bool = False,
    need_primary: bool = False,
) -> Callable[[F], F]: ...


def cooker(
    fn: Optional[F] = None,
    *,
    need_cache: bool = False,
    need_primary: bool = False,
) -> Union[
    F,
    Callable[[F], F],
]:
    """Decorator to mark a function as a cooker, optionally enabling cache and primary dataset
    arguments."""
    if fn is None:
        return functools.partial(
            cooker,
            need_cache=need_cache,
            need_primary=need_primary,
        )

    @functools.wraps(fn)
    def fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    setattr(fn_wrapper, "__cooker_need_cache__", need_cache)
    setattr(fn_wrapper, "__cooker_need_primary__", need_primary)
    return fn_wrapper


def get_cooker_need_cache(fn: Callable[..., T_sample]) -> bool:
    """Get whether a function is a cooker."""
    return getattr(fn, "__cooker_need_cache__", False)


def get_cooker_need_primary(fn: Callable[..., T_sample]) -> bool:
    """Get whether a function is a cooker."""
    return getattr(fn, "__cooker_need_primary__", False)


@dataclass
class Cooker(Generic[T_sample]):
    """A cooker transforms a crude sample (simple dict) into a specific sample type inheriting
    from `Sample`.
    The `cook` method performs the transformation, the other fields are used to select the
    samples which this cooker can transform. If no filters are provided, the cooker will transform
    any `CrudeSample`.
    """

    #: The callable that performs the cooking (i.e. loading / transforming the crude sample).
    # Signature is:
    # `(/, raw_sample: dict, *, primary?: RandomAccessDataset, **aux: RandomAccessDataset, cache?: Cache) -> Sample`.
    # `primary` is passed only if want_primary_random_access is true.
    # `cache` is passed only if want_cache is true.
    cook: Callable[..., T_sample]

    #: The subflavors to be present in the sample to be cooked by this cooker. All keys and values
    # must match.
    has_subflavors: Optional[dict] = None

    @property
    def need_primary(self) -> bool:
        return get_cooker_need_primary(self.cook)

    @property
    def need_cache(self) -> bool:
        return get_cooker_need_cache(self.cook)

    def is_match(self, crude_sample: CrudeSample) -> bool:
        if self.has_subflavors is not None:
            # Checks if the dict entries provided as a filter all match
            # the ones in the sample. The sample may have additional entries.
            for k, v in self.has_subflavors.items():
                if (
                    k not in crude_sample["__subflavors__"]
                    or crude_sample["__subflavors__"][k] != v
                ):
                    return False

        return True


def basic_sample_keys(
    crude_sample: dict, additional_source_info: tuple[SourceInfo, ...] = ()
) -> dict:
    """A convenience helper to extract the basic keys from a crude sample,
    which you will always need to forward to the cooked sample."""

    res = {
        field.name: crude_sample[field.name]
        for field in dataclasses.fields(Sample)
        if field.name in crude_sample
    }

    if additional_source_info:
        res["__sources__"] = (*crude_sample["__sources__"], *additional_source_info)
    return res
