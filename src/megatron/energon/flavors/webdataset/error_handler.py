# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Callable, Generic, Optional, Tuple, TypeVar, Union

from megatron.energon.errors import SYSTEM_EXCEPTIONS, FatalSampleError
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset.structs import FilteredSample, reraise_exception
from megatron.energon.source_info import SourceInfo

T_sample = TypeVar("T_sample", covariant=True)


class ErrorHandler(Generic[T_sample]):
    handler: Callable[[Exception, Optional[str], Optional[list[SourceInfo]]], None] = (
        reraise_exception
    )

    def sample_error_handler(
        self, e: Exception, sample_key: Optional[str], sources: Optional[list[SourceInfo]] = None
    ):
        if isinstance(e, SYSTEM_EXCEPTIONS):
            raise FatalSampleError(f"Error in sample {sample_key!r}: {e}") from e

        self.handler(e, sample_key, sources)

    def error_handler(
        self,
        e: Exception,
        sample: Union[
            T_sample,
            dict,
            FilteredSample,
            None,
            Tuple[Union[T_sample, dict, FilteredSample, None], ...],
        ],
    ):
        if isinstance(sample, dict):
            key = sample.get("__key__")
            sources = sample.get("__sources__")
        elif isinstance(sample, list):
            if isinstance(sample[0], dict):
                key = ",".join("None" if s is None else s.get("__key__") for s in sample)
                sources = [src for s in sample for src in s.get("__sources__", ())]
            elif isinstance(sample[0], Sample):
                key = ",".join("None" if s is None else s.__key__ for s in sample)
                sources = [src for s in sample for src in s.__sources__]
            else:
                key = None
                sources = None
        elif isinstance(sample, Sample):
            key = sample.__key__
            sources = sample.__sources__
        else:
            key = None
            sources = None
        self.sample_error_handler(e, key, sources)
