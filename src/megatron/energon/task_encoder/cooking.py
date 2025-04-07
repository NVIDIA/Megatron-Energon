# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, TypeVar

from megatron.energon.flavors.base_dataset import RandomAccessDataset, Sample
from megatron.energon.flavors.crude import CrudeSample
from megatron.energon.flavors.webdataset.sample_decoder import (
    AVDecoder,
    ImageDecoder,
    SampleDecoder,
)

T_sample = TypeVar("T_sample", bound=Sample, covariant=True)


class CookerCallable(Protocol[T_sample]):
    """A protocol for the callable that performs the cooking."""

    def __call__(self, raw_sample: CrudeSample, **aux: RandomAccessDataset) -> T_sample:
        """
        Cooks the sample.

        Args:
            raw_sample: The crude sample to cook.
            **aux: The auxiliary side dishes to use for cooking.

        Returns:
            The cooked sample.
        """
        ...


@dataclass
class Cooker(Generic[T_sample]):
    """A cooker transforms a crude sample (simple dict) into a specific sample type inheriting
    from `Sample`.
    The `cook` method performs the transformation, the other fields are used to select the
    samples which this cooker can transform. If no filters are provided, the cooker will transform
    any `CrudeSample`.
    """

    #: The callable that performs the cooking (i.e. loading / transforming the crude sample).
    # Signature is: (raw_sample: dict, **aux: RandomAccessDataset) -> Sample
    cook: CookerCallable[T_sample]

    #: (Deprecated) The subflavor to check for a sample to be cooked by this cooker.
    # Use `has_subflavors` instead.
    # If combined with `has_subflavors` or `condition`, all must be satisfied.
    is_subflavor: Optional[str] = None
    #: The subflavors to be present in the sample to be cooked by this cooker. All keys and values
    # must match.
    # If combined with `is_subflavor` or `condition`, all must be satisfied.
    has_subflavors: Optional[dict] = None
    #: The custom condition on the raw sample to check if the sample should be cooked by this
    # cooker.
    # If combined with `is_subflavor` or `has_subflavors`, all must be satisfied.
    condition: Optional[Callable[[CrudeSample], bool]] = None

    #: If true, the auxiliary loaders are decoded by default.
    auto_decode: bool = True
    #: The image decoder to use for auxiliary loaders.
    image_decode: ImageDecoder = "torchrgb"
    #: The AV decoder to use for auxiliary loaders.
    av_decode: AVDecoder = "AVDecoder"
    #: Whether to decode audio from video files for auxiliary loaders.
    video_decode_audio: bool = False

    #: Internal: The decoder to use for auxiliary loaders.
    _decoder: Optional[SampleDecoder] = None

    def __post_init__(self):
        if self.auto_decode:
            self._decoder = SampleDecoder(
                image_decode=self.image_decode,
                av_decode=self.av_decode,
                video_decode_audio=self.video_decode_audio,
            )

    def is_match(self, crude_sample: CrudeSample) -> bool:
        if self.is_subflavor is not None:
            if crude_sample["__subflavor__"] != self.is_subflavor:
                return False

        if self.has_subflavors is not None:
            # Checks if the dict entries provided as a filter all match
            # the ones in the sample. The sample may have additional entries.
            for k, v in self.has_subflavors.items():
                if (
                    k not in crude_sample["__subflavors__"]
                    or crude_sample["__subflavors__"][k] != v
                ):
                    return False

        if self.condition is not None:
            if not self.condition(crude_sample):
                return False

        return True


def basic_sample_keys(crude_sample: dict) -> dict:
    """A convenience helper to extract the basic keys from a crude sample,
    which you will always need to forward to the cooked sample."""

    return {
        field.name: crude_sample[field.name]
        for field in dataclasses.fields(Sample)
        if field.name in crude_sample
    }
