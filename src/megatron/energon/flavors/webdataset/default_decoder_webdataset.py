# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Optional, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.default_generic_webdataset import (
    DefaultGenericWebdatasetFactory,
)
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
from megatron.energon.flavors.webdataset.structs import FilteredSample

T_sample = TypeVar("T_sample", covariant=True)


class DefaultDecoderWebdatasetFactory(DefaultGenericWebdatasetFactory[T_sample], Generic[T_sample]):
    """
    Extends the default webdataset loading with decoding of contained files, such as images, videos or nested
    containers.
    """

    # The webdataset decoder function, if to be applied
    _decoder: Optional[SampleDecoder]

    def __init__(
        self,
        path: EPath,
        *,
        decoder: Optional[SampleDecoder] = SampleDecoder(),
        **kwargs,
    ):
        """
        Factory for the webdataset sample loader including the decoder.

        Args:
            path: Path to the dataset (passed to parent)
            decoder: If provided, use this decoder, otherwise just load raw bytes.
            **kwargs: Args passed to parent constructor
        """
        self._decoder = decoder
        super().__init__(path, **kwargs)

    def load_sample(self, sample: FilteredSample) -> T_sample:
        if self._decoder is not None:
            sample = self._decoder(sample)
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            **(self._decoder.config() if self._decoder is not None else {}),
        )
