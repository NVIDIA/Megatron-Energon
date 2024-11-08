# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, List, Optional, Union

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory, ImageDecoder


class CrudeSample(dict):
    """Generic sample type to be processed later."""


class CrudeWebdataset(DefaultDecoderWebdatasetFactory[CrudeSample]):
    """The CrudeWebdataset is used to load crude / raw samples and
    decode them in the user code using so-called cookers.

    See the documentation under "Crude Data" for more information.
    """

    __sample_type__ = CrudeSample

    def __init__(
        self,
        path: EPath,
        *,
        subflavor: Optional[str] = None,
        subflavors: Optional[Dict[str, Any]] = None,
        part_filter: Union[str, List[str], Callable[[str], bool]] = lambda _: True,
        auto_decode: bool = True,
        image_decode: ImageDecoder = "torchrgb",
        ignore_decoder_errors: bool = False,
        **kwargs,
    ):
        """
        Constructs a crude webdataset.

        Args:
            path: Root path to the joined datasets.
            subflavor: Deprecated. Subflavor to set for all loaded samples.
            subflavors: Subflavors dictionary to set for all loaded samples.
            part_filter: Function for filtering tar files to load by dict keys.
            auto_decode: Whether to decode the samples using webdataset decode or not.
            image_decode: Image decoding method to use. Only applies if `decode=True`.
            ignore_decoder_errors: Whether to ignore decoding errors or not.
            **kwargs: Additional arguments to the BaseWebdataset constructor.
        """
        # We skip the parent class __init__ and call the BaseWebdataset.__init__ directly
        super().__init__(
            path,
            auto_decode=auto_decode,
            image_decode=image_decode,
            ignore_decoder_errors=ignore_decoder_errors,
            subflavor=subflavor,
            subflavors=subflavors,
            sample_loader=lambda sample: sample,
            part_filter=part_filter,
            **kwargs,
        )
