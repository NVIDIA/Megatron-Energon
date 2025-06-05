# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, List, Optional, Union

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


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
        subflavors: Optional[Dict[str, Any]] = None,
        part_filter: Union[str, List[str], Callable[[str], bool]] = lambda _: True,
        **kwargs,
    ):
        """
        Constructs a crude webdataset.

        Args:
            path: Root path to the joined datasets.
            subflavors: Subflavors dictionary to set for all loaded samples.
            part_filter: Function for filtering tar files to load by dict keys.
            **kwargs: Additional arguments to the BaseWebdataset constructor.
        """
        # We skip the parent class __init__ and call the BaseWebdataset.__init__ directly

        if "sample_loader" in kwargs:
            raise ValueError("sample_loader is not allowed to be set when using CrudeWebdataset")

        super().__init__(
            path,
            subflavors=subflavors,
            sample_loader=lambda sample: sample,
            part_filter=part_filter,
            **kwargs,
        )
