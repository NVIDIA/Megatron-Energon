# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass
from typing import List, Union

import torch

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class InterleavedSample(Sample):
    """Sample type for interleaved media such as text with images."""

    #: The interleaved media (either torch.tensor for an image, or str for text)
    sequence: List[Union[torch.Tensor, str]]


class InterleavedWebdataset(DefaultDecoderWebdatasetFactory[InterleavedSample]):
    __sample_type__ = InterleavedSample

    def __init__(self, path: EPath, **kwargs):
        warnings.warn(
            f"{type(self)} is deprecated, use the default instead and set the sample_type:\n"
            f"To convert, update your {path}/.nv-meta/dataset.yaml to:\n"
            f"# remove top-level __module__ and __class__\n"
            f"sample_type:\n"
            f"  __module__: megatron.energon\n"
            f"  __class__: {self.__sample_type__.__name__}\n"
            f"# Keep the remaining content",
            DeprecationWarning,
        )
        super().__init__(path, **kwargs)
