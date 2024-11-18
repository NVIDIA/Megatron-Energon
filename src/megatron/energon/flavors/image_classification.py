# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class ImageClassificationSample(Sample):
    """Sample type for classifying an image."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The class label of the image
    label: Optional[int] = None
    #: The class label of the image
    label_name: Optional[str] = None


class ImageClassificationWebdataset(DefaultDecoderWebdatasetFactory[ImageClassificationSample]):
    __sample_type__ = ImageClassificationSample

    def __init__(self, path: EPath, **kwargs):
        warnings.warn(
            f"{type(self)} is deprecated, use the default instead and set the sample_type:\n"
            f"To convert, update your {path}/.nv-meta/dataset.yaml to:\n"
            f"# remove top-level __module__ and __class__\n"
            f"sample_type:\n"
            f"  __module__: {self.__sample_type__.__module__}\n"
            f"  __class__: {self.__sample_type__.__name__}\n"
            f"# Keep the remaining content",
            DeprecationWarning,
        )
        super().__init__(path, **kwargs)
