# Copyright (c) 2025, NVIDIA CORPORATION and Alibaba PAI. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory
from megatron.energon.flavors.webdataset import VideoData

@dataclass
class GeneralQASample(Sample):
    """Sample type for general multi-turn visual question answering."""

    #: The list of input image tensors in the shape (C, H, W)
    images: List[torch.Tensor]
    #: The list of input videos
    videos: List[VideoData]
    #: The context/question for the image
    context: str

    #: The possible answers. Not set for testing.
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available.
    answer_weights: Optional[torch.Tensor] = None

class GeneralQAWebdataset(DefaultDecoderWebdatasetFactory[GeneralQASample]):
    __sample_type__ = GeneralQASample

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
