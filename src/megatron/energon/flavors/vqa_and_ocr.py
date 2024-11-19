# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch

from megatron.energon.epathlib.epath import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class VQAOCRSample(Sample):
    """Sample type for visual question answering."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor

    #: The context/question for the image (VQA)
    context: str
    #: The text contained in the image (OCR)
    text: str

    #: The possible answers. Not set for testing. (VQA)
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available. (VQA)
    answer_weights: Optional[torch.Tensor] = None
    #: The bounding boxes of the words in the image (N, 4|5<x, y, w, h[, confidence]>) (OCR)
    words_boxes: Optional[torch.Tensor] = None
    #: The text contained in each word (N,) (OCR)
    words_text: Optional[List[str]] = None


class VQAOCRWebdataset(DefaultDecoderWebdatasetFactory[VQAOCRSample]):
    __sample_type__ = VQAOCRSample

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
