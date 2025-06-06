# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional, Union

import torch

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib.epath import EPath
from megatron.energon.errors import warn_deprecated
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@edataclass
class OCRSample(Sample):
    """Sample type for optical character recognition."""

    #: The input image tensor in the shape (C, H, W)
    image: torch.Tensor
    #: The text contained in the image
    text: str
    #: The bounding boxes of the blocks in the image float(N, 4|5<x, y, w, h>)
    block_boxes: Optional[torch.Tensor] = None
    #: The classes of the blocks in the image int(N, 1<block_class>)
    block_classes: Optional[Union[torch.Tensor, List[str]]] = None
    #: The text contained in each block (N,)
    block_text: Optional[List[str]] = None
    #: The bounding boxes of the lines in the image float(N, 4|5<x, y, w, h[, confidence]>)
    lines_boxes: Optional[torch.Tensor] = None
    #: The text contained in each line (N,)
    lines_text: Optional[List[str]] = None
    #: The bounding boxes of the words in the image float(N, 4|5<x, y, w, h[, confidence]>)
    words_boxes: Optional[torch.Tensor] = None
    #: The text contained in each word (N,)
    words_text: Optional[List[str]] = None
    #: The bounding boxes of the chars in the image float(N, 4|5<x, y, w, h[, confidence]>)
    chars_boxes: Optional[torch.Tensor] = None
    #: The character contained in each char (N,)
    chars_text: Optional[List[str]] = None


class OCRWebdataset(DefaultDecoderWebdatasetFactory[OCRSample]):
    __sample_type__ = OCRSample

    def __init__(self, path: EPath, **kwargs):
        warn_deprecated(
            f"{type(self)} is deprecated, use the default instead and set the sample_type:\n"
            f"To convert, update your {path}/.nv-meta/dataset.yaml to:\n"
            f"# remove top-level __module__ and __class__\n"
            f"sample_type:\n"
            f"  __module__: megatron.energon\n"
            f"  __class__: {self.__sample_type__.__name__}\n"
            f"# Keep the remaining content"
        )
        super().__init__(path, **kwargs)
