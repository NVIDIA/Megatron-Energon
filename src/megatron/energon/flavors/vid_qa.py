# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List, Optional

import torch

from megatron.energon.av import AVDecoder
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib.epath import EPath
from megatron.energon.errors import warn_deprecated
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@edataclass
class VidQASample(Sample):
    """Sample type for video question answering."""

    #: The video data containing the image and audio info.
    video: AVDecoder
    #: The context/question for the image.
    context: str
    #: The possible answers. Not set for testing.
    answers: Optional[List[str]] = None
    #: The weights of the possible answers. Optionally available.
    answer_weights: Optional[torch.Tensor] = None


class VidQAWebdataset(DefaultDecoderWebdatasetFactory[VidQASample]):
    __sample_type__ = VidQASample

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
