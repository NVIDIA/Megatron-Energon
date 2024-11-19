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
class SimilarityInterleavedSample(Sample):
    """Sample type for interleaved media such as text with images, but without image-text alignment.
    That alignment has to be assigned from the similarity matrix."""

    #: The images of the sequence
    images: List[torch.Tensor]
    #: The texts of the sequence
    texts: List[str]

    #: Similarity matrix between image and text entries in the sequence
    similarity_matrix: Optional[torch.Tensor] = None

    #: The index within texts representing the sentence that this image is matched to
    matched_text_indices: Optional[List[int]] = None


class SimilarityInterleavedWebdataset(DefaultDecoderWebdatasetFactory[SimilarityInterleavedSample]):
    __sample_type__ = SimilarityInterleavedSample

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
