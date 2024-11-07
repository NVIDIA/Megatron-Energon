# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass

from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset import DefaultDecoderWebdatasetFactory


@dataclass
class TextSample(Sample):
    """Sample type for simple text."""

    #: The text of the sample
    text: str


class TextWebdataset(DefaultDecoderWebdatasetFactory[TextSample]):
    __sample_type__ = TextSample
