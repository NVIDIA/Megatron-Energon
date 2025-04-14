# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Literal

import webdataset

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.cache import FileStoreDecoder
from megatron.energon.flavors.webdataset.config import split_name_re

ImageDecoderType = Literal[
    "l8",
    "rgb8",
    "rgba8",
    "l",
    "rgb",
    "rgba",
    "torchl8",
    "torchrgb8",
    "torchrgba8",
    "torchl",
    "torchrgb",
    "torch",
    "torchrgba",
    "pill",
    "pil",
    "pilrgb",
    "pilrgba",
]

AVDecoderType = Literal["torch", "AVDecoder", "pyav"]


class SampleDecoder(FileStoreDecoder):
    """The default decoder for webdataset samples."""

    def __init__(
        self,
        *,
        image_decode: ImageDecoderType = "torchrgb",
        av_decode: AVDecoderType = "AVDecoder",
        video_decode_audio: bool = False,
    ):
        """
        Args:
            image_decode: This defines the decoding results.
            av_decode: If "AVDecoder", returns an AVDecoder instance for flexible decoding. If "torch",
                returns decoded VideoData.
            video_decode_audio: Whether to decode audio from video files.
        """
        self._config = dict(
            image_decode=image_decode,
            av_decode=av_decode,
            video_decode_audio=video_decode_audio,
        )
        self._decoder = webdataset.autodecode.Decoder(
            [
                webdataset.autodecode.imagehandler(image_decode),
                AVWebdatasetDecoder(
                    video_decode_audio=video_decode_audio,
                    av_decode=av_decode,
                ),
            ]
        )

    def decode(self, fname: str, raw: bytes) -> Any:
        m = split_name_re.match(fname)
        if not m:
            raise ValueError(f"Invalid file name: {fname}")
        cur_base_name, ext = m.groups()

        return self(
            {
                "__key__": cur_base_name,
                ext: raw,
            }
        )[ext]

    def __call__(self, sample: dict) -> dict:
        return self._decoder(sample)

    def config(self) -> dict:
        return self._config
