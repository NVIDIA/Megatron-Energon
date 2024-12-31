# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Literal, Optional, TypeVar

import re
import webdataset

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.decode_av_frames import decode_video_frames, decode_audio_samples
from megatron.energon.flavors.webdataset.default_generic_webdataset import (
    DefaultGenericWebdatasetFactory,
)
from megatron.energon.flavors.webdataset.structs import FilteredSample
from megatron.energon.flavors.webdataset.video_data import VideoData

T_sample = TypeVar("T_sample", covariant=True)

ImageDecoder = Literal[
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


class DefaultDecoderWebdatasetFactory(DefaultGenericWebdatasetFactory[T_sample], Generic[T_sample]):
    """
    Extends the default webdataset loading with decoding of contained files, such as images, videos or nested
    containers.
    """

    #: Image decoding result type
    image_decode: ImageDecoder
    #: If true, ignore errors when decoding.
    ignore_decoder_errors: bool

    # The webdataset decoder function, if to be applied
    _decoder: Optional[Callable[[FilteredSample], FilteredSample]]

    def __init__(
        self,
        path: EPath,
        *,
        auto_decode: bool = True,
        image_decode: ImageDecoder = "torchrgb",
        ignore_decoder_errors: bool = False,
        **kwargs,
    ):
        """
        Factory for the webdataset sample loader including the decoder.

        Args:
            path: Path to the dataset (passed to parent)
            auto_decode: If true, use the default webdataset sample decoder.
            image_decode: This defines the decoding results.
            ignore_decoder_errors: If true, ignore errors when decoding.
            **kwargs: Args passed to parent constructor
        """
        self.image_decode = image_decode
        self.ignore_decoder_errors = ignore_decoder_errors
        super().__init__(path, **kwargs)

        if auto_decode:
            self._decoder = webdataset.autodecode.Decoder(
                [
                    webdataset.autodecode.imagehandler(self.image_decode),
                    self._video_decoder,
                ]
            )
        else:
            self._decoder = None

    def _decode_error_handler(self, exc: Exception) -> bool:
        if self.ignore_decoder_errors:
            return True
        raise exc

    def _video_decoder(self, key, data):
        """
        Extract the video or audio data from default media extensions.

        Args:
            key: media file extension
            data: raw media bytes
        """
        extension = re.sub(r".*[.]", "", key)
        # TODO(jbarker): we should add a debug log here
        if extension in "mov mp4 webm mkv".split():
            # TODO(jbarker): make the magic numbers configurable
            media = decode_video_frames(
                data,
                num_frames=64,
                out_frame_size=(224, 224),
                decode_audio=False
            )
        elif extension in "flac mp3".split():
            # TODO(jbarker): make the magic numbers configurable
            media = decode_audio_samples(
                data,
                num_clips=5,
                clip_duration=1,
            )
        else:
            return None
        if media is not None:
            return VideoData(
                frames=media[0].permute((0, 3, 1, 2)),
                aframes=media[1],
                info=media[2],
            )
        return None

    def load_sample(self, sample: FilteredSample) -> T_sample:
        if self._decoder is not None:
            sample = self._decoder(sample)
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            image_decode=self.image_decode,
            ignore_decoder_errors=self.ignore_decoder_errors,
        )
