# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Literal, Optional, TypeVar

import webdataset

from megatron.energon.av import AVDecoder
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.default_generic_webdataset import (
    DefaultGenericWebdatasetFactory,
)
from megatron.energon.flavors.webdataset.structs import FilteredSample

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
    #: If "AVData", returns an AVData instance for flexible decoding. If "torch",
    #: returns decoded VideoData.
    video_decode: Literal["torch", "AVData"]
    #: Whether to decode audio from video files.
    video_decode_audio: bool
    #: Number of video frames to extract.
    video_num_frames: int
    #: Output size for video frames (width, height).
    video_out_frame_size: tuple
    #: Duration of each audio clip in seconds.
    audio_clip_duration: int
    #: Number of audio clips to extract (-1 for all).
    audio_num_clips: int

    # The webdataset decoder function, if to be applied
    _decoder: Optional[Callable[[FilteredSample], FilteredSample]]

    def __init__(
        self,
        path: EPath,
        *,
        auto_decode: bool = True,
        image_decode: ImageDecoder = "torchrgb",
        ignore_decoder_errors: bool = False,
        audio_clip_duration: int = 1,
        audio_num_clips: int = -1,
        video_decode: Literal["torch", "AVData"] = "torch",
        video_decode_audio: bool = False,
        video_num_frames: int = 64,
        video_out_frame_size: tuple = (224, 224),
        **kwargs,
    ):
        """
        Factory for the webdataset sample loader including the decoder.

        Args:
            path: Path to the dataset (passed to parent)
            auto_decode: If true, use the default webdataset sample decoder.
            image_decode: This defines the decoding results.
            ignore_decoder_errors: If true, ignore errors when decoding.
            audio_clip_duration: Duration of each audio clip in seconds.
            audio_num_clips: Number of audio clips to extract (-1 for all).
            video_decode: If "AVData", returns an AVData instance for flexible decoding. If "torch",
                returns decoded VideoData.
            video_decode_audio: Whether to decode audio from video files.
            video_num_frames: Number of video frames to extract.
            video_out_frame_size: Output size for video frames (width, height).
            **kwargs: Args passed to parent constructor
        """
        self.image_decode = image_decode
        self.ignore_decoder_errors = ignore_decoder_errors
        self.video_decode = video_decode
        self.video_decode_audio = video_decode_audio
        self.video_num_frames = video_num_frames
        self.video_out_frame_size = video_out_frame_size
        self.audio_clip_duration = audio_clip_duration
        self.audio_num_clips = audio_num_clips
        super().__init__(path, **kwargs)

        if auto_decode:
            self._decoder = webdataset.autodecode.Decoder(
                [
                    webdataset.autodecode.imagehandler(self.image_decode),
                    AVDecoder(
                        audio_clip_duration=audio_clip_duration,
                        audio_num_clips=audio_num_clips,
                        video_decode_audio=video_decode_audio,
                        video_num_frames=video_num_frames,
                        video_out_frame_size=video_out_frame_size,
                        video_decode=video_decode,
                    ),
                ]
            )
        else:
            self._decoder = None

    def _decode_error_handler(self, exc: Exception) -> bool:
        if self.ignore_decoder_errors:
            return True
        raise exc

    def load_sample(self, sample: FilteredSample) -> T_sample:
        if self._decoder is not None:
            sample = self._decoder(sample)
        return super().load_sample(sample)

    def config(self) -> Dict[str, Any]:
        return dict(
            **super().config(),
            image_decode=self.image_decode,
            ignore_decoder_errors=self.ignore_decoder_errors,
            video_decode=self.video_decode,
        )
