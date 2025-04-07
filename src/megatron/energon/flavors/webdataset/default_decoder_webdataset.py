# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Dict, Generic, Optional, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.default_generic_webdataset import (
    DefaultGenericWebdatasetFactory,
)
from megatron.energon.flavors.webdataset.sample_decoder import (
    AVDecoder,
    ImageDecoder,
    SampleDecoder,
)
from megatron.energon.flavors.webdataset.structs import FilteredSample

T_sample = TypeVar("T_sample", covariant=True)


class DefaultDecoderWebdatasetFactory(DefaultGenericWebdatasetFactory[T_sample], Generic[T_sample]):
    """
    Extends the default webdataset loading with decoding of contained files, such as images, videos or nested
    containers.
    """

    #: If true, ignore errors when decoding.
    ignore_decoder_errors: bool

    # The webdataset decoder function, if to be applied
    _decoder: Optional[SampleDecoder]

    def __init__(
        self,
        path: EPath,
        *,
        auto_decode: bool = True,
        image_decode: ImageDecoder = "torchrgb",
        ignore_decoder_errors: bool = False,
        av_decode: AVDecoder = "AVDecoder",
        video_decode_audio: bool = False,
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
            av_decode: If "AVDecoder", returns an AVDecoder instance for flexible decoding. If "torch",
                returns decoded VideoData.
            video_decode_audio: Whether to decode audio from video files.
            video_num_frames: Number of video frames to extract.
            video_out_frame_size: Output size for video frames (width, height).
            **kwargs: Args passed to parent constructor
        """
        self.image_decode = image_decode
        self.ignore_decoder_errors = ignore_decoder_errors
        self.av_decode = av_decode
        self.video_decode_audio = video_decode_audio
        super().__init__(path, **kwargs)

        if auto_decode:
            self._decoder = SampleDecoder(
                image_decode=image_decode,
                av_decode=av_decode,
                video_decode_audio=video_decode_audio,
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
            **(self._decoder.config() if self._decoder is not None else {}),
        )
