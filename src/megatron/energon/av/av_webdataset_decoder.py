# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import io
from typing import Literal, Optional, Union

import av

from megatron.energon.av.av_data import AVData
from megatron.energon.av.av_decoder import AVDecoder
from megatron.energon.av.av_decoder_gpu import AVDecoderGpu
from megatron.energon.av.av_init import av_open


class AVWebdatasetDecoder:
    """A decoder class for audio and video data that provides a consistent interface for decoding media files.

    This class encapsulates the decoding parameters and provides a callable interface that can be used
    with webdataset or other data loading pipelines. It supports both video and audio decoding with
    configurable parameters for frame extraction, resizing, and audio clip extraction.

    Args:
        video_decode_audio: Whether to decode audio from video files. If True, audio will be
            extracted alongside video frames.
        av_decode: If "AVDecoder", returns an AVDecoder instance for flexible decoding. If "torch",
            returns decoded VideoData.
        device: If "gpu" or a numerical device ID the video is decoded using NVDec hardware acceleration
          on the GPU. This option is ignored if using "torch" or "pyav" for the av_decode parameter.

    Example:
        >>> decoder = AVWebdatasetDecoder(
        ...     video_decode_audio=True,
        ...     av_decode="AVDecoder"
        ... )
        >>> result = decoder("video.mp4", video_bytes)
    """

    def __init__(
        self,
        video_decode_audio: bool,
        av_decode: Literal["torch", "AVDecoder", "pyav"] = "AVDecoder",
        device: Union[Literal["cpu", "gpu"], int] = "cpu",
    ) -> None:
        self.video_decode_audio = video_decode_audio
        self.av_decode = av_decode
        self.device = device

    def read_av_data(self, data: bytes) -> AVDecoder:
        """Decoder function that returns an AVData object for flexible decoding.

        Args:
            data: The raw bytes of the media file

        Returns:
            AVData object that can be used to decode the media with custom parameters
        """
        if self.device != "cpu":
            device_id = 0 if self.device == "gpu" else self.device
            return AVDecoderGpu(io.BytesIO(data), device_id=int(device_id))

        return AVDecoder(io.BytesIO(data))

    def __call__(
        self, key: str, data: bytes
    ) -> Optional[
        Union[AVData, AVDecoder, "av.container.InputContainer", "av.container.OutputContainer"]
    ]:
        """
        Extract the video or audio data from default media extensions.

        Args:
            key: media file extension
            data: raw media bytes

        Returns:
            If av_decode is "torch", returns VideoData containing the decoded frames and metadata.
            If av_decode is "AVDecoder", returns an AVDecoder instance for flexible decoding.
            If av_decode is "pyav", returns an av.container.InputContainer instance.
            Returns None if decoding failed or file type is not supported.
        """
        key = key.lower()
        if not any(
            key == ext or key.endswith("." + ext)
            for ext in ("mp4", "avi", "mov", "webm", "mkv", "flac", "mp3", "wav", "flv")
        ):
            return None

        av_decoder = self.read_av_data(data)

        if self.av_decode == "AVDecoder":
            return av_decoder
        elif self.av_decode == "pyav":
            return av_open(av_decoder.stream)
        elif self.av_decode == "torch":
            return av_decoder.get_frames(
                video_decode_audio=self.video_decode_audio,
            )
        else:
            raise ValueError(f"Invalid av_decode value: {self.av_decode}")
