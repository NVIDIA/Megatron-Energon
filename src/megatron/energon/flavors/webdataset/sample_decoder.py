# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any, Callable, Literal

import webdataset

from megatron.energon.av.av_webdataset_decoder import AVWebdatasetDecoder
from megatron.energon.cache import FileStoreDecoder
from megatron.energon.flavors.webdataset.config import split_name_re
from megatron.energon.flavors.webdataset.nvimagecodec_decoder import NVImageCodecDecoder

try:
    import filetype
except ImportError:
    filetype = None

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


class GuessingHandlerWrapper:
    """A wrapper that guesses the extension of the file using the `filetype` package."""

    def __init__(self, handler: Callable[[str, bytes], Any]):
        """
        Wraps a handler to guess the extension of the file using the `filetype` package.

        Args:
            handler: The handler to wrap.
        """
        self.handler = handler
        if filetype is None:
            raise ImportError("filetype is not installed. Install it with `pip install filetype`.")

    def __call__(self, key: str, data: bytes) -> Any:
        """The handler that guesses the extension of the file using the `filetype` package, then calls the delegate handler."""
        kind = filetype.guess(data)
        if kind is not None:
            key = kind.extension
        return self.handler(key, data)

    @staticmethod
    def wrap(
        active: bool, handlers: list[Callable[[str, bytes], Any]]
    ) -> list[Callable[[str, bytes], Any]]:
        """
        Wraps a list of handlers to guess the extension of the file using the `filetype` package.

        Args:
            active: Whether to wrap the handlers.
            handlers: The handlers to wrap.

        Returns:
            The list of wrapped handlers.
        """
        if not active:
            return handlers
        return [GuessingHandlerWrapper(handler) for handler in handlers]


class SampleDecoder(FileStoreDecoder):
    """The default decoder for webdataset samples."""

    def __init__(
        self,
        *,
        image_decode: ImageDecoderType = "torchrgb",
        image_decode_device: Literal["cpu", "gpu"] | int = "cpu",
        av_decode: AVDecoderType = "AVDecoder",
        video_decode_audio: bool = False,
        video_decode_device: Literal["cpu", "gpu"] | int = "cpu",
        guess_content: bool = False,
    ):
        """
        Args:
            image_decode: This defines the decoding results.
            image_decode_device: device to use for decoding images, use `gpu` or an integer device
              ordinal to enable hardware accelerated image decoding.
              NOTE: GPU accelerated decoding is only compatible with `torch*` settings for `image_decode`
            av_decode: If "AVDecoder", returns an AVDecoder instance for flexible decoding. If "torch",
                returns decoded VideoData.
            video_decode_audio: Whether to decode audio from video files.
            video_decode_device: The device to use for decoding video. If "gpu" or a numerical device ID
              the video is decoded using NVDec hardware acceleration on the GPU.
            guess_content: Whether to guess the contents of the file using the `filetype` package.
        """
        self._config = dict(
            image_decode=image_decode,
            av_decode=av_decode,
            video_decode_audio=video_decode_audio,
            video_decode_device=video_decode_device,
            guess_content=guess_content,
        )
        self._creator_pid = os.getpid()
        self._requires_threading = image_decode_device != "cpu"
        if image_decode_device != "cpu":
            if not image_decode.startswith("torch"):
                raise ValueError(
                    f"GPU accelerated image decoding is only compatible with torch result formats (got {image_decode=})"
                )

            image_decoders = [
                NVImageCodecDecoder(
                    image_decode, 0 if image_decode_device == "gpu" else image_decode_device
                ),
                webdataset.autodecode.imagehandler(image_decode),
            ]
        else:
            image_decoders = [webdataset.autodecode.imagehandler(image_decode)]
        self._decoder = webdataset.autodecode.Decoder(
            GuessingHandlerWrapper.wrap(
                guess_content,
                [
                    *image_decoders,
                    AVWebdatasetDecoder(
                        video_decode_audio=video_decode_audio,
                        av_decode=av_decode,
                        device=video_decode_device,
                    ),
                ],
            ),
        )

    def decode(self, fname: str, raw: bytes) -> Any:
        m = split_name_re.match(fname)
        if m:
            cur_base_name, ext = m.groups()
        else:
            # If the file name does not match the expected format, use the file name as is.
            # The handlers may still handle the file correctly (e.g. if guessing by content).
            cur_base_name = fname
            ext = "."

        return self(
            {
                "__key__": cur_base_name,
                ext: raw,
            }
        )[ext]

    def __call__(self, sample: dict) -> dict:
        if self._requires_threading and os.getpid() != self._creator_pid:
            raise SystemError(
                "GPU accelerated media decoding is incompatible with forking dataloader workers "
                "pass `worker_type='thread'` or `worker_type='main'`."
            )

        return self._decoder(sample)

    def config(self) -> dict:
        return self._config


DEFAULT_DECODER = SampleDecoder()
