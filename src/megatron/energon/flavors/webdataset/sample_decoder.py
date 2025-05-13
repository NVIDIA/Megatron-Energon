# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Literal

import webdataset

from megatron.energon.av import AVWebdatasetDecoder
from megatron.energon.cache import FileStoreDecoder
from megatron.energon.flavors.webdataset.config import split_name_re

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
        av_decode: AVDecoderType = "AVDecoder",
        video_decode_audio: bool = False,
        guess_content: bool = False,
    ):
        """
        Args:
            image_decode: This defines the decoding results.
            av_decode: If "AVDecoder", returns an AVDecoder instance for flexible decoding. If "torch",
                returns decoded VideoData.
            video_decode_audio: Whether to decode audio from video files.
            guess_content: Whether to guess the contents of the file using the `filetype` package.
        """
        self._config = dict(
            image_decode=image_decode,
            av_decode=av_decode,
            video_decode_audio=video_decode_audio,
            guess_content=guess_content,
        )
        self._decoder = webdataset.autodecode.Decoder(
            GuessingHandlerWrapper.wrap(
                guess_content,
                [
                    webdataset.autodecode.imagehandler(image_decode),
                    AVWebdatasetDecoder(
                        video_decode_audio=video_decode_audio,
                        av_decode=av_decode,
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
        return self._decoder(sample)

    def config(self) -> dict:
        return self._config
