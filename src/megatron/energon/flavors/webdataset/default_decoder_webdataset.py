# Copyright (c) 2025, NVIDIA CORPORATION and Alibaba PAI.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Generic, Literal, Optional, TypeVar, Sequence

import re
import pickle
import webdataset
import webdataset.autodecode

from megatron.energon.epathlib import EPath
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

class NestedMultimodalHandler:
    def __init__(self, base_image_handler, base_video_handler):
        """Create an multimodal handler for images or videos.

        :param imagespec: short string indicating the type of decoding
        """
        self.extensions = webdataset.autodecode.IMAGE_EXTENSIONS
        self.image_handler = base_image_handler
        self.video_handler = base_video_handler

    def __call__(self, key, data):
        """Perform nested multimodal decoding. 
        Any extension of the list should be "{base key}" + "s", e.g., "jpgs".

        :param key: file name extension
        :param data: binary data
        """    
        extension = re.sub(r".*[.]", "", key).lower()
        if not extension.endswith("s"):
            return None
        base_key = key[:-1]
        base_extension = extension[:-1]
        try:
            data = pickle.loads(data)
        except:
            return None
        if not isinstance(data, Sequence):
            return None
        maybe_videos = [self.video_handler(base_key, item) for item in data]
        if sum([d is not None for d in maybe_videos]) == len(maybe_videos):
            return maybe_videos
        if base_extension in self.extensions:
            return [self.image_handler(base_key, d) for d in data]
        return None

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
            image_decoder = webdataset.autodecode.imagehandler(self.image_decode)
            self._decoder = webdataset.autodecode.Decoder(
                [
                    image_decoder,
                    self._video_decoder,
                    NestedMultimodalHandler(image_decoder, self._video_decoder)
                ]
            )
        else:
            self._decoder = None

    def _decode_error_handler(self, exc: Exception) -> bool:
        if self.ignore_decoder_errors:
            return True
        raise exc

    def _video_decoder(self, key, data):
        """Extract the video data from default video extensions."""
        # TODO: This function could be more efficient. It will write the data to `/tmp`,
        # then load it using `torchvision.io.video.read_video` which uses `av.open` from pyav.
        # pyav allows providing a file-like object, but torchvision does not expose that interface.
        # (https://github.com/pytorch/vision/issues/8438)
        video = webdataset.torch_video(key, data)
        if video is not None:
            return VideoData(
                frames=video[0].permute((0, 3, 1, 2)),
                aframes=video[1],
                info=video[2],
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
