from typing import Any, Callable, Dict, Generic, Literal, Optional, TypeVar

import webdataset

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.base_webdataset import T_sample
from megatron.energon.flavors.webdataset.default_generic_webdataset import DefaultGenericWebdataset
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


class DefaultDecoderWebdataset(DefaultGenericWebdataset[T_sample], Generic[T_sample]):
    """
    Basic class for any dataset which contains images and / or videos. Applies default wds loading logic for all
    known extensions.
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
        Decoder dataset.

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

    # def _process_samples(self, dataset: SavableDataset[Tuple[Optional[FilteredSample], ...]]) -> SavableDataset[T_sample]:
    #     f = webdataset.autodecode.Decoder([
    #         webdataset.autodecode.imagehandler(self.image_decode),
    #         self._video_decoder,
    #     ])

    #     # Un/Wraps the tuple for the decoder
    #     def decoder_fn(sample: Tuple[Optional[FilteredSample], ...]) -> Tuple[Optional[FilteredSample], ...]:
    #         return f(*sample),

    #     dataset = MapDataset(
    #         dataset,
    #         decoder_fn,
    #         error_handler=self.error_handler,
    #         stateless_map_fn=True,
    #         worker_config=self.worker_config,
    #     )
    #     return super()._process_samples(dataset)

    def config(self) -> Dict[str, Any]:
        return {
            **super().config(),
            "image_decode": self.image_decode,
            "ignore_decoder_errors": self.ignore_decoder_errors,
        }
