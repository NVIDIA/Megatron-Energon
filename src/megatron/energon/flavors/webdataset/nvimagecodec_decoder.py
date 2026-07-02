# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import warnings
from typing import Literal

import torch

try:
    from nvidia import nvimgcodec

    NVIMAGECODEC_AVAILABLE = True
except ImportError as e:
    NVIMAGECODEC_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)


ColorSpaces = Literal[
    "nvimgcodecl8",
    "nvimgcodecrgb8",
    "nvimgcodecrgba8",
    "nvimgcodecl",
    "nvimgcodecrgb",
    "nvimgcodec",
    "nvimgcodecrgba",
]


class NVImageCodecDecoder:
    """A decoder class for image data that uses the GPU accelerated NVImageCodec library

    Abstracts NVImageCodec so that image in webdataset can be transparently decoded on GPU.
    This can significantly accelerate image decoding via the optimized CUDA implementations
    of the decoders as well as the hardware JPEG decoders present on modern NVIDIA GPUs.

    Args:
      color_space: The color space to use for decoding, prefixed with nvimgcodec. Can be one of
        nvimgcodecl: force grayscale
        nvimgcodecrgb: force 3 channel RGB (expands grayscale or drops alpha)
        nvimgcodec: don't change source color space (default)
        nvimgcodecrgba: force 4 channel alpha

        Suffix with "8" to load in uint8, otherwise returned tensors will be converted to float in [0, 1]
      device: The CUDA device ordinal to use for decoding (0 by default)
      suppress_warnings: Don't warn about failed decodings
    """

    def __init__(
        self,
        color_space: ColorSpaces = "nvimgcodec",
        device: int = 0,
        suppress_warnings: bool = False,
    ) -> None:
        if not NVIMAGECODEC_AVAILABLE:
            raise ImportError(
                f"GPU image decoding was requested but is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[gpu_image_decode]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}. Install megatron-energon[gpu_image_decode] to use GPU image decoding."
            )

        self.suppress_warnings = suppress_warnings
        self.convert_to_float = not color_space.endswith("8")
        color_space_map = {
            "nvimgcodecl": nvimgcodec.ColorSpec.GRAY,
            "nvimgcodecrgb": nvimgcodec.ColorSpec.SRGB,
            "nvimgcodecrgba": nvimgcodec.ColorSpec.UNCHANGED,
            "nvimgcodec": nvimgcodec.ColorSpec.UNCHANGED,
        }

        self.color_space = color_space
        self.decode_params = nvimgcodec.DecodeParams(
            color_spec=color_space_map[color_space.replace("8", "")]
        )
        self.decoder = nvimgcodec.Decoder(
            device_id=device,
        )

    def __call__(self, key: str, data: bytes) -> torch.Tensor | None:
        """Decode image data using the GPU accelerated decoder

        Args:
            key: image file extension
            data: raw image bytes

        Returns:
            A tensor with the image pixels or None if decoding failed or file type is not supported.
        """
        key = key.lower()
        if not any(
            key == ext or key.endswith("." + ext)
            for ext in (
                "jpeg",
                "jfif",
                "jpg",
                "jp2",
                "jpx",
                "j2k",
                "tiff",
                "tif",
                "bmp",
                "png",
                "pnm",
                "ppm",
                "pgm",
                "pbm",
            )
        ):
            return None
        nv_img = self.decoder.decode(data, params=self.decode_params)

        if hasattr(nv_img, "__dlpack__"):
            tensor_img = torch.from_dlpack(nv_img).permute(2, 0, 1)

            if self.color_space == "nvimgcodecrgba" and tensor_img.shape[0] < 4:
                if tensor_img.shape[0] == 1:
                    tensor_img = tensor_img.expand(3, *tensor_img.shape[1:]).contiguous()
                alpha = torch.full(
                    [1, *tensor_img.shape[1:]],
                    255,
                    dtype=tensor_img.dtype,
                    device=tensor_img.device,
                )
                tensor_img = torch.cat([tensor_img, alpha])

            if self.convert_to_float:
                tensor_img = tensor_img.float().div(255)

            return tensor_img

        if not self.suppress_warnings:
            warnings.warn(
                "GPU Image decoding failed, will attempt fallback",
                RuntimeWarning,
            )

        return None
