# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import torch

try:
    from nvidia import nvimgcodec

    NVIMAGECODEC_AVAILABLE = True
except ImportError as e:
    NVIMAGECODEC_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)


class NVImageCodecDecoder:
    """A decoder class for image data that uses the GPU accelerated NVImageCodec library

    Abstracts NVImageCodec so that image in webdataset can be transparently decoded on GPU.
    This can significantly accelerate image decoding via the optimized CUDA implementations
    of the decoders as well as the hardware JPEG decoders present on modern NVIDIA GPUs.

    """

    def __init__(self, colorspec: str = "nvimgcodec", decode_device: int = 0) -> None:

        if not NVIMAGECODEC_AVAILABLE:
            raise ImportError(
                f"GPU image decoding was requested but is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[gpu_image_decode]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}. Install megatron-energon[gpu_image_decode] to use GPU image decoding."
            )

        self.convert_to_float = not colorspec.endswith("8")
        colorspec_map = {
            "nvimgcodecl": nvimgcodec.ColorSpec.GRAY,
            "nvimgcodecrgb": nvimgcodec.ColorSpec.SRGB,
            "nvimgcodecrgba": nvimgcodec.ColorSpec.UNCHANGED,
            "nvimgcodec": nvimgcodec.ColorSpec.SRGB,
        }

        self.color_spec = colorspec_map[colorspec.replace("8", "")]
        self.decode_params = nvimgcodec.DecodeParams(color_spec=self.color_spec)
        self.decoder = nvimgcodec.Decoder(
            device_id=decode_device,
        )

    def __call__(self, key: str, data: bytes) -> torch.Tensor | None:
        """ """
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

        if getattr(nv_img, "__dlpack__", False):
            tensor_img = torch.from_dlpack(nv_img).permute(2, 0, 1)

            if self.color_spec == nvimgcodec.ColorSpec.UNCHANGED and tensor_img.shape[0] < 4:
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

        return None
