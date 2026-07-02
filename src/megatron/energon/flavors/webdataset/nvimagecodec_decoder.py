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

    def __init__(
        self,
        decode_device: int = 0

    ) -> None:
      self.decoder = nvimgcodec.Decoder(device_id = decode_device)

      if not NVIMAGECODEC_AVAILABLE:
        raise ImportError(
            f"GPU image decoding was requested but is not available. Please install the required dependencies with:\n"
            f"pip install megatron-energon[gpu_image_decode]\n"
            f"Missing dependency: {MISSING_DEPENDENCY}. Install megatron-energon[gpu_image_decode] to use GPU image decoding."
        )

    def __call__(
        self, key: str, data: bytes
    ) -> torch.Tensor | None:
        """

        """
        key = key.lower()
        if not any(
            key == ext or key.endswith("." + ext)
            for ext in ("jpeg", "jfif", "jpg", "jp2", "jpx", "j2k", "tiff", "tif", "bmp", "png", "pnm", "ppm", "pgm", "pbm")
        ):
            return None
        nv_img = self.decoder.decode(data)

        if getattr(nv_img, "to_dlpack", False):
          tensor = torch.from_dlpack(nv_img.to_dlpack())
          tensor = tensor.permute(2, 0, 1).float().div(255)
          return tensor

        return None
