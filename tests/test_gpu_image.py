import unittest
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from megatron.energon.flavors.webdataset.nvimagecodec_decoder import NVImageCodecDecoder
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder

from .test_av_decoder import tensors_close


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestGPUImageDecode(unittest.TestCase):
    """Test GPU image decoding."""
    def setUp(self):
      self.decoder = NVImageCodecDecoder()
      self.image_data = Path("tests/data/test_image.png").read_bytes()

    def test_non_image_returns_none(self):
      assert self.decoder("pdf", b"") is None

    def test_invalid_image_returns_none(self):
      assert self.decoder("png", b"") is None

    def test_decode_to_gpu(self) -> None:
      tensor = self.decoder("png", self.image_data)

      assert tensor is not None
      assert tensor.device.type == "cuda"

    def test_sample_decoder_dispatch(self):
        sample_decoder = SampleDecoder(image_decode="nvimagecodec")
        result = sample_decoder.decode("test.png", self.image_data)

        assert result is not None and result.device.type == "cuda"

        sample_decoder = SampleDecoder()
        result = sample_decoder.decode("test.png", self.image_data)

        assert result is not None and result.device.type == "cpu"

    def test_decode_matches_baseline(self) -> None:
      gpu_image = self.decoder("png", self.image_data)

      assert gpu_image is not None
      assert gpu_image.dtype == torch.float32
      assert gpu_image.ndim == 3
      assert gpu_image.shape == torch.Size([3, 248, 330])

      with BytesIO(self.image_data) as io:
        cpu_image = pil_to_tensor(Image.open(io)).float().div(255)

      assert tensors_close(cpu_image, gpu_image.cpu(), 0.001)
