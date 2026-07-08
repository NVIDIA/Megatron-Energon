# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import multiprocessing
import queue
import unittest
from io import BytesIO
from pathlib import Path

import PIL
import torch
import webdataset
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from megatron.energon.flavors.webdataset.nvimagecodec_decoder import NVImageCodecDecoder
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder


@unittest.skipUnless(torch.cuda.is_available(), "GPU required")
class TestGPUImageDecode(unittest.TestCase):
    """Test GPU image decoding."""

    def setUp(self):
        self.decoder = NVImageCodecDecoder()
        self.image_data = Path("tests/data/test_image.png").read_bytes()

    def test_non_image_returns_none(self):
        """Tests that non-image types are correctly skipped"""
        assert self.decoder("pdf", b"") is None

    def test_invalid_image_returns_none(self):
        """Tests that invalid images are correctly skipped"""
        assert self.decoder("png", b"") is None

    def test_decode_to_gpu(self) -> None:
        """Tests that valid images are decoded to GPU"""
        tensor = self.decoder("png", self.image_data)

        assert tensor is not None
        assert tensor.device.type == "cuda"

    def test_sample_decoder_dispatch(self):
        """Tests that the SampleDecoder correctly dispatches to CPU or GPU decoder depending on user choice"""
        sample_decoder = SampleDecoder(image_decode="torch", image_decode_device="gpu")
        result = sample_decoder.decode("test.png", self.image_data)

        assert result is not None and result.device.type == "cuda"

        sample_decoder = SampleDecoder(image_decode="torch")
        result = sample_decoder.decode("test.png", self.image_data)

        assert result is not None and result.device.type == "cpu"

    def test_cpu_fallback(self):
        """Tests that failed GPU decoding falls back to CPU decoding"""
        sample_decoder = SampleDecoder(image_decode="torch", image_decode_device="gpu")

        with self.assertRaises(webdataset.autodecode.DecodingError) as ctx:
            sample_decoder.decode("test.png", b"")

        assert isinstance(ctx.exception.__cause__, PIL.UnidentifiedImageError)

    def test_gpu_decoding_throws_on_non_torch(self):
        """Test that a non-torch result type throws on GPU decoding"""
        with self.assertRaises(ValueError):
            SampleDecoder(image_decode="pil", image_decode_device="gpu")

    def test_decode_matches_baseline(self) -> None:
        """Tests that GPU decoded images are correctly decoded"""
        gpu_image = self.decoder("png", self.image_data)

        assert gpu_image is not None
        assert gpu_image.dtype == torch.float32
        assert gpu_image.ndim == 3
        assert gpu_image.shape == torch.Size([3, 248, 330])

        with BytesIO(self.image_data) as io:
            cpu_image = pil_to_tensor(Image.open(io)).float().div(255)

        assert torch.allclose(cpu_image, gpu_image.cpu())

    def test_decode_uint8(self) -> None:
        """Tests that 8bit decoding requests work correctly"""
        decoder = NVImageCodecDecoder("torchrgb8")
        gpu_image = decoder("png", self.image_data)

        assert gpu_image.dtype == torch.uint8

    def test_decode_channel_coercion(self) -> None:
        """Tests that channel counts (1/3/4) are forced correctly if requested"""
        gray_image_data = Path("tests/data/test_image_l.png").read_bytes()
        rgba_image_data = Path("tests/data/test_image_rgba.png").read_bytes()

        decoder = NVImageCodecDecoder("torch")
        gray_image = decoder("png", gray_image_data)
        rgba_image = decoder("png", rgba_image_data)

        assert gray_image.shape[0] == 1
        assert rgba_image.shape[0] == 4

        decoder = NVImageCodecDecoder("torchrgb")
        gray_image = decoder("png", gray_image_data)
        rgba_image = decoder("png", rgba_image_data)

        assert gray_image.shape[0] == rgba_image.shape[0] == 3

        decoder = NVImageCodecDecoder("torchl")
        gray_image = decoder("png", gray_image_data)
        rgba_image = decoder("png", rgba_image_data)

        assert gray_image.shape[0] == rgba_image.shape[0] == 1

        decoder = NVImageCodecDecoder("torchrgba")
        gray_image = decoder("png", gray_image_data)
        rgba_image = decoder("png", rgba_image_data)

        assert gray_image.shape[0] == rgba_image.shape[0] == 4

    def test_decode_non_png(self) -> None:
        """Tests that a non-png is decodable correctly"""
        jp2_image_data = Path("tests/data/test_image.jp2").read_bytes()

        decoder = NVImageCodecDecoder()
        gpu_image = decoder("jp2", jp2_image_data)

        assert gpu_image is not None
        assert gpu_image.device.type == "cuda"
        assert gpu_image.dtype == torch.float32
        assert gpu_image.ndim == 3
        assert gpu_image.shape == torch.Size([4, 248, 330])

        with Path("tests/data/test_image.jp2").open("rb") as f:
            cpu_image = pil_to_tensor(Image.open(f)).float().div(255)

        assert torch.allclose(cpu_image, gpu_image.cpu(), atol=0.05)

    def test_gpu_decode_fails_on_fork(self) -> None:
      sample_decoder = SampleDecoder(image_decode="torch", image_decode_device="gpu")

      ctx = multiprocessing.get_context("fork")
      result = ctx.Queue()

      def decode_in_fork() -> None:
        try:
          sample_decoder.decode("test.png", self.image_data)
          result.put(None)
        except Exception as e:
          result.put(e)

      proc = ctx.Process(target=decode_in_fork)
      proc.start()
      proc.join(30)

      try:
        maybeError = result.get(timeout=30)
      except queue.Empty:
        self.fail("No result from forked child")
      finally:
        if proc.is_alive():
          proc.kill()

      self.assertIsInstance(maybeError, SystemError)
