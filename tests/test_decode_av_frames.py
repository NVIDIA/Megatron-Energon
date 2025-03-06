# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests video decoding."""

import av
import io
import logging
import numpy as np
import sys
import torch
import torchvision.transforms as transforms
import unittest
import os

from megatron.energon.flavors.webdataset.decode_av_frames import decode_audio_samples, decode_video_frames, get_clip_indices

def load_video_to_tensor(video_path: str) -> torch.Tensor:
    """Load a video file into a tensor using PyAV directly.

    Args:
        video_path: Path to the video file

    Returns:
        Tensor of shape [num_frames, height, width, channels]
    """
    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        frame = frame.reformat(format="rgb24")
        frames.append(torch.from_numpy(frame.to_ndarray()))

    video_tensor = torch.stack(frames)
    return video_tensor


def are_resized_frames_close(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 0.01) -> bool:
    """Compare two tensors of video frames with a tolerance for resizing differences.

    Args:
        tensor1: First tensor of frames
        tensor2: Second tensor of frames
        tolerance: Maximum allowed mean absolute error

    Returns:
        True if tensors are close enough, False otherwise
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Input tensors must have the same shape.")
    tensor1 = tensor1.float() / 255.0
    tensor2 = tensor2.float() / 255.0
    # Compute Mean Absolute Error
    mae = torch.mean(torch.abs(tensor1 - tensor2)).item()
    return mae <= tolerance


class TestVideoDecode(unittest.TestCase):
    """Test video decoding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.decode_baseline_video_pyav()

    def decode_baseline_video_pyav(self):
        """Load the baseline video using PyAV directly."""
        self.complete_video_tensor = load_video_to_tensor("tests/data/sync_test.mp4")

    def test_decode_all_frames(self):
        """Test decoding all frames from a video file."""
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using fastseek Energon wrapper
        video_tensor, _, _ = decode_video_frames(stream=stream)

        assert (video_tensor == self.complete_video_tensor).all(), \
            "Energon decoded video does not match baseline"

    def test_decode_strided_resized(self):
        """Test decoding a subset of frames with resizing."""
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using fastseek Energon wrapper
        video_tensor, _, _ = decode_video_frames(
            stream=stream,
            num_frames=64,
            out_frame_size=(224, 224),
        )

        # Get strided frames from baseline complete video tensor
        strided_baseline_tensor = self.complete_video_tensor[
            np.linspace(0, self.complete_video_tensor.shape[0] - 1, 64, dtype=int).tolist()
        ]
        # Now resize the baseline frames
        resize = transforms.Resize((224, 224))
        strided_baseline_tensor = strided_baseline_tensor.permute(0, 3, 1, 2)  # b, h, w, c -> b, c, h, w
        strided_resized_baseline_tensor = resize(strided_baseline_tensor)
        strided_resized_baseline_tensor = strided_resized_baseline_tensor.permute(0, 2, 3, 1)  # b, c, h, w -> b, h, w, c

        # We allow small numerical differences due to different resize implementations
        assert are_resized_frames_close(video_tensor, strided_resized_baseline_tensor, tolerance=0.01), \
            "Energon decoded video does not match baseline"

    def test_decode_strided_resized_with_audio(self):
        """Test decoding video frames and audio clips together."""
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using fastseek Energon wrapper
        video_tensor, audio_tensor, metadata = decode_video_frames(
            stream=stream,
            num_frames=64,
            out_frame_size=(224, 224),
            decode_audio=True,
            num_clips=5,
            clip_duration=3,
        )

        # Get strided frames from baseline complete video tensor
        strided_baseline_tensor = self.complete_video_tensor[
            np.linspace(0, self.complete_video_tensor.shape[0] - 1, 64, dtype=int).tolist()
        ]
        # Now resize the baseline frames
        resize = transforms.Resize((224, 224))
        strided_baseline_tensor = strided_baseline_tensor.permute(0, 3, 1, 2)  # b, h, w, c -> b, c, h, w
        strided_resized_baseline_tensor = resize(strided_baseline_tensor)
        strided_resized_baseline_tensor = strided_resized_baseline_tensor.permute(0, 2, 3, 1)  # b, c, h, w -> b, h, w, c

        # We allow small numerical differences due to different resize implementations
        assert are_resized_frames_close(video_tensor, strided_resized_baseline_tensor, tolerance=0.01), \
            "Energon decoded video does not match baseline"

        # Check audio tensor shape (5 clips, channels, 3 seconds at original sample rate)
        expected_samples = int(3 * metadata["audio_fps"])  # 3 seconds at original sample rate
        assert audio_tensor.shape == torch.Size([5, audio_tensor.shape[1], expected_samples]), \
            f"Energon decoded audio clips have wrong size: {audio_tensor.shape}"


def load_audio_to_tensor(audio_path: str) -> torch.Tensor:
    """Load an audio file into a tensor using PyAV directly.

    Args:
        audio_path: Path to the audio file

    Returns:
        Tensor of shape [channels, samples]
    """
    container = av.open(audio_path)
    frames = []

    for frame in container.decode(audio=0):
        frames.append(torch.from_numpy(frame.to_ndarray()))

    audio_tensor = torch.cat(frames, dim=-1)
    return audio_tensor


class TestAudioDecode(unittest.TestCase):
    """Test audio decoding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.decode_baseline_audio_pyav()

    def decode_baseline_audio_pyav(self):
        """Load the baseline audio using PyAV directly."""
        self.complete_audio_tensor = load_audio_to_tensor("tests/data/test_audio.flac")

    def test_decode_all_samples(self):
        """Test decoding all samples from an audio file."""
        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        _, audio_tensor, _ = decode_audio_samples(stream=stream, num_clips=-1)

        assert (audio_tensor == self.complete_audio_tensor).all(), \
            "Energon decoded audio does not match baseline"

    def test_decode_clips(self):
        """Test decoding multiple clips from an audio file."""
        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using fastseek Energon wrapper
        _, audio_tensor, metadata = decode_audio_samples(
            stream=stream,
            num_clips=5,
            clip_duration=3,
        )

        # Check audio tensor shape (5 clips, channels, 3 seconds at original sample rate)
        expected_samples = int(3 * metadata["audio_fps"])  # 3 seconds at original sample rate
        expected_samples = min(expected_samples, audio_tensor.shape[2])  # Don't exceed actual length
        assert audio_tensor.shape == torch.Size([5, audio_tensor.shape[1], expected_samples]), \
            f"Energon decoded audio clips have wrong size: {audio_tensor.shape}"

    def test_decode_wav(self):
        """Test decoding a WAV file."""
        # Skip WAV test if file doesn't exist
        if not os.path.exists("tests/data/test_audio.wav"):
            self.skipTest("WAV test file not found")
            return

        with open("tests/data/test_audio.wav", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        _, audio_tensor, metadata = decode_audio_samples(
            stream=stream,
            num_clips=1,
            clip_duration=3,
            audio_format="wav",
        )

        # Check audio tensor shape (1 clip, channels, samples)
        expected_samples = int(3 * metadata["audio_fps"])  # 3 seconds at original sample rate
        assert audio_tensor.shape == torch.Size([expected_samples, audio_tensor.shape[1]]), \
            f"Energon decoded WAV file has wrong size: {audio_tensor.shape}"


if __name__ == "__main__":
    unittest.main()