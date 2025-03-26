# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests video decoding."""

import io
import logging
import os
import sys
import unittest

import av
import numpy as np
import torch
import torchvision.transforms as transforms

from megatron.energon.av import AVDecoder, get_clips_uniform, get_single_frames_uniform

# Set multiprocessing start method to 'spawn' on macOS to avoid DataLoader cleanup issues
if sys.platform == "darwin":
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)


def load_video_to_tensor(video_path: str) -> torch.Tensor:
    """Load a video file into a tensor using PyAV directly.

    Args:
        video_path: Path to the video file

    Returns:
        Tensor of shape [num_frames, channels, height, width]
    """
    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        frame = frame.reformat(format="rgb24")
        frames.append(torch.from_numpy(frame.to_ndarray()))

    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    return video_tensor


def are_resized_frames_close(
    tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 0.01
) -> bool:
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
        self.loaders = []  # Keep track of loaders for cleanup

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any loaders
        for loader in self.loaders:
            if hasattr(loader, "_iterator"):
                loader._iterator = None
            if hasattr(loader, "_shutdown_workers"):
                try:
                    loader._shutdown_workers()
                except Exception:
                    pass

    def decode_baseline_video_pyav(self):
        """Load the baseline video using PyAV directly."""
        self.complete_video_tensor = load_video_to_tensor("tests/data/sync_test.mp4")

    def test_decode_all_frames(self):
        """Test decoding all frames from a video file."""
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using AVData
        av_decoder = AVDecoder(stream)
        av_data = av_decoder.get_frames()
        video_tensor = av_data.video_clips[0]

        print(video_tensor.shape)
        assert (video_tensor == self.complete_video_tensor).all(), (
            "Energon decoded video does not match baseline"
        )

    def test_decode_strided_resized(self):
        """Test decoding a subset of frames with resizing."""
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using AVData
        av_decoder = AVDecoder(stream)
        video_tensor = get_single_frames_uniform(
            av_decoder=av_decoder, num_frames=64, video_out_frame_size=(224, 224)
        )

        # Get strided frames from baseline complete video tensor
        strided_baseline_tensor = self.complete_video_tensor[
            np.linspace(0, self.complete_video_tensor.shape[0] - 1, 64, dtype=int).tolist()
        ]
        # Now resize the baseline frames
        resize = transforms.Resize((224, 224))
        strided_resized_baseline_tensor = resize(strided_baseline_tensor)

        # We allow small numerical differences due to different resize implementations
        assert are_resized_frames_close(
            video_tensor, strided_resized_baseline_tensor, tolerance=0.01
        ), "Energon decoded video does not match baseline"

    def test_decode_strided_resized_with_audio(self):
        """Test decoding video frames and audio clips together."""
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using AVData
        av_data = AVDecoder(stream)
        video_tensor, audio_tensor, metadata = av_data.decode_video_frames(
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
        strided_resized_baseline_tensor = resize(strided_baseline_tensor)

        # We allow small numerical differences due to different resize implementations
        assert are_resized_frames_close(
            video_tensor, strided_resized_baseline_tensor, tolerance=0.01
        ), "Energon decoded video does not match baseline"

        # Check audio tensor shape (5 clips, channels, 3 seconds at original sample rate)
        expected_samples = int(3 * metadata["audio_fps"])  # 3 seconds at original sample rate
        assert audio_tensor.shape == torch.Size([5, audio_tensor.shape[1], expected_samples]), (
            f"Energon decoded audio clips have wrong size: {audio_tensor.shape}"
        )


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
        self.loaders = []  # Keep track of loaders for cleanup

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any loaders
        for loader in self.loaders:
            if hasattr(loader, "_iterator"):
                loader._iterator = None
            if hasattr(loader, "_shutdown_workers"):
                try:
                    loader._shutdown_workers()
                except Exception:
                    pass

    def decode_baseline_audio_pyav(self):
        """Load the baseline audio using PyAV directly."""
        self.complete_audio_tensor = load_audio_to_tensor("tests/data/test_audio.flac")

    def test_decode_all_samples(self):
        """Test decoding all samples from an audio file."""
        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        av_decoder = AVDecoder(stream)
        # audio_tensor, _ = av_data.decode_audio_samples(num_clips=-1)
        av_data = av_decoder.get_clips(audio_clip_ranges=[(0, float("inf"))], audio_unit="samples")
        audio_tensor = av_data.audio_clips[0]

        assert (audio_tensor == self.complete_audio_tensor).all(), (
            "Energon decoded audio does not match baseline"
        )

    def test_decode_clips(self):
        """Test decoding multiple clips from an audio file."""
        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        # Decode using AVData
        av_decoder = AVDecoder(stream)
        av_data = get_clips_uniform(
            av_decoder=av_decoder, num_clips=5, clip_duration_seconds=3, request_audio=True
        )
        audio_tensor = av_data.audio_clips[0]
        audio_sps = av_decoder.get_audio_samples_per_second()

        # Check audio tensor shape (5 clips, channels, 3 seconds at original sample rate)
        assert len(av_data.audio_clips) == 5
        assert len(av_data.audio_timestamps) == 5
        assert audio_tensor.shape[1] >= int(3 * audio_sps)
        assert audio_tensor.shape[1] <= int(4 * audio_sps)

    def test_decode_wav(self):
        """Test decoding a WAV file."""
        # Skip WAV test if file doesn't exist
        if not os.path.exists("tests/data/test_audio.wav"):
            self.skipTest("WAV test file not found")
            return

        with open("tests/data/test_audio.wav", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        av_decoder = AVDecoder(stream)
        av_data = get_clips_uniform(
            av_decoder=av_decoder, num_clips=3, clip_duration_seconds=3, request_audio=True
        )
        audio_sps = av_decoder.get_audio_samples_per_second()

        # Check audio tensor shape (3 clips, 2 channels, samples)
        expected_samples = int(3 * audio_sps)  # 3 seconds at original sample rate
        assert all(
            audio_tensor.shape == torch.Size([2, expected_samples])
            for audio_tensor in av_data.audio_clips
        ), "Energon decoded WAV file has wrong shape."


if __name__ == "__main__":
    unittest.main()
