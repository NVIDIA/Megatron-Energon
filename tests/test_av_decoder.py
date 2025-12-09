# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""This module tests video decoding."""

import io
import logging
import os
import pickle
import sys
import time
import unittest
from pathlib import Path

import av
import numpy as np
import torch
import torchvision.transforms as transforms

from megatron.energon.av import AVDecoder, get_clips_uniform, get_single_frames_uniform
from megatron.energon.av.fastseek.fastseek import Fastseek
from megatron.energon.av.fastseek.keyframeinfo import KeyframeInfo
from megatron.energon.media import AVMetadata

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
    for stream in container.streams:
        cc = stream.codec_context
        cc.thread_type = "NONE"
        cc.thread_count = 0

    frames = []

    for frame in container.decode(video=0):
        frame = frame.reformat(format="rgb24")
        frames.append(torch.from_numpy(frame.to_ndarray()))

    video_tensor = torch.stack(frames)
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    return video_tensor


def tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor, tolerance: float = 0.01) -> bool:
    """Compare two tensors with a tolerance.

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


class TestFastseek(unittest.TestCase):
    """Test fastseek functionality."""

    def test_fastseek_mp4(self):
        """Test fastseek."""
        fastseek = Fastseek(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))
        print(fastseek.keyframes_by_frames)
        assert list(fastseek.keyframes_by_frames.keys()) == [1]
        assert fastseek.keyframes_by_frames[1][0] == [
            KeyframeInfo(index=0, pts=0),
            KeyframeInfo(index=250, pts=128000),
            KeyframeInfo(index=500, pts=256000),
            KeyframeInfo(index=750, pts=384000),
            KeyframeInfo(index=1000, pts=512000),
            KeyframeInfo(index=1250, pts=640000),
            KeyframeInfo(index=1500, pts=768000),
            KeyframeInfo(index=1750, pts=896000),
        ]
        assert fastseek.keyframe_pts == {
            1: [0, 128000, 256000, 384000, 512000, 640000, 768000, 896000]
        }
        assert fastseek.frames_supported
        assert fastseek.mime == "video/mp4"

        assert fastseek.should_seek(0, 0) is None
        assert fastseek.should_seek(0, 249) is None
        assert fastseek.should_seek(0, 250) == KeyframeInfo(index=250, pts=128000)
        assert fastseek.should_seek(0, 251) == KeyframeInfo(index=250, pts=128000)
        assert fastseek.should_seek(0, 499) == KeyframeInfo(index=250, pts=128000)
        assert fastseek.should_seek(499, 500) == KeyframeInfo(index=500, pts=256000)
        assert fastseek.should_seek(250, 499) is None
        assert fastseek.should_seek(250, 500) == KeyframeInfo(index=500, pts=256000)

        assert fastseek.get_next_keyframe_pts(0) == 128000, (
            f"Expected 128000, got {fastseek.get_next_keyframe_pts(0)}"
        )
        assert fastseek.get_next_keyframe_pts(127999) == 128000
        assert fastseek.get_next_keyframe_pts(128000) == 256000

    def test_fastseek_mkv(self):
        """Test fastseek."""
        fastseek = Fastseek(io.BytesIO(Path("tests/data/sync_test.mkv").read_bytes()))
        print(fastseek.keyframe_pts)
        assert list(fastseek.keyframe_pts.keys()) == [1]
        assert fastseek.keyframe_pts[1] == [0, 8354, 16688, 25021, 33354, 41688, 50021, 58354]
        assert not fastseek.frames_supported
        assert fastseek.mime == "video/x-matroska"

        assert fastseek.get_next_keyframe_pts(0) == 8354
        assert fastseek.get_next_keyframe_pts(8353) == 8354
        assert fastseek.get_next_keyframe_pts(8354) == 16688


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
        av_decoder = AVDecoder(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))
        av_data = av_decoder.get_frames()
        video_tensor = av_data.video_clips[0]

        print(video_tensor.shape)
        assert (video_tensor == self.complete_video_tensor).all(), (
            "Energon decoded video does not match baseline"
        )

    def test_verify_video_decode(self):
        """Verify the video decode matches the baseline."""
        av_decoder = AVDecoder(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))
        all_timestamps = []
        for frame in [*range(5), *range(245, 255), *range(1881, 1891)]:
            # print(f"Loading frame {frame}")
            video_data, timestamps = av_decoder.get_video_clips(
                video_clip_ranges=[(frame, frame)], video_unit="frames"
            )
            assert len(video_data) == 1
            assert video_data[0].shape == (1, 3, 108, 192), (
                f"Shape of frame {frame} is {video_data[0].shape}"
            )
            assert (video_data[0] == self.complete_video_tensor[frame : frame + 1]).all()
            # print(f"Timestamp for frame {frame}: {timestamps[0]}")
            all_timestamps.append(0.5 * (timestamps[0][0] + timestamps[0][1]))

        for frame, timestamp1, timestamp2 in zip(
            [*range(5), *range(245, 255), *range(1881, 1891)], all_timestamps, all_timestamps[1:] + [float("inf")]
        ):
            if frame in (4, 254):
                continue
            # print(f"Loading frame {frame}")
            video_data, timestamps = av_decoder.get_video_clips(
                video_clip_ranges=[(timestamp1, timestamp1)], video_unit="seconds"
            )
            assert len(video_data) == 1
            assert video_data[0].shape == (1, 3, 108, 192), (
                f"Shape of frame {frame} is {video_data[0].shape}"
            )
            assert (video_data[0] == self.complete_video_tensor[frame : frame + 1]).all()
            assert 0.5 * (timestamps[0][0] + timestamps[0][1]) == timestamp1, (
                f"Timestamp for frame {frame} is {timestamps[0][0]} + {timestamps[0][1]}"
            )

            video_data, timestamps = av_decoder.get_video_clips(
                video_clip_ranges=[(timestamp1, timestamp2)], video_unit="seconds"
            )
            assert len(video_data) == 1
            if frame == 1890:
                assert video_data[0].shape == (1, 3, 108, 192), (
                    f"Shape of frame {frame} is {video_data[0].shape}"
                )
                assert (video_data[0] == self.complete_video_tensor[frame : frame + 1]).all()
            else:
                assert video_data[0].shape == (2, 3, 108, 192), (
                    f"Shape of frame {frame} is {video_data[0].shape}"
                )
                assert (video_data[0] == self.complete_video_tensor[frame : frame + 2]).all()

    def test_decode_metadata(self):
        """Test decoding metadata."""
        expected_metadata = [
            AVMetadata(
                video_duration=63.054,
                video_num_frames=1891,
                video_fps=30.0,
                video_width=192,
                video_height=108,
                audio_duration=63.103,
                audio_channels=2,
                audio_sample_rate=48000,
            ),
            AVMetadata(
                video_duration=63.03333333333333,
                video_num_frames=1891,
                video_fps=30.0,
                video_width=192,
                video_height=108,
                audio_duration=63.068,
                audio_channels=2,
                audio_sample_rate=48000,
            ),
        ]
        for video_file, expected_metadata in zip(
            ["tests/data/sync_test.mkv", "tests/data/sync_test.mp4"], expected_metadata
        ):
            av_decoder = AVDecoder(io.BytesIO(Path(video_file).read_bytes()))
            assert av_decoder.get_metadata() == expected_metadata, (
                f"Metadata does not match expected metadata for {video_file}"
            )

            assert av_decoder.get_video_duration(get_frame_count=False) in (
                (expected_metadata.video_duration, None),
                (expected_metadata.video_duration, expected_metadata.video_num_frames),
            )
            assert av_decoder.get_video_duration(get_frame_count=True) == (
                expected_metadata.video_duration,
                expected_metadata.video_num_frames,
            )

            assert av_decoder.get_audio_duration() == expected_metadata.audio_duration
            assert av_decoder.get_video_fps() == expected_metadata.video_fps
            assert av_decoder.get_audio_samples_per_second() == expected_metadata.audio_sample_rate

    def test_decode_strided_resized(self):
        """Test decoding a subset of frames with resizing."""
        for video_file in ["tests/data/sync_test.mkv", "tests/data/sync_test.mp4"]:
            print(f"================= Testing {video_file} ==================")
            av_decoder = AVDecoder(io.BytesIO(Path(video_file).read_bytes()))

            video_tensor = get_single_frames_uniform(
                av_decoder=av_decoder,
                num_frames=64,
                video_out_frame_size=(224, 224),
            )

            # Get strided frames from baseline complete video tensor
            strided_baseline_tensor = self.complete_video_tensor[
                np.linspace(0, self.complete_video_tensor.shape[0] - 1, 64, dtype=int).tolist()
            ]
            # Now resize the baseline frames
            resize = transforms.Resize((224, 224))
            strided_resized_baseline_tensor = resize(strided_baseline_tensor)

            # We allow small numerical differences due to different resize implementations
            assert tensors_close(video_tensor, strided_resized_baseline_tensor, tolerance=0.01), (
                "Energon decoded video does not match baseline"
            )

    def test_time_precision(self):
        """Test decoding video frames with time precision."""
        av_decoder = AVDecoder(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))
        video_data, timestamps = av_decoder.get_video_clips(
            video_clip_ranges=[
                (4 + 1 / 30, 4 + 1 / 30),
                (4 + 1 / 30 + 1 / 60, 4 + 1 / 30 + 1 / 60),
            ],
            video_unit="seconds",
        )
        assert (timestamps[0][0] == 4 + 1 / 30) and (timestamps[0][1] == 4 + 2 / 30), (
            f"Timestamp for frame 0 is {timestamps[0][0]} and {timestamps[0][1]}"
        )
        assert (timestamps[1][0] == 4 + 1 / 30) and (timestamps[1][1] == 4 + 2 / 30), (
            f"Timestamp for frame 0 is {timestamps[1][0]} and {timestamps[1][1]}"
        )
        # from PIL import Image

        # Image.fromarray(video_data[0][0, :, 18:55, 18:55].numpy().transpose(1, 2, 0)).save(
        #     "circ.png"
        # )
        assert (video_data[0][0, :, 18:55, 18:55] > 250).all(), (
            "First extracted frame is not all white in the area (18, 18, 55, 55)"
        )

        av_decoder = AVDecoder(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))
        video_data, timestamps = av_decoder.get_video_clips(
            video_clip_ranges=[(4 * 30 + 1, 4 * 30 + 1), (4 * 30 + 1, 4 * 30 + 1)],
            video_unit="frames",
        )
        assert (timestamps[0][0] == 4 + 1 / 30) and (timestamps[0][1] == 4 + 2 / 30), (
            f"Timestamp for frame 0 is {timestamps[0][0]} and {timestamps[0][1]}"
        )
        assert (timestamps[1][0] == 4 + 1 / 30) and (timestamps[1][1] == 4 + 2 / 30), (
            f"Timestamp for frame 0 is {timestamps[1][0]} and {timestamps[1][1]}"
        )
        from PIL import Image

        Image.fromarray(video_data[0][0, :, 18:55, 18:55].numpy().transpose(1, 2, 0)).save(
            "circ.png"
        )
        assert (video_data[0][0, :, 18:55, 18:55] > 250).all(), (
            "First extracted frame is not all white in the area (18, 18, 55, 55)"
        )

    def test_video_audio_sync(self):
        """Test decoding video frames and audio clips together."""
        av_decoder = AVDecoder(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))

        # Extract a single frame every 2 seconds and an audio clip (0.05 seconds long) at the same time.
        # We extract the frames from the sync video that shows the full white circle on the left,
        # when the click sound occurs.
        # Note that the click sound is actually off by 0.022 secs in the original video,
        # I verified this in Davinci Resolve.
        av_data = av_decoder.get_clips(
            video_clip_ranges=[(a * 2 + 1 / 30, a * 2 + 1 / 30) for a in range(65)],
            audio_clip_ranges=[(a * 2 + 1 / 30, a * 2 + 1 / 30 + 0.05) for a in range(65)],
            video_unit="seconds",
            audio_unit="seconds",
            video_out_frame_size=None,
        )

        # We drop the first two extracted frames because the click sequence hasn't started yet
        video_clips = av_data.video_clips[2:]
        audio_clips = av_data.audio_clips[2:]
        # Then we check that the first extracted frame is all white in the area (18, 18, 55, 55)
        # from PIL import Image

        # Image.fromarray(video_clips[0][0, :, 18:55, 18:55].numpy().transpose(1, 2, 0)).save(
        #     "circ.png"
        # )
        assert (video_clips[0][0, :, 18:55, 18:55] > 250).all(), (
            "First extracted frame is not all white in the area (18, 18, 55, 55)"
        )

        # Check that all the video frames are the same (close value)
        for video_clip in video_clips:
            assert tensors_close(video_clip, video_clips[0], tolerance=0.01), (
                "All video frames are not the same"
            )

        # Check that the first audio clip has the click sound
        assert (audio_clips[0] > 0.5).any(), "Audio click not found"

        # Check that all the audio clips are the same (close value)
        for audio_clip in audio_clips:
            assert tensors_close(audio_clip, audio_clips[0], tolerance=0.01), (
                "All audio clips are not the same"
            )

    def test_pickle_decoder(self):
        """Test AVDecoder on a video file can be pickled and unpickled."""
        av_decoder = AVDecoder(io.BytesIO(Path("tests/data/sync_test.mp4").read_bytes()))

        # Get metadata from original decoder
        original_metadata = av_decoder.get_metadata()

        # Pickle the decoder
        pickled_data = pickle.dumps(av_decoder)

        # Unpickle the decoder
        unpickled_decoder = pickle.loads(pickled_data)

        # Verify metadata matches
        unpickled_metadata = unpickled_decoder.get_metadata()
        assert unpickled_metadata == original_metadata, (
            f"Unpickled metadata {unpickled_metadata} does not match original {original_metadata}"
        )

        # Verify we can still decode frames from the unpickled decoder
        video_tensor = get_single_frames_uniform(
            av_decoder=unpickled_decoder,
            num_frames=16,
            video_out_frame_size=(64, 64),
        )

        # Check that we got the expected shape
        assert video_tensor.shape == (16, 3, 64, 64), (
            f"Expected shape (16, 3, 64, 64), got {video_tensor.shape}"
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
        av_data = av_decoder.get_audio()
        audio_tensor = av_data.audio_clips[0]

        assert (audio_tensor == self.complete_audio_tensor).all(), (
            "Energon decoded audio does not match baseline"
        )

    def test_decode_clips(self):
        """Test decoding multiple clips from an audio file."""
        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

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

    def test_decode_wav_same_shape(self):
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
            av_decoder=av_decoder,
            num_clips=10,
            clip_duration_seconds=0.9954783485892385,
            request_audio=True,
        )
        audio_sps = av_decoder.get_audio_samples_per_second()

        print(f"SPS: {audio_sps}")
        for audio_tensor in av_data.audio_clips:
            print(audio_tensor.shape)

        assert all(
            audio_tensor.shape == av_data.audio_clips[0].shape
            for audio_tensor in av_data.audio_clips
        ), "Audio clips have different shapes"

    def test_wav_decode_against_soundfile(self):
        """Test decoding a WAV file against the soundfile library."""

        try:
            import soundfile
        except ImportError:
            self.skipTest("soundfile library not found")

        with open("tests/data/test_audio.wav", "rb") as f:
            raw_bytes = f.read()
            stream = io.BytesIO(raw_bytes)

        av_decoder = AVDecoder(stream)
        av_data = av_decoder.get_clips(audio_clip_ranges=[(0, float("inf"))], audio_unit="samples")
        audio_tensor = av_data.audio_clips[0]

        # Load the same audio file using soundfile

        audio_data, _ = soundfile.read("tests/data/test_audio.wav", dtype="int16")
        audio_tensor_soundfile = torch.from_numpy(audio_data).transpose(0, 1)

        # Check that the two tensors are close
        assert tensors_close(audio_tensor, audio_tensor_soundfile, tolerance=0.01), (
            "Energon decoded audio does not match baseline"
        )

        # Now check partial extraction in the middle of the audio
        av_data = av_decoder.get_clips(audio_clip_ranges=[(0.5, 1.0)], audio_unit="seconds")
        audio_tensor = av_data.audio_clips[0]
        audio_sps = av_decoder.get_audio_samples_per_second()
        audio_tensor_soundfile = torch.from_numpy(
            audio_data[int(0.5 * audio_sps) : int(1.0 * audio_sps)]
        ).transpose(0, 1)

        # Check that the two tensors are close
        assert tensors_close(audio_tensor, audio_tensor_soundfile, tolerance=0.01), (
            "Energon decoded audio does not match baseline"
        )

        # Now compare the speed of the two implementations by repeatedly decoding the same audio
        num_trials = 100

        start_time = time.perf_counter()
        for _ in range(num_trials):
            av_data = av_decoder.get_clips(
                audio_clip_ranges=[(0, float("inf"))], audio_unit="samples"
            )
            audio_tensor = av_data.audio_clips[0]
        end_time = time.perf_counter()
        print(f"AVDecoder time: {end_time - start_time} seconds")

        # Now do the same with soundfile
        start_time = time.perf_counter()
        for _ in range(num_trials):
            audio_data, _ = soundfile.read("tests/data/test_audio.wav", dtype="int16")
            audio_tensor_soundfile = torch.from_numpy(audio_data).transpose(0, 1)
        end_time = time.perf_counter()
        print(f"Soundfile time: {end_time - start_time} seconds")

        start_time = time.perf_counter()
        for _ in range(num_trials):
            av_data = av_decoder.get_clips(
                audio_clip_ranges=[(0, float("inf"))], audio_unit="samples"
            )
            audio_tensor = av_data.audio_clips[0]
        end_time = time.perf_counter()
        print(f"AVDecoder time: {end_time - start_time} seconds")

        # Now do the same with soundfile
        start_time = time.perf_counter()
        for _ in range(num_trials):
            audio_data, _ = soundfile.read("tests/data/test_audio.wav", dtype="int16")
            audio_tensor_soundfile = torch.from_numpy(audio_data).transpose(0, 1)
        end_time = time.perf_counter()
        print(f"Soundfile time: {end_time - start_time} seconds")

    def test_decode_metadata(self):
        """Test decoding metadata."""
        expected_metadata = [
            AVMetadata(
                audio_duration=10.0,
                audio_channels=1,
                audio_sample_rate=32000,
            ),
            AVMetadata(
                audio_duration=12.782585034013605,
                audio_channels=2,
                audio_sample_rate=44100,
            ),
        ]
        for audio_file, expected_metadata in zip(
            ["tests/data/test_audio.flac", "tests/data/test_audio.wav"], expected_metadata
        ):
            av_decoder = AVDecoder(io.BytesIO(Path(audio_file).read_bytes()))
            assert av_decoder.get_metadata() == expected_metadata, (
                f"Metadata does not match expected metadata for {audio_file}: {av_decoder.get_metadata()}"
            )

            assert av_decoder.get_audio_duration() == expected_metadata.audio_duration
            assert av_decoder.get_audio_samples_per_second() == expected_metadata.audio_sample_rate


if __name__ == "__main__":
    unittest.main()
