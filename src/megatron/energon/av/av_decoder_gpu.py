# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from typing import BinaryIO, Literal, Optional, Sequence

import torch

# Try importing optional dependencies
try:
    import PyNvVideoCodec as nvc

    AV_GPU_DECODE_AVAILABLE = True
except ImportError as e:
    AV_GPU_DECODE_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)

from torch.nn.functional import interpolate

from megatron.energon.av.av_decoder import AVDecoder


class AVDecoderGpu(AVDecoder):
    """A class that provides a flexible interface for decoding audio and video data.

    This class allows users to control decoding parameters at runtime rather than having them fixed
    during initialization. It's particularly useful for cases where different samples may need different
    decoding parameters.
    """

    stream: BinaryIO
    suppress_warnings: bool
    device_id: int

    def __init__(self, stream: BinaryIO, suppress_warnings: bool = False, device_id=0) -> None:
        super().__init__(stream, suppress_warnings)
        if not AV_GPU_DECODE_AVAILABLE:
            raise ImportError(
                f"GPU video decoding is not available. Please install the required dependencies with:\n"
                f"pip install megatron-energon[av_decode_gpu]\n"
                f"Missing dependency: {MISSING_DEPENDENCY}. Install megatron-energon[av_decode_gpu] to use AVDecoderGpu."
            )
        self.device_id = device_id

    def get_video_clips(
        self,
        video_clip_ranges: Sequence[tuple[float, float]],
        video_unit: Literal["frames", "seconds"] = "seconds",
        video_out_frame_size: Optional[tuple[int, int]] = None,
    ) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:

        assert video_unit in ("frames", "seconds")

        try:
            decoder = nvc.SimpleDecoder(
                self.stream,
                gpu_id=self.device_id,
                use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGB,
            )
        except Exception as e:
            if not self.suppress_warnings:
                warnings.warn(f"GPU decode failed, falling back to CPU: {e}")

            self.stream.seek(0)

            return super().get_video_clips(video_clip_ranges, video_unit, video_out_frame_size)

        average_fps = decoder.get_stream_metadata().average_fps
        last_frame = len(decoder) - 1
        if video_unit == "seconds":
            video_clip_ranges = [
                (
                  range_start * average_fps if range_start != float("inf") else last_frame,
                  range_end * average_fps if range_end != float("inf") else last_frame
                )
                for range_start, range_end in video_clip_ranges
            ]
        elif video_unit == "frames":
            video_clip_ranges = [
              (
                range_start if range_start != float("inf") else last_frame,
                range_end if range_end != float("inf") else last_frame,

              )
                for range_start, range_end in video_clip_ranges
            ]

        # NOTE the CPU decode path silently drops out-of-range frames, this filter matches that behavior
        video_clip_ranges = [
            (range_start, min(range_end, last_frame))
            for range_start, range_end in video_clip_ranges
            if range_start <= last_frame
        ]
        video_clips_frames: list[list[torch.Tensor]] = []
        video_clips_timestamps: list[tuple[float, float]] = []
        for video_clip_range in video_clip_ranges:
            range_start, range_end = video_clip_range
            decoded_frames = decoder.get_batch_frames_by_index(
              list(range(int(range_start), int(range_end) + 1))
            )

            # NOTE PyNVC does not currently timestamp decoded frames reliably, so we assume constant framerate instead
            clip_timestamp_start = float(range_start) / float(average_fps)
            clip_timestamp_end = float(range_end + 1) / float(
                average_fps
            )  # range_end + 1 accounts for last frame duration

            video_clips_timestamps.append((clip_timestamp_start, clip_timestamp_end))
            video_clips_frames.append([torch.from_dlpack(frame).clone() for frame in decoded_frames])

        out_video_clips = [
            torch.stack(clip_frames).permute((0, 3, 1, 2)) for clip_frames in video_clips_frames
        ]

        if video_out_frame_size is not None:  # NOTE float/byte conversion, is it OK?
            out_video_clips = [
                interpolate(
                    clip.float(),
                    size=(video_out_frame_size[1], video_out_frame_size[0]),
                    mode="bilinear",
                    align_corners=False,
                ).byte()
                for clip in out_video_clips
            ]

        return out_video_clips, video_clips_timestamps
