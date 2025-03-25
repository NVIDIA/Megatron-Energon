from typing import Optional

import numpy as np

from megatron.energon.av.av_decoder import AVData, AVDecoder


def get_clips_uniform(
    av_decoder: AVDecoder,
    clip_duration_seconds: float,
    num_clips: int,
    request_video: bool = False,
    request_audio: bool = False,
    video_out_frame_size: Optional[tuple[int, int]] = None,
) -> AVData:
    """Generates a sequence of (start, end) clips, such that each clip is of
    equal duration and the clips are equidistant from each other.

    Args:
        av_decoder: An AVDecoder instance.
        clip_duration_seconds: The duration of each clip in seconds.
        num_clips: The number of clips to extract.
        request_video: Whether to request video clips.
        request_audio: Whether to request audio clips.
        video_out_frame_size: The size of the video frames to output, or None to use the original size.

    Returns:
        An AVData object containing the extracted video and audio clips.
    """

    if not request_video and not request_audio:
        raise ValueError("You must request at least one of video or audio")

    total_duration, _ = av_decoder.get_duration()

    last_start_time = total_duration - clip_duration_seconds
    clips = [
        (start_time, start_time + clip_duration_seconds)
        for start_time in np.linspace(0, last_start_time, num_clips)
    ]

    return av_decoder.get_clips(
        video_clip_ranges=clips if request_video else None,
        audio_clip_ranges=clips if request_audio else None,
        video_unit="seconds",
        audio_unit="seconds",
        video_out_frame_size=video_out_frame_size,
    )


def get_single_frames_uniform(
    av_decoder: AVDecoder,
    num_frames: int,
    video_out_frame_size: Optional[tuple[int, int]] = None,
) -> AVData:
    """Generates a sequence of (start, -1) clips, such that each clip contains
    only a single frame and the clips are equidistant from each other.

    Args:
        av_decoder: An AVDecoder instance.
        num_frames: The number of frames to extract.
        video_out_frame_size: The size of the video frames to output, or None to use the original size.

    Returns:
        An AVData object containing the extracted images.
    """

    return get_clips_uniform(
        av_decoder=av_decoder,
        clip_duration_seconds=0,
        num_clips=num_frames,
        request_video=True,
        request_audio=False,
        video_out_frame_size=video_out_frame_size,
    )
