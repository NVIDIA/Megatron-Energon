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

from megatron.energon.flavors.webdataset.decode_av_frames import decode_audio_samples, decode_video_frames, get_clip_indices

def load_video_to_tensor(video_path):
    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        frame = frame.reformat(format="rgb24")
        frames.append(torch.from_numpy(frame.to_ndarray()))

    video_tensor = torch.stack(frames)
    return video_tensor

# class TestVideoDecode(unittest.TestCase):

#     def setUp(self):
#         logging.basicConfig(stream=sys.stderr, level=logging.INFO)
#         self.decode_baseline_video_pyav()

#     def tearDown(self):
#         pass

#     def decode_baseline_video_pyav(self):
#         self.complete_video_tensor = load_video_to_tensor("tests/data/sync_test.mp4")

#     def test_decode_all_frames(self):

#         with open("tests/data/sync_test.mp4", "rb") as f:
#             raw_bytes = f.read()

#         # Decode using fastseek Energon wrapper
#         video_tensor, _, _ = decode_video_frames(data = raw_bytes)

#         assert (video_tensor == self.complete_video_tensor).all(), \
#             "Energon decoded video does not match baseline"

#     def test_decode_strided_resized(self):
#         with open("tests/data/sync_test.mp4", "rb") as f:
#             raw_bytes = f.read()

#         # Decode using fastseek Energon wrapper
#         video_tensor, _, _ = decode_video_frames(
#             data = raw_bytes,
#             num_frames = 64,
#             out_frame_size = (224, 224),
#         )

#         # get strided frames from baseline complete video tensor
#         # this is a little pointless as Energon does this the same way
#         strided_baseline_tensor = self.complete_video_tensor[
#             np.linspace(0, self.complete_video_tensor.shape[0] - 1, 64, dtype=int).tolist()
#         ]
#         # now resize the baseline frames
#         resize = transforms.Resize((224, 224))
#         strided_baseline_tensor = strided_baseline_tensor.permute(0, 3, 1, 2) # b, h, w, c -> b, c, h, w
#         strided_resized_baseline_tensor = resize(strided_baseline_tensor)
#         strided_resized_baseline_tensor = strided_resized_baseline_tensor.permute(0, 2, 3, 1) # b, c, h, w -> b, h, w, c

#         def are_resized_frames_close(tensor1, tensor2, tolerance=0.01):
#             if tensor1.shape != tensor2.shape:
#                 raise ValueError("Input tensors must have the same shape.")
#             tensor1 = tensor1.float() / 255.0
#             tensor2 = tensor2.float() / 255.0
#             # Compute Mean Absolute Error
#             mae = torch.mean(torch.abs(tensor1 - tensor2)).item()
#             return mae <= tolerance

#         # we allow small numerical differences due to different resize implementations
#         assert are_resized_frames_close(video_tensor, strided_resized_baseline_tensor, tolerance=0.01), \
#             "Energon decoded video does not match baseline"

def load_audio_to_tensor(audio_path, target_rate=16000):
    container = av.open(audio_path)
    audio_stream = container.streams.audio[0]

    # Initialize resampler to convert each frame to target_rate
    resampler = av.audio.resampler.AudioResampler(
        format=audio_stream.format,
        layout=audio_stream.layout,
        rate=target_rate
    )

    frames = []

    for frame in container.decode(audio=0):
        resampled_frame = resampler.resample(frame)[0]
        frames.append(torch.from_numpy(resampled_frame.to_ndarray()))

    audio_tensor = torch.cat(frames, 1)
    return audio_tensor


class TestAudioDecode(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.decode_baseline_audio_pyav()

    def tearDown(self):
        pass

    def decode_baseline_audio_pyav(self):
        self.complete_audio_tensor = load_audio_to_tensor("tests/data/test_audio.flac")

    def test_decode_all_samples(self):

        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()

        _, audio_tensor, _ = decode_audio_samples(data = raw_bytes, num_clips = -1)

        assert (audio_tensor == self.complete_audio_tensor).all(), \
            "Energon decoded audio does not match baseline"

    def test_decode_clips(self):
        with open("tests/data/test_audio.flac", "rb") as f:
            raw_bytes = f.read()

        # Decode using fastseek Energon wrapper
        _, audio_tensor, _ = decode_audio_samples(
            data = raw_bytes,
            num_clips = 5,
            clip_duration = 1,
        )

        # get strided frames from baseline complete video tensor
        # this is a little pointless as Energon does this the same way
        # e.g. shape [channels, total_length]
        total_samples = self.complete_audio_tensor.shape[1] - 1
        clip_indices = get_clip_indices(16000, total_samples, 5, 1)

        # now resize the baseline frames
        clips = []
        for indices in clip_indices:
            clips.append(self.complete_audio_tensor[0, indices])
        clips_baseline_tensor = torch.stack(clips)

        # TODO(jbarker): Fix this test, sub-sampling the full tensor should match energon
        assert (audio_tensor == clips_baseline_tensor).all(), \
            "Energon decoded audio does not match baseline"
        # assert audio_tensor.shape == torch.Size([5, 16000]), \
        #     "Energon decoded audio clips have wrong size"

if __name__ == "__main__":

    ## VSCODE DEBUGGER INIT
    import os
    if int(os.environ["RANK"]) == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print(">>>> RANK 0 IS WAITING FOR DEBUGGER...")
        debugpy.wait_for_client()

    unittest.main()