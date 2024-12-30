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

from megatron.energon.flavors.webdataset.decode_video_frames import decode_video_frames

def load_video_to_tensor(video_path):
    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        frame = frame.reformat(format="rgb24")
        frames.append(torch.from_numpy(frame.to_ndarray()))

    video_tensor = torch.stack(frames)
    return video_tensor

class TestVideoDecode(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.decode_baseline_video_pyav()

    def tearDown(self):
        pass

    def decode_baseline_video_pyav(self):
        self.complete_video_tensor = load_video_to_tensor("tests/data/sync_test.mp4")

    def test_decode_all_frames(self):

        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()

        # Decode using fastseek Energon wrapper
        video_tensor, _, _ = decode_video_frames(data = raw_bytes)

        assert (video_tensor == self.complete_video_tensor).all(), \
            "Energon decoded video does not match baseline"

    def test_decode_strided_resized(self):
        with open("tests/data/sync_test.mp4", "rb") as f:
            raw_bytes = f.read()

        # Decode using fastseek Energon wrapper
        video_tensor, _, _ = decode_video_frames(
            data = raw_bytes,
            num_frames = 64,
            out_frame_size = (224, 224),
        )

        # get strided frames from baseline complete video tensor
        # this is a little idionsyncratic as Energon does this the same way
        strided_baseline_tensor = self.complete_video_tensor[
            np.linspace(0, self.complete_video_tensor.shape[0] - 1, 64, dtype=int).tolist()
        ]
        # now resize the baseline frames
        resize = transforms.Resize((224, 224))
        strided_baseline_tensor = strided_baseline_tensor.permute(0, 3, 1, 2) # b, h, w, c -> b, c, h, w
        strided_resized_baseline_tensor = resize(strided_baseline_tensor)
        strided_resized_baseline_tensor = strided_resized_baseline_tensor.permute(0, 2, 3, 1) # b, c, h, w -> b, h, w, c

        def are_resized_frames_close(tensor1, tensor2, tolerance=0.01):
            if tensor1.shape != tensor2.shape:
                raise ValueError("Input tensors must have the same shape.")
            tensor1 = tensor1.float() / 255.0
            tensor2 = tensor2.float() / 255.0
            # Compute Mean Absolute Error
            mae = torch.mean(torch.abs(tensor1 - tensor2)).item()
            return mae <= tolerance

        # we allow small numerical differences due to different resize implementations
        assert are_resized_frames_close(video_tensor, strided_resized_baseline_tensor, tolerance=0.01), \
            "Energon decoded video does not match baseline"

if __name__ == "__main__":

    unittest.main()