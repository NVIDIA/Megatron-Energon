# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause


import pickle

import numpy as np
import torch

from megatron.energon.safe_globals import register_safe_globals


class UnsafeClass:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"UnsafeClass(value={self.value})"


class TestSafeGlobals:
    def test_safe_globals(self, tmp_path_dist_ckpt):
        register_safe_globals()

        # create dummy checkpoint
        ckpt_path = tmp_path_dist_ckpt / "test_safe_globals.pt"
        dummy_obj = np.array([1, 2, 3])
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(dummy_obj, ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        torch.load(ckpt_path)

    def test_unsafe_globals(self, tmp_path_dist_ckpt):
        register_safe_globals()

        # create dummy checkpoint
        ckpt_path = tmp_path_dist_ckpt / "test_safe_globals.pt"
        dummy_obj = UnsafeClass(123)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(dummy_obj, ckpt_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # expected error
        try:
            torch.load(ckpt_path)
        except Exception as e:
            assert e == pickle.UnpicklingError

        # add class to safe globals
        torch.serialization.add_safe_globals([UnsafeClass])
        torch.load(ckpt_path)