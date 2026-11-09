# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import pickle

import numpy as np

from megatron.energon.safe_unpickler import SafeUnpickler


class UnsafeClass:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"UnsafeClass(value={self.value})"


class TestSafeUnpickler:
    def test_safe_types(self):
        data = np.array([1,2,3])
        raw = pickle.dumps(data)
        result = SafeUnpickler(io.BytesIO(raw)).load()
        assert result == data

    def test_unsafe_types(self):
        raw = pickle.dumps(UnsafeClass(123))
        try:
	        SafeUnpickler(io.BytesIO(raw)).load()
        except Exception as e:
            assert e == pickle.UnpicklingError
