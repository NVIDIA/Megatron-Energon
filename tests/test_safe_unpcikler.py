# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io
import pickle
import pytest

from collections import OrderedDict

from megatron.core.safe_globals import SafeUnpickler


class UnsafeClass:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"UnsafeClass(value={self.value})"


class TestSafeUnpickler:
    def test_safe_types(self):
        data = {"key": [1, 2.0, True, "s"], "od": OrderedDict(a=1)}
        raw = pickle.dumps(data)
        result = SafeUnpickler(io.BytesIO(raw)).load()
        assert result == data

    def test_unsafe_types(self):
        raw = pickle.dumps(UnsafeClass(123))
        with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
	        SafeUnpickler(io.BytesIO(raw)).load()
