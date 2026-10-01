# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import io

import numpy
import torch

from megatron.energon.flavors.webdataset.sample_loader import SliceState
from megatron.energon.rng import SystemRngState
from megatron.energon.savable_loader import (
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
)
from megatron.energon.state import FlexState

SAFE_GLOBALS = [
    SavableDataLoaderState,
    SavableDatasetCheckpoint,
    SavableDatasetState,
    SliceState,
    SystemRngState,
    FlexState,
    numpy._core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
    numpy.dtypes.UInt32DType,
]


def register_safe_globals():
    """Register megatron-energon safe classes with torch serialization."""
    for cls in SAFE_GLOBALS:
        torch.serialization.add_safe_globals([cls])


def safe_load_from_bytes(b):
    """Safe version (weights_only=True) of `torch.storage._load_from_bytes`."""
    return torch.load(io.BytesIO(b), weights_only=True)
