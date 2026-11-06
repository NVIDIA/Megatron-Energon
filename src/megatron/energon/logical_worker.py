# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LogicalWorkerAssignment:
    """How a physical global worker maps onto logical dataset partitions."""

    #: Primary logical global worker id for this physical worker.
    logical_global_worker_id: int
    #: Subworker index within the logical stream.
    stride_offset: int = 0
    #: Stride between owned output ordinals.
    stride: int = 1
