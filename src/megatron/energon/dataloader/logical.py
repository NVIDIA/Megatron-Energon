# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.energon.dataloader.dataloader import GroupedBatchCheckpointConfig

from megatron.energon.dataloader.workers.base_worker import WorkerState


@dataclass(frozen=True, kw_only=True)
class LogicalCheckpoint:
    """Consumed progress for one logical stream at a synchronized delivery boundary."""

    logical_worker_id: int
    logical_workers: int
    iteration_index: int
    state: WorkerState | None
    observed_eof: bool
    micro_batch_size: int | None
    grouped_batch_config: "GroupedBatchCheckpointConfig | None" = None
