# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Dict, Generic, Iterator, Optional, Sequence, Tuple, TypeVar, Union

import torch
import torch.distributed as dist

from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import State
from megatron.energon.savable_loader import (
    BasicDataLoader,
    SavableDataLoader,
    SavableDataLoaderState,
)
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.base import get_sample_restore_key

__all__ = ["StopFirstLoader", "StopFirstDataLoaderState"]


T = TypeVar("T")


@edataclass
class StopFirstDataLoaderState(State):
    inner_state: SavableDataLoaderState | None
    iterating_from_start: bool
    next_sample_restore_key: Tuple[Union[str, int, tuple], ...] | None

    def __repr__(self):
        return f"StopFirstDataLoaderState(inner_state={self.inner_state!r}, iterating_from_start={self.iterating_from_start!r}, next_sample_restore_key={self.next_sample_restore_key!r})"


class StopFirstLoader(Generic[T]):
    """
    A loader that stops as soon as the first rank is exhausted.
    If continuing a second time, it will restart the previously exhausted rank and iterate until the next rank is
    exhausted, restarting all ranks once.

    This is useful for trainings where the dataset is not repeated.
    """

    loader: SavableDataLoader | BasicDataLoader
    worker_config: WorkerConfig
    distributed_device: str
    _iterator: Iterator[T] | None = None
    _iterating_from_start: bool = True
    _next_sample: T | None = None
    _next_sample_restore_key: Tuple[Union[str, int, tuple], ...] | None = None

    def __init__(self, loader: SavableDataLoader | BasicDataLoader):
        self.loader = loader
        self.worker_config = loader.worker_config
        self.distributed_device = (
            "cuda"
            if dist.is_available()
            and dist.is_initialized()
            and dist.get_backend() == dist.Backend.NCCL
            else "cpu"
        )

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.loader)

        # Check if torch distributed is using cuda
        flag = torch.zeros(
            1, dtype=torch.uint8, device=self.distributed_device, requires_grad=False
        )

        # If there is a pending sample, use it
        if self._next_sample is not None:
            sample = self._next_sample
            local_has_sample = 1
            self._next_sample = None
            self._next_sample_restore_key = None
            # print(f"[r={self.worker_config.rank}]: Using pending sample\n", end="")
        elif self._next_sample_restore_key is not None:
            sample = self.restore_sample(self._next_sample_restore_key)
            self._next_sample = None
            self._next_sample_restore_key = None
            local_has_sample = 1
            # print(f"[r={self.worker_config.rank}]: Using restored pending sample\n", end="")
        else:
            sample = None
            local_has_sample = 0
            # print(f"[r={self.worker_config.rank}]: No pending sample\n", end="")

        while True:
            if not local_has_sample:
                try:
                    sample = next(self._iterator)
                    local_has_sample = 1
                except StopIteration:
                    if not self._iterating_from_start:
                        # If not iterating from start (i.e. another rank already exhausted and ended the epoch),
                        # The second epoch should ignore ending iterators and continue iterating.
                        self._iterator = iter(self.loader)
                        self._iterating_from_start = True
                        # print(f"[r={self.worker_config.rank}]: Restarting iterator\n", end="")
                        continue
                    # print(f"[r={self.worker_config.rank}]: No samples left\n", end="")
                    local_has_sample = 0

            flag.fill_(local_has_sample)
            # Compute *global* logical *AND* over all ranks.  Using MIN as this
            # is equivalent to logical AND for 0/1 bits.
            dist.all_reduce(
                flag, op=dist.ReduceOp.MIN, group=self.worker_config.data_parallel_group
            )
            global_all_have_sample = bool(flag.item())

            if not global_all_have_sample:
                if local_has_sample == 0:
                    self._next_sample = sample
                else:
                    self._next_sample = None
                # At least one rank is exhausted – terminate *all* ranks.  We
                # purposely ignore any *local* sample obtained in this round
                # to keep the step count aligned across ranks.
                # The rank(s) which ended is iterating from start again, the other ranks are not.
                self._iterating_from_start = local_has_sample == 0
                if local_has_sample == 0:
                    self._iterator = None
                # print(f"[r={self.worker_config.rank}]: All exhausted, iter_from_start={self._iterating_from_start}\n", end="")
                break

            # Otherwise every rank had a sample in this step – yield it.
            yield sample
            local_has_sample = 0
            sample = None

    def save_state_rank(self) -> StopFirstDataLoaderState:
        assert isinstance(self.loader, SavableDataLoader)
        if self._next_sample_restore_key is not None:
            restore_key = self._next_sample_restore_key
        elif self._next_sample is not None:
            restore_key = self._next_sample_restore_key = get_sample_restore_key(self._next_sample)
        else:
            restore_key = None
        return StopFirstDataLoaderState(
            inner_state=self.loader.save_state_rank(),
            iterating_from_start=self._iterating_from_start,
            next_sample_restore_key=restore_key,
        )

    def restore_state_rank(self, state: StopFirstDataLoaderState) -> None:
        assert isinstance(self.loader, SavableDataLoader)
        self._iterating_from_start = state.iterating_from_start
        self._next_sample_restore_key = state.next_sample_restore_key
        self.loader.restore_state_rank(state.inner_state)

    def can_restore_sample(self) -> bool:
        return self.loader.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        return self.loader.restore_sample(restore_key)

    def save_state_global(
        self, global_dst_rank: int
    ) -> Optional[Sequence[StopFirstDataLoaderState]]:
        """
        See :meth:`megatron.energon.SavableDataLoader.save_state_global`
        """
        # Fetch current rank's worker's state
        merged_state = self.save_state_rank()

        # Gather the merged states
        if self.worker_config.world_size > 1:
            output: Optional[Sequence[StopFirstDataLoaderState]]
            if self.worker_config.global_rank() == global_dst_rank:
                output = [None] * self.worker_config.world_size
            else:
                # Check if the global_dst_rank is in the same group at all
                if self.worker_config.data_parallel_group is not None:
                    try:
                        _ = torch.distributed.get_group_rank(
                            self.worker_config.data_parallel_group, global_dst_rank
                        )
                    except RuntimeError:
                        raise ValueError(
                            f"global_dst_rank {global_dst_rank} is not in the group of the current rank's worker config"
                        )

                output = None

            torch.distributed.gather_object(
                merged_state,
                output,
                global_dst_rank,
                group=self.worker_config.data_parallel_group,
            )

            return output
        else:
            # Not distributed -> return the merged state
            return [merged_state]

    def restore_state_global(
        self,
        state: Optional[Sequence[StopFirstDataLoaderState]],
        *,
        src_rank: Optional[int] = None,
    ) -> None:
        """
        See :meth:`megatron.energon.SavableDataLoader.restore_state_global`
        """
        assert self._iterator is None, "Cannot restore state while workers are running"

        if src_rank is None or self.worker_config.world_size == 1:
            assert isinstance(state, list), "State must be a list in distributed setup"
            assert len(state) == self.worker_config.world_size, (
                "State must be a list of size world_size"
            )

            rank_state = state[self.worker_config.rank]
        else:
            if self.worker_config.data_parallel_group is not None:
                # Only the src_rank has the state within this dp group
                try:
                    global_src_rank = torch.distributed.get_global_rank(
                        self.worker_config.data_parallel_group, src_rank
                    )
                except RuntimeError:
                    raise ValueError(
                        f"src_rank {src_rank} is not in the group of the current rank's worker config"
                    )
            else:
                # If no DP group is given, we assume the global rank is
                # the same as the data parallel rank
                global_src_rank = src_rank

            if self.worker_config.rank != src_rank:
                # Send the state to all other ranks
                assert state is None
                # Must still be a list of Nones
                state = [None] * self.worker_config.world_size
            else:
                assert isinstance(state, list), "State must be a list in distributed setup"
                assert len(state) == self.worker_config.world_size, (
                    "State must be a list of size world_size"
                )

            local_object = [None]
            torch.distributed.scatter_object_list(
                local_object,
                state,
                src=global_src_rank,
                group=self.worker_config.data_parallel_group,
            )
            rank_state = local_object[0]

        self.restore_state_rank(rank_state)

    def config(self) -> Dict[str, Any]:
        return {
            "type": type(self).__qualname__,
            "loader": self.loader.config(),
        }
