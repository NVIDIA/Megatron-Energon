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

__all__ = ["RedistributeLoader", "RedistributeDataLoaderState"]


T = TypeVar("T")


@edataclass
class RedistributeDataLoaderState(State):
    inner_state: SavableDataLoaderState | None

    exhausted_state: bool
    overuse_count: int

    next_samples_restore_keys: list[Tuple[Union[str, int, tuple], ...]] | None

    def __repr__(self):
        return f"RedistributeLoaderState(inner_state={self.inner_state!r}, exhausted_state={self.exhausted_state!r}, overuse_count={self.overuse_count!r})"


class RedistributeLoader(Generic[T]):
    """
    A loader that wraps the actual loader and redistributes the last samples to the ranks that are not exhausted.
    The last incomplete batch (i.e. where not all ranks have data available) is not iterated.

    It is useful for trainings where the dataset is not repeated.

    Stages:
      First stage: Iterate until one rank is exhausted.
      Second stage: Iterate until all ranks are exhausted and the global batch is incomplete.
        Collect how many samples are required to satisfy the need for a global batch.
        Fetch those additional needed samples from the ranks that have the least overuse count.
        Directly communicate the samples from the overfetched ranks to the exhausted ranks in round robin fashion.
        Distribute the samples to the ranks that are not exhausted.
    If starting a new iterator after global exhaustion, perform another epoch (also emitting the samples from the last
    incomplete batch).
    """
    # large int64 number we'll never reach for overuse counts.
    OVERUSE_COUNT_MAX = 0x1000000000000000

    loader: SavableDataLoader | BasicDataLoader
    worker_config: WorkerConfig
    distributed_device: str

    overuse_counts: torch.Tensor
    exhausted_states: torch.Tensor
    _exhausted_states_list: list[torch.Tensor]

    _iterator: Iterator[T] | None = None

    _next_samples: list[T] | None = None
    _next_sample_restore_keys: list[Tuple[Union[str, int, tuple], ...]] | None = None

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
        self.overuse_counts = torch.zeros(
            self.worker_config.world_size, dtype=torch.int64, requires_grad=False
        )
        self.exhausted_states = torch.zeros(
            self.worker_config.world_size,
            dtype=torch.uint8,
            device=self.distributed_device,
            requires_grad=False,
        )
        self._exhausted_states_list = [
            self.exhausted_states[i] for i in range(self.worker_config.world_size)
        ]

    def _find_ranks_to_oversample(self, needed_samples: int) -> int:
        oversample_self = 0
        while needed_samples > 0:
            min_overuse_idx = torch.where(self.overuse_counts == torch.min(self.overuse_counts))[
                0
            ].cpu()
            # print(f"[r={self.worker_config.rank}]: Min overuse idx: {min_overuse_idx}\n", end="")
            for rank in min_overuse_idx:
                self.overuse_counts[rank] += 1
                if rank == self.worker_config.rank:
                    oversample_self += 1
                needed_samples -= 1
                if needed_samples == 0:
                    break

        return oversample_self

    def _as_global_rank(self, rank: int) -> int:
        if self.worker_config.data_parallel_group is not None:
            return dist.get_global_rank(self.worker_config.data_parallel_group, rank)
        else:
            return rank

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.loader)

        samples: list[T] = []

        if self._next_samples is not None:
            samples.extend(self._next_samples)
            self._next_samples = None
            self._next_sample_restore_keys = None
        elif self._next_sample_restore_keys is not None:
            samples.extend(
                self.restore_sample(restore_key) for restore_key in self._next_sample_restore_keys
            )
            self._next_sample_restore_keys = None

        rank = self.worker_config.rank

        # Ensure the initial state is synchronized (e.g. if restored from a checkpoint)
        dist.all_gather(
            self._exhausted_states_list,
            self.exhausted_states[rank],
            group=self.worker_config.data_parallel_group,
        )
        overuse_count_sync = torch.zeros(
            self.worker_config.world_size,
            dtype=torch.int64,
            device=self.distributed_device,
            requires_grad=False,
        )
        dist.all_gather(
            [overuse_count_sync[i] for i in range(self.worker_config.world_size)],
            self.overuse_counts[rank].to(device=self.distributed_device),
            group=self.worker_config.data_parallel_group,
        )
        self.overuse_counts[:] = overuse_count_sync.cpu()

        # Iterate until any rank is exhausted
        self_exhausted = 0
        while not self.exhausted_states.any():
            if len(samples) > 0:
                # First use pending samples from previous iteration
                sample = samples.pop(0)
            else:
                try:
                    sample = next(self._iterator)
                except StopIteration:
                    # print(f"[r={rank}]: StopIteration\n", end="")
                    self.exhausted_states[rank] = self_exhausted = 1
            dist.all_reduce(
                self.exhausted_states[rank],
                op=dist.ReduceOp.MAX,
                group=self.worker_config.data_parallel_group,
            )
            global_any_exhausted = bool(self.exhausted_states[rank].item())

            if global_any_exhausted:
                # print(f"[r={rank}]: One rank exhausted\n", end="")
                self.exhausted_states[rank] = self_exhausted
                if not self_exhausted:
                    # print(f"[r={rank}]: Not exhausted, storing sample\n", end="")
                    samples.append(sample)
                break

            yield sample

        sync_ranks = True
        sample_count = torch.zeros(
            self.worker_config.world_size,
            dtype=torch.int64,
            device=self.distributed_device,
            requires_grad=False,
        )
        sample_count_list = [sample_count[i] for i in range(self.worker_config.world_size)]

        # Redistribute the samples until all ranks are exhausted
        #  * The ranks with the least overuse count shall fetch more sample(s) as needed
        #  * The ranks which are already exhausted shall receive one sample of the additionally fetched samples
        while not self.exhausted_states.all() or sync_ranks:
            if sync_ranks:
                # Share all exhausted states
                dist.all_gather(
                    self._exhausted_states_list,
                    self.exhausted_states[rank],
                    group=self.worker_config.data_parallel_group,
                )
                exhausted_cpu = self.exhausted_states.cpu()
                if exhausted_cpu.all():
                    break
                for i in range(self.worker_config.world_size):
                    if exhausted_cpu[i]:
                        self.overuse_counts[i] = self.OVERUSE_COUNT_MAX

                # Check if there are enough samples to satisfy the need
                dist.all_gather(
                    sample_count_list,
                    torch.tensor(len(samples), dtype=torch.int64, device=self.distributed_device),
                    group=self.worker_config.data_parallel_group,
                )
                needed_samples = self.worker_config.world_size - sample_count.sum().item()

                # print(f"[r={rank}]: Exhausted: {self.exhausted_states.cpu()}, Sample count: {sample_count.cpu()}, overuse counts: {self.overuse_counts}\n", end="")

                # The ranks are now in sync with all dataloader states and sample counts
                sync_ranks = False
            if needed_samples > 0:
                # print(f"[r={rank}]: Need {needed_samples} samples\n", end="")
                # Not enough samples to satisfy the need -> fetch more on non-exhausted ranks
                oversample_self = self._find_ranks_to_oversample(needed_samples)
                # print(f"[r={rank}]: Oversample self {oversample_self} samples\n", end="")
                while oversample_self > 0:
                    try:
                        samples.append(next(self._iterator))
                        # print(f"[r={rank}]: Got {len(samples)} samples\n", end="")
                    except StopIteration:
                        # print(f"[r={rank}]: Exhausted\n", end="")
                        self.exhausted_states[rank] = 1
                        break
                    else:
                        oversample_self -= 1
                # print(f"[r={rank}]: Got {len(samples)} samples\n", end="")
                # Loop again, in case another rank exhausted now and did not get a sample, sync ranks again to be sure
                sync_ranks = True
                continue
            else:
                # All ranks in sum have enough samples -> distribute the samples.
                assert needed_samples == 0, (
                    f"Needed {needed_samples} samples, but have {sample_count.sum().item()}"
                )

                # For each sample, compute the samples that can be distributed to other ranks
                sending_ranks = [
                    (rank, idx)
                    for rank, count in enumerate(sample_count)
                    for idx in range(1, count)
                ]
                # Compute the rank that is going to send a sample to this rank
                # xor the ranks that are going to receive a sample from this rank
                self_source_rank: int | None = None
                # List of (target_rank, source_sample_idx) that are going to receive a sample from this rank
                self_target_ranks: list[tuple[int, int]] = []
                for chk_rank in range(self.worker_config.world_size):
                    if sample_count[chk_rank] == 0:
                        # This rank is not going to receive a sample, because it has no samples
                        # Take the first sample from the sending ranks and send it to this rank
                        src_rank, src_idx = sending_ranks.pop()
                        # print(f"[r={rank}]: Sending sample {src_idx} from rank {src_rank} to rank {chk_rank}\n", end="")
                        if chk_rank == rank:
                            # If the self rank is the receiving rank, store the rank we're receiving from
                            self_source_rank = src_rank
                        elif src_rank == rank:
                            # If the self rank is the sending rank, store the rank we're sending to and which sample
                            self_target_ranks.append((chk_rank, src_idx))

                if self_source_rank is not None:
                    # print(f"[r={rank}]: Receiving sample from rank {self_source_rank}\n", end="")
                    # This rank is going to receive a sample from that other rank
                    object_list = [None]
                    # Receive the sample from the source rank. Requires the global rank, disregarding the group
                    dist.recv_object_list(object_list, src=self._as_global_rank(self_source_rank))
                    samples.append(object_list[0])
                elif len(self_target_ranks) > 0:
                    # This rank is going to send a sample to that other rank(s)
                    for dst_rank, sample_idx in self_target_ranks:
                        # print(f"[r={rank}]: Sending sample {sample_idx} to rank {dst_rank}\n", end="")
                        object_list = [samples[sample_idx]]
                        samples[sample_idx] = None
                        # Send the sample to the destination rank. Requires the global rank, disregarding the group
                        dist.send_object_list(object_list, dst=self._as_global_rank(dst_rank))

                # Remove the samples that have been distributed
                for i in range(len(samples) - 1, -1, -1):
                    if samples[i] is None:
                        del samples[i]
                if len(samples) > 1:
                    # It may happen, that there are more samples than needed (case: a rank had oversampled, but not enough
                    # to provide for all ranks; restarting the loop, and one rank has no initial samples, but others do have
                    # samples now, then the oversampled samples cannot be distributed to all ranks)
                    # Need to save the samples in case the loop is interrupted at the yield
                    self._next_samples = samples
                else:
                    # Ensure no next samples are set, all were consumed
                    self._next_samples = None
                # assert len(samples) == 1, f"Every rank should have one sample now, have {len(samples)} on rank {self.worker_config.rank}"
                # Important: When yielding, the samples list is empty. It's not part of the state, so it does not need
                # to be saved.
                yield samples.pop(0)
                # print(f"[r={rank}]: Yielded sample {len(samples)}, getting next\n", end="")

                needed_samples = self.worker_config.world_size

        # print(f"[r={rank}]: Done iterating\n", end="")

        self._next_samples = samples
        self._next_sample_restore_keys = None

        # Done iterating, reset the iterator
        self._iterator = None
        self.exhausted_states.fill_(0)
        self.overuse_counts.fill_(0)

    def __len__(self):
        return len(self.loader)

    def save_state_rank(self) -> RedistributeDataLoaderState:
        assert isinstance(self.loader, SavableDataLoader)
        if self._next_sample_restore_keys is not None:
            restore_keys = self._next_sample_restore_keys
        elif self._next_samples is not None:
            restore_keys = self._next_sample_restore_keys = [
                get_sample_restore_key(sample) for sample in self._next_samples
            ]
        else:
            restore_keys = None
        return RedistributeDataLoaderState(
            inner_state=self.loader.save_state_rank(),
            overuse_count=int(self.overuse_counts[self.worker_config.rank].item()),
            exhausted_state=bool(self.exhausted_states[self.worker_config.rank].item()),
            next_samples_restore_keys=restore_keys,
        )

    def restore_state_rank(self, state: RedistributeDataLoaderState) -> None:
        assert isinstance(self.loader, SavableDataLoader)
        self.loader.restore_state_rank(state.inner_state)
        self._next_sample_restore_keys = state.next_samples_restore_keys
        self._next_samples = None
        self.overuse_counts[self.worker_config.rank] = state.overuse_count
        self.exhausted_states[self.worker_config.rank] = state.exhausted_state

    def can_restore_sample(self) -> bool:
        return self.loader.can_restore_sample()

    def restore_sample(self, restore_key: Tuple[Union[str, int, tuple], ...]) -> T:
        return self.loader.restore_sample(restore_key)

    def save_state_global(
        self, global_dst_rank: int
    ) -> Optional[Sequence[RedistributeDataLoaderState]]:
        """
        See :meth:`megatron.energon.SavableDataLoader.save_state_global`
        """
        # Fetch current rank's worker's state
        merged_state = self.save_state_rank()

        # Gather the merged states
        if self.worker_config.world_size > 1:
            output: Optional[Sequence[RedistributeDataLoaderState]]
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
        state: Optional[Sequence[RedistributeDataLoaderState]],
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
