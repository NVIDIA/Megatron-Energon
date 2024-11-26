# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Generator, List, Optional, Sequence

from megatron.energon.flavors.webdataset.structs import ShardInfo
from megatron.energon.worker import WorkerConfig


class Sharder:

    @staticmethod
    def _split_shard(
        subshards: Sequence[ShardInfo],
        start_offset: int,
        end_offset: int,
        max_samples_per_sequence: Optional[int],
    ) -> List[Sequence[ShardInfo]]:
        if (
            max_samples_per_sequence is not None
            and end_offset - start_offset > max_samples_per_sequence * 1.5
        ):
            # Split the shard into slices of max_samples_per_sequence (more or less)
            slice_count = max(round((end_offset - start_offset) / max_samples_per_sequence), 1)
            samples_per_sequence = (end_offset - start_offset) / slice_count
            # Note this must include the end offset as well, so slice_count + 1 steps (down there,
            # idx+1 is used to access the end offset)
            offsets = [
                start_offset + int(slice * samples_per_sequence) for slice in range(slice_count + 1)
            ]
            return [
                [
                    dataclasses.replace(
                        shard,
                        offset=offsets[idx],
                        count=offsets[idx + 1] - offsets[idx],
                        byte_offset=None,
                        byte_size=None,
                    )
                    for shard in subshards
                ]
                for idx in range(slice_count)
            ]
        else:
            return [
                [
                    dataclasses.replace(
                        shard,
                        offset=start_offset,
                        count=end_offset - start_offset,
                        byte_offset=None,
                        byte_size=None,
                    )
                    for shard in subshards
                ]
            ]

    @classmethod
    def _split_shards(
        cls,
        shards: List[Sequence[ShardInfo]],
        offsets: List[int],
        *,
        max_samples_per_sequence: Optional[int],
    ) -> Generator[List[Sequence[ShardInfo]], None, None]:
        """
        Splits the shards into multiple lists based on the offsets. The first offset is the start
        of the first shard emitted, the last offset is the end of the last shard emitted.
        (i.e. number of shards emitted is `len(offsets) - 1`)

        Args:
            shards: The source shards
            offsets: The offsets to samples to get shards for (must be strictly increasing)
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).

        Returns:
            A list of shards for each offset pair
        """
        # The start index of the current shard
        cum_count = 0

        # Find shard idx for start
        for start_index, start_subshards in enumerate(shards):
            if cum_count + start_subshards[0].count < offsets[0]:
                # The shard is before the offset -> go to next shard
                cum_count += start_subshards[0].count
                continue
            else:
                # The shard contains the offset
                start_offset = offsets[0] - cum_count
                break
        else:
            raise ValueError("Invalid shard distribution")

        for offset in offsets[1:]:
            # Find shard idx for end
            for end_index, end_subshards in enumerate(shards[start_index:], start=start_index):
                if cum_count + end_subshards[0].count < offset:
                    # The shard is before the offset -> go to next shard
                    cum_count += end_subshards[0].count
                    continue
                else:
                    # The shard contains the offset
                    end_offset = offset - cum_count
                    break
            else:
                raise ValueError("Invalid shard distribution")
            if start_index == end_index:
                yield cls._split_shard(
                    start_subshards,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    max_samples_per_sequence=max_samples_per_sequence,
                )
            else:
                # Middle is the original shards, start and end get an offset/length
                yield (
                    (
                        cls._split_shard(
                            start_subshards,
                            start_offset=start_offset,
                            end_offset=start_subshards[0].count,
                            max_samples_per_sequence=max_samples_per_sequence,
                        )
                        if start_subshards[0].count > start_offset
                        else []
                    )
                    + sum(
                        (
                            cls._split_shard(
                                subshards,
                                start_offset=subshards[0].offset,
                                end_offset=subshards[0].count,
                                max_samples_per_sequence=max_samples_per_sequence,
                            )
                            for subshards in shards[start_index + 1 : end_index]
                        ),
                        start=[],
                    )
                    + cls._split_shard(
                        end_subshards,
                        start_offset=end_subshards[0].offset,
                        end_offset=end_offset,
                        max_samples_per_sequence=max_samples_per_sequence,
                    )
                )
            start_index = end_index
            start_subshards = end_subshards
            start_offset = end_offset

    @classmethod
    def shard_workers(
        cls,
        shards: List[Sequence[ShardInfo]],
        worker_config: WorkerConfig,
        *,
        max_samples_per_sequence: Optional[int],
        rotation_offset: int = 0,
    ) -> List[List[Sequence[ShardInfo]]]:
        """
        Creates subshards (ShardInfo) for each worker of the current rank.
        For that, the total number of samples is split into the number of global workers across all
        ranks. Then each worker gets a slice of the global samples.

        Args:
            shards: The shards to split
            worker_config: The config for the current rank and workers

        Returns:
            The shards for the current rank and all workers
        """

        # We split the total number of samples into the number of global workers across all ranks.
        # Note that the global number of workers intentionally stay the same if you
        # divide the number of ranks by N, and multiply the number of workers per rank by N.
        # This allows to reproduce the same global batches with a different number of ranks.

        num_workers = max(1, worker_config.num_workers)

        total_samples = sum(subshards[0].count for subshards in shards)
        global_workers = num_workers * worker_config.world_size

        samples_per_worker = total_samples / global_workers

        # Let's look at the workers of the current local rank.
        # Each worker gets a slice of the global samples as follows:
        local_rank_worker_sample_offsets = [
            int((num_workers * worker_config.rank + local_worker_idx) * samples_per_worker)
            for local_worker_idx in range(num_workers + 1)
        ]

        # Note that the first workers will get a bit more samples if the total number of samples
        # is not divisible by the number of workers (which is usually the case).
        # We will now rotate the offsets, to continue filling the workers, where the previous dataset left off.
        local_rank_worker_sample_offsets = (
            local_rank_worker_sample_offsets[rotation_offset:]
            + local_rank_worker_sample_offsets[:rotation_offset]
        )

        first_worker_with_less_samples = (total_samples + rotation_offset) % global_workers

        return list(
            # Filter out any empty shards for this worker
            [subshards for subshards in shards if subshards[0].count > 0]
            for shards in cls._split_shards(
                shards,
                local_rank_worker_sample_offsets,
                max_samples_per_sequence=max_samples_per_sequence,
            )
        )
