# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from itertools import zip_longest
from typing import List, Optional, Sequence, Union

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
        start: int,
        end: int,
        *,
        max_samples_per_sequence: Optional[int],
    ) -> List[Sequence[ShardInfo]]:
        """
        Splits the shards into multiple lists based on the start/end sample offsets.
        The splitting is done across shard boundaries and multiple shards can be returned.

        Args:
            shards: The source shards
            start: The index of the first sample to include
            end: The index of the end sample (not included)
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                  will be sequential).

        Returns:
            A list of shards for each offset pair
        """
        # The start index of the current shard
        cum_count = 0

        # Find shard idx for start
        for start_index, start_subshards in enumerate(shards):
            if cum_count + start_subshards[0].count < start:
                # The shard is before the offset -> go to next shard
                cum_count += start_subshards[0].count
                continue
            else:
                # The shard contains the offset
                start_offset = start - cum_count
                break
        else:
            raise ValueError("Invalid shard distribution")

        # Find shard idx for end
        for end_index, end_subshards in enumerate(shards[start_index:], start=start_index):
            if cum_count + end_subshards[0].count < end:
                # The shard is before the offset -> go to next shard
                cum_count += end_subshards[0].count
                continue
            else:
                # The shard contains the offset
                end_offset = end - cum_count
                break
        else:
            raise ValueError("Invalid shard distribution")
        if start_index == end_index:
            return cls._split_shard(
                start_subshards,
                start_offset=start_offset,
                end_offset=end_offset,
                max_samples_per_sequence=max_samples_per_sequence,
            )
        else:
            # Middle is the original shards, start and end get an offset/length
            return (
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

    @classmethod
    def _magic_sequence(cls, length_or_indices: Union[int, List[int]]) -> List[int]:
        """This function creates sequence of indices for interleaving.
        The sequence is created by a divide and interleave algorithm to ensure
        a balanced distribution across ranks. For lengths of power of two,
        the sequence corresponds to the reversed binary representation of the
        indices.

        For exapmle for 16 indices, the sequence is:
        [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]
        """

        if isinstance(length_or_indices, int):
            indices = list(range(length_or_indices))
        else:
            indices = length_or_indices

        if len(indices) <= 2:
            return indices
        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        left_result = cls._magic_sequence(left)
        right_result = cls._magic_sequence(right)

        # Interleave the results
        zipped = zip_longest(left_result, right_result)
        result = [item for sublist in zipped for item in sublist if item is not None]
        return result

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
        # Note that the global number of workers intentionally stays the same if you
        # divide the number of ranks by N, and multiply the number of workers per rank by N.
        # This allows to reproduce the same global batches with a different number of ranks.

        num_workers = max(1, worker_config.num_workers)

        total_samples = sum(subshards[0].count for subshards in shards)
        global_workers = num_workers * worker_config.world_size

        min_samples_per_worker = int(total_samples / global_workers)
        num_workers_with_more_samples = total_samples % global_workers

        # We are going to compute the samples assigned to each worker on the current rank.
        # First we compute the global worker sample assignments (rotated by the rotation offset).
        # Then we extract the local worker sample assignments for the current rank.

        # Let's compute it globally for all workers first
        global_worker_sample_start_ends = []
        cur_offset = 0
        for global_worker_idx in range(global_workers):
            if (
                global_worker_idx - rotation_offset + global_workers
            ) % global_workers < num_workers_with_more_samples:
                # This worker gets one more sample
                cur_offset += min_samples_per_worker + 1
            else:
                # This worker gets the minimum number of samples
                cur_offset += min_samples_per_worker

            if len(global_worker_sample_start_ends) == 0:
                global_worker_sample_start_ends.append((0, cur_offset))
            else:
                global_worker_sample_start_ends.append(
                    (global_worker_sample_start_ends[-1][1], cur_offset)
                )

        # Now we extract the local rank's worker ranges but in a strided way
        # for better balance across ranks.
        # We cannot have this striding depend on the number of ranks, because
        # we want reproducible results when changing the number of ranks.
        # Hence we use a magic sequence to interleave the global worker indices.

        worker_magic_seq = cls._magic_sequence(global_workers)
        # The worker_magic_seq is the order in which workers shall be assigned samples.
        # We need to reverse this mapping to get the local worker indices.

        rev_map = [-1] * len(worker_magic_seq)
        for i, x in enumerate(worker_magic_seq):
            rev_map[x] = i

        local_worker_sample_start_ends = [
            global_worker_sample_start_ends[rev_map[idx]]
            for idx in range(
                worker_config.rank * num_workers, (worker_config.rank + 1) * num_workers
            )
        ]

        assert (
            len(local_worker_sample_start_ends) == num_workers
        ), "If this fails, there's a bug in the code above."

        # Now we can split the shards
        final_worker_shards = []
        for start, end in local_worker_sample_start_ends:
            cur_shards = cls._split_shards(
                shards,
                start,
                end,
                max_samples_per_sequence=max_samples_per_sequence,
            )

            # Filter out any empty shards for this worker
            final_worker_shards.append(
                [subshards for subshards in cur_shards if subshards[0].count > 0]
            )

        return final_worker_shards
