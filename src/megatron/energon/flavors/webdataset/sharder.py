# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from itertools import zip_longest
from typing import Generator, List, Optional, Sequence, Union

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
    def _generalized_bit_reversal(cls, length_or_indices: Union[int, List[int]]) -> List[int]:
        """This function creates a permutation of given length.
        The sequence is created by a recursive divide and interleave algorithm
        to ensure a balanced distribution across ranks.
        It corresponds to a generalized bit reversal permutation, which - for lengths
        of power of two - is the reversed binary representation of the original indices.

        For example for 16 indices, the sequence is:
            [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15]

        Visual illustration:
            Step|0|1|2|3|4|5|6|7|8|9|A|B|C|D|E|F|
                |-------------------------------|
               0|X| | | | | | | | | | | | | | | |
               1|X| | | | | | | |X| | | | | | | |
               2|X| | | |X| | | |X| | | | | | | |
               3|X| | | |X| | | |X| | | |X| | | |
               4|X| |X| |X| | | |X| | | |X| | | |
               5|X| |X| |X| | | |X| |X| |X| | | |
               6|X| |X| |X| |X| |X| |X| |X| | | |
               7|X| |X| |X| |X| |X| |X| |X| |X| |
               8|X|X|X| |X| |X| |X| |X| |X| |X| |
               9|X|X|X| |X| |X| |X|X|X| |X| |X| |
              10|X|X|X| |X|X|X| |X|X|X| |X| |X| |
              11|X|X|X| |X|X|X| |X|X|X| |X|X|X| |
              12|X|X|X|X|X|X|X| |X|X|X| |X|X|X| |
              13|X|X|X|X|X|X|X| |X|X|X|X|X|X|X| |
              14|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X| |
              15|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X|X|
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

        left_result = cls._generalized_bit_reversal(left)
        right_result = cls._generalized_bit_reversal(right)

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
        For that, the number of global samples is split across the number of global workers across all
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
        # This is done in multiple steps.
        # Some of these steps could be collapsed into one, but we keep them separate for clarity:
        # 1. Compute the number of samples per global worker (rotated by rotation_offset,
        #    typically given by previous datasets).
        # 2. Permute the nuber of samples per global worker by a generalized bit reversal sequence
        # 3. Given the sample counts, compute the start and end indices for each global worker
        # 4. Extract the local worker sample assignments for the current rank.
        # 5. Split the shards based on the start and end indices.

        # 1. Let's compute it globally for all workers first
        num_samples_per_global_worker = []
        for global_worker_idx in range(global_workers):
            if (
                global_worker_idx - rotation_offset + global_workers
            ) % global_workers < num_workers_with_more_samples:
                # This worker gets one more sample
                num_samples_per_global_worker.append(min_samples_per_worker + 1)
            else:
                # This worker gets the minimum number of samples
                num_samples_per_global_worker.append(min_samples_per_worker)

        # 2. Permute the number of samples per global worker
        worker_bitrev_seq = cls._generalized_bit_reversal(global_workers)

        # The worker_bitrev_seq is the order in which any remainder samples shall
        # be assigned to workers.
        # That means, the x-axis (array index) is the remainder sample index
        # and the y-axis (value) is the global worker index.
        # So we map the y (value) to the old global worker index from the linear sequence.

        new_num_samples_per_global_worker = [-1] * global_workers
        for old_worker_idx, new_worker_idx in enumerate(worker_bitrev_seq):
            new_num_samples_per_global_worker[new_worker_idx] = num_samples_per_global_worker[
                old_worker_idx
            ]

        num_samples_per_global_worker = new_num_samples_per_global_worker

        # 3. Compute the global worker sample start and end indices
        global_worker_sample_split_offsets = [0]
        cur_offset = 0
        for global_worker_idx in range(global_workers):
            cur_offset += num_samples_per_global_worker[global_worker_idx]
            global_worker_sample_split_offsets.append(cur_offset)

        # 4. Now we extract the local rank's worker ranges
        local_worker_sample_split_offsets = global_worker_sample_split_offsets[
            worker_config.rank * num_workers : (worker_config.rank + 1) * num_workers + 1
        ]

        assert (
            len(local_worker_sample_split_offsets) == num_workers + 1
        ), "If this fails, there's a bug in the code above."

        # 5. Now we can split the shards
        return list(
            # Filter out any empty shards for this worker
            [subshards for subshards in shards if subshards[0].count > 0]
            for shards in cls._split_shards(
                shards,
                local_worker_sample_split_offsets,
                max_samples_per_sequence=max_samples_per_sequence,
            )
        )
