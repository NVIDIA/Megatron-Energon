# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import re
from dataclasses import replace
from typing import Callable, List

import torch

from megatron.energon.dataloader.dataloader import RankState
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import RestoreKey
from megatron.energon.savable_loader import SavableDataLoaderState
from megatron.energon.state import FlexState
from megatron.energon.wrappers.base import WrappedRestoreKey
from megatron.energon.wrappers.batch_dataset import BatchGenRestoreKey, BatchRestoreKey
from megatron.energon.wrappers.iter_map_dataset import IterMapRestoreKey
from megatron.energon.wrappers.map_dataset import MapGenRestoreKey


def _split_batch_restore_key(
    key: RestoreKey | None, batch_size: int, *, drop_last: bool = False
) -> list[RestoreKey]:
    if batch_size < 1:
        raise ValueError("The target batch size must be positive")
    if key is None:
        raise ValueError("Cannot split a batch without a restore key")
    if isinstance(key, (BatchGenRestoreKey, MapGenRestoreKey, IterMapRestoreKey)):
        raise ValueError("Cannot resize a checkpoint with prefetched generator batches")
    if isinstance(key, BatchRestoreKey):
        samples = key.inner
        if not samples:
            raise ValueError("Cannot split an empty batch restore key")
        stop = len(samples) - len(samples) % batch_size if drop_last else len(samples)
        return [
            replace(key, inner=samples[start : start + batch_size])
            for start in range(0, stop, batch_size)
        ]
    if isinstance(key, WrappedRestoreKey):
        return [
            replace(key, inner=inner)
            for inner in _split_batch_restore_key(key.inner, batch_size, drop_last=drop_last)
        ]
    raise ValueError(f"Unsupported restore key type for splitting batch: {type(key).__name__}")


def _has_pending_generator_batch(state: FlexState) -> bool:
    if state.get("__class__") == "BatchDataset" and state.get("_generator_sample_keys") is not None:
        return True
    key = state.get("_generator_sample_key")
    while isinstance(key, WrappedRestoreKey):
        key = key.inner
    if isinstance(key, BatchRestoreKey):
        return True
    return any(_has_pending_generator_batch(child) for child in state.get("datasets", ()))


def resize_rank_state(
    state: RankState, batch_size: int | None, *, drop_last: bool = False
) -> RankState:
    """Resize pending batch keys while preserving the worker snapshots."""
    old_size = state.micro_batch_size
    if old_size == batch_size:
        return state
    if (
        old_size is None
        or batch_size is None
        or not 0 < batch_size < old_size
        or old_size % batch_size
    ):
        raise ValueError("The target micro-batch size must be a smaller divisor of the saved size")
    if any(
        worker is not None and _has_pending_generator_batch(worker.dataset)
        for worker in state.worker_states
    ):
        raise ValueError("Cannot resize a checkpoint with an in-progress generator batch")
    return replace(
        state,
        micro_batch_size=batch_size,
        prefetched_restore_keys=[
            [
                piece
                for key in worker_keys
                for piece in _split_batch_restore_key(key, batch_size, drop_last=drop_last)
            ]
            for worker_keys in state.prefetched_restore_keys
        ],
    )


def natural_sort_key(s):
    """
    Function to use for natural sorting of filenames.

    This splits the input string by numbers and non-numbers and ensures
    that numbers are compared as integers, not as strings.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def detect_and_replicate_pattern(file_list):
    """
    Given a list of file paths, detect the single numeric pattern and return
    a function that, when called with integer n (starting from 0), generates
    the n-th filename following that pattern.

    Raises an Exception if no pattern or multiple patterns are found.
    """
    if not file_list:
        raise ValueError("Cannot detect a pattern from an empty list.")

    sorted_files = sorted(file_list, key=natural_sort_key)

    def tokenize_filename(fname):
        return re.split(r"(\d+)", fname)

    tokenized = [tokenize_filename(f) for f in sorted_files]

    token_len = len(tokenized[0])
    for t in tokenized:
        if len(t) != token_len:
            raise Exception("Filenames do not share a consistent token structure.")

    num_positions = []
    for pos in range(token_len):
        example_token = tokenized[0][pos]
        example_is_digit = example_token.isdigit()

        all_tokens_at_pos = [t[pos] for t in tokenized]

        if example_is_digit:
            values = [int(x) if x.isdigit() else None for x in all_tokens_at_pos]
            if len(set(values)) > 1:
                num_positions.append(pos)
        elif len(set(all_tokens_at_pos)) != 1:
            raise Exception("Non-digit token differs among files. Invalid pattern.")

    if len(num_positions) == 0:
        raise Exception("No numeric portion found that differs among files.")
    if len(num_positions) > 1:
        raise Exception("Multiple numeric portions found that differ. Not a single pattern.")

    varying_pos = num_positions[0]
    numeric_values = [int(t[varying_pos]) for t in tokenized]

    for i in range(len(numeric_values) - 1):
        if numeric_values[i + 1] - numeric_values[i] != 1:
            raise Exception("Numeric values are not consecutive. Pattern is invalid.")

    base_value = numeric_values[0]
    zero_padding_width = len(tokenized[0][varying_pos])

    def generate_filename(n):
        new_tokens = tokenized[0][:]
        new_int_value = base_value + n
        new_str_value = str(new_int_value).zfill(zero_padding_width)
        new_tokens[varying_pos] = new_str_value
        return "".join(new_tokens)

    for i in range(len(sorted_files)):
        candidate = generate_filename(i)
        if candidate != sorted_files[i]:
            raise Exception(
                "Verification failed. The generated pattern does not match the input list."
            )

    return generate_filename


class RankStateIterable:
    """Iterates the SavableDatasetCheckpoints of multiple ranks in a round-robin fashion."""

    file_pattern_func: Callable[[int], str]
    is_global_checkpoint: bool
    megatron_style: bool
    num_states: int
    rank_cur_worker: list[int]
    rank_num_workers: list[int]
    #: Rank checkpoints in rank order, in either the custom or legacy loader format.
    rank_states: list[SavableDataLoaderState | RankState]
    rank_worker_offset: list[int]

    def __init__(self, state_files: List[EPath]):
        state_file_names = [state_file.name for state_file in state_files]

        self.file_pattern_func = (
            (lambda rank: str(rank) + "-" + state_file_names[0])
            if len(state_files) == 1
            else detect_and_replicate_pattern(state_file_names)
        )
        self.num_states = len(state_files)

        first_state = torch.load(str(state_files[0]), weights_only=False)

        if isinstance(first_state, dict) and "dataloader_state_dict" in first_state:
            self.megatron_style = True
            first_state = first_state["dataloader_state_dict"]
        else:
            self.megatron_style = False

        if isinstance(first_state, (SavableDataLoaderState, RankState)):
            if self.megatron_style:
                self.rank_states = [first_state] + [
                    torch.load(str(state_file), weights_only=False)["dataloader_state_dict"]
                    for state_file in state_files[1:]
                ]
            else:
                self.rank_states = [first_state] + [
                    torch.load(str(state_file), weights_only=False)
                    for state_file in state_files[1:]
                ]
            self.is_global_checkpoint = False
        elif isinstance(first_state, list):
            assert len(state_files) == 1, "Global checkpoint must contain exactly one file"
            assert first_state and all(type(state) is type(first_state[0]) for state in first_state)
            self.rank_states = first_state
            self.is_global_checkpoint = True
        else:
            raise ValueError(f"Unknown checkpoint type: {type(first_state)}")

        self.rank_cur_worker = [0] * len(self.rank_states)
        self.rank_worker_offset = [state.next_worker_id for state in self.rank_states]

        self.rank_num_workers = [len(state.worker_states) for state in self.rank_states]
        assert all(
            self.rank_num_workers[0] == num_workers for num_workers in self.rank_num_workers
        ), "All ranks must have the same number of workers."

    def write_new_states_to_folder(
        self, output_folder: EPath, new_states: List[SavableDataLoaderState | RankState]
    ):
        for rank_idx, rank_state in enumerate(new_states):
            output_file = output_folder / self.file_pattern_func(rank_idx)
            if self.megatron_style:
                torch.save(
                    {"dataloader_state_dict": rank_state},
                    str(output_file),
                )
            else:
                torch.save(rank_state, str(output_file))

    def get_num_ranks(self):
        return len(self.rank_states)

    def get_num_workers(self):
        return self.rank_num_workers[0]

    def get_micro_batch_size(self):
        return self.rank_states[0].micro_batch_size

    def __iter__(self):
        """Iterates the SavableDatasetCheckpoints of multiple ranks in a round-robin fashion."""
        for state in self.rank_states:
            if isinstance(state, RankState):
                exhausted = state.exhausted_workers or [False] * len(state.worker_states)
                yield from zip(state.worker_states, state.prefetched_restore_keys, exhausted)
            else:
                yield from state.worker_states
