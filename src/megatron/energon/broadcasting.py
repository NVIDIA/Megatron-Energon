# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from pathlib import Path
from typing import Optional, TypeVar, Union, Any, Literal, List

from torch.utils.data import DataLoader

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.savable_loader import BasicDataLoader, SavableDataLoader
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.gc_dataset import GcDataset

from megatron.energon.epathlib import EPath
from megatron.energon.flavors import SavableDataset
from megatron.energon.metadataset import load_dataset
from megatron.energon.task_encoder.base import DefaultTaskEncoder, TaskEncoder, WorkerConfig
from megatron.energon.task_encoder import get_train_dataset

T = TypeVar("T")


def get_broadcasting_train_loader(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["train"], str] = "train",
    batch_size: int,
    data_loader_groups: Union[List[List[int]], List[torch.distributed.ProcessGroup]],
    world_rank: Optional[int] = None,
    world_size: Optional[int] = None,
    num_workers: int = 4,
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    shuffle_buffer_size: Optional[int],
    max_samples_per_sequence: Optional[int],
    virtual_epoch_length: int = 0,
    shuffle_over_epochs_multiplier: int = 1,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    savable: bool = True,
    checkpoint_every_sec: float = 60,
    checkpoint_every_min_n_samples: Optional[int] = None,
    n_checkpoints: int = 2,
    **kwargs,
) -> SavableDataLoader[T]:
    """

    Get a loader that may read data from disk or receive data from other ranks via broadcasting.
    TODO: More explanation

    Args:
        TODO

    Returns:
        The instantiated :class:`megatron.energon.SavableDataLoader`, yielding batches from the dataset,
        allowing to save the state of the dataset.
    """

    if world_rank is None or world_size is None:
        assert world_rank is None and world_size is None, "Must set either both or none of world_rank and world_size."
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    
    assert len(data_loader_groups) > 0, "Must define at least one data_loader_group."

    # First convert all groups to ProcessGroups if they are lists of int.
    # All ranks need to do this together even if they are not part of the group
    # that's being created
    if isinstance(data_loader_groups[0], list):
        new_data_loader_groups = []
        for data_loader_group in data_loader_groups:
            assert isinstance(data_loader_group, list), "Cannot mix lists of ranks with ProcessGroup in data_loader_groups."
            new_data_loader_groups.append(
                torch.distributed.new_group(ranks=data_loader_group)
            )
        data_loader_groups = new_data_loader_groups

    my_group = None
    my_group_index = None
    ranks_broadcasting = []
    for group_index, data_loader_group in enumerate(data_loader_groups):
        group_ranks = torch.distributed.get_process_group_ranks(data_loader_group)
        
        ranks_broadcasting.append(group_ranks[0])

        if world_rank in group_ranks:
            my_group = data_loader_group
            mu_group_index = group_index
    
    assert my_group is not None, f"The rank {world_rank} is not part of any group in data_loader_groups."

    # Only the first rank in each data_loader_group will actually load the data
    # It will then broadcast the data to all other ranks in the group

    rank_is_broadcasting = world_rank in ranks_broadcasting

    if rank_is_broadcasting:

        worker_config = WorkerConfig(
            rank=my_group_index,
            world_size=len(data_loader_groups),
            num_workers=num_workers,
        )
        dataset = get_train_dataset(
            path=path,
            split_part=split_part,
            worker_config=worker_config,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            shuffle_buffer_size=shuffle_buffer_size,
            max_samples_per_sequence=max_samples_per_sequence,
            virtual_epoch_length=virtual_epoch_length,
            shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
            task_encoder=task_encoder,
            **kwargs
        )

        dataset.verify_worker_config(worker_config)
    
        loader = SavableDataLoader(
            dataset,
            worker_config=worker_config,
            checkpoint_every_sec=checkpoint_every_sec,
            checkpoint_every_min_n_samples=checkpoint_every_min_n_samples,
            n_checkpoints=n_checkpoints,
        )
        # TODO: Broadcast data
    else:
        # TODO: Receive broadcasted data
        pass

def get_loader(
    dataset: SavableDataset[T],
    *,
    worker_config: WorkerConfig,
) -> BasicDataLoader[T]:
    """
    Get a dataloader for the given dataset.

    Args:
        dataset: The dataset to create a loader for.
        worker_config: The worker config. Set the same value to the `get(_val|_train)_dataset` call.

    Returns:
        The instantiated :class:`torch.data.DataLoader`, yielding batches from the dataset.
    """
    dataset.verify_worker_config(worker_config)
    return BasicDataLoader(
        GcDataset(dataset),
        worker_config=worker_config,
    )
