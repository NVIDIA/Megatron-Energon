# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any, Union

from megatron.energon.epathlib import EPath


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceInfo:
    """Information about the source of a sample, i.e. where the data was loaded from."""

    #: The path to the dataset
    dataset_path: EPath
    #: The index of the sample in the dataset
    index: Union[str, int]
    #: The name of the shard tar file
    shard_name: str
    #: The names of the files in the shard used to create the sample
    file_names: tuple[str, ...]


def add_source_info(sample: Any, *source_infos: SourceInfo) -> Any:
    """Adds a source info to a sample."""
    from megatron.energon.flavors.base_dataset import Sample

    if isinstance(sample, Sample) or hasattr(sample, "__sources__"):
        if sample.__sources__ is None:
            sample.__sources__ = source_infos
        else:
            sample.__sources__ = (*sample.__sources__, *source_infos)
    elif isinstance(sample, dict):
        if sample.get("__sources__") is None:
            sample["__sources__"] = source_infos
        else:
            sample["__sources__"] = (*sample["__sources__"], *source_infos)
    return sample
