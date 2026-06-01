# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Union

from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.webdataset.metadata import EnergonDatasetType, get_dataset_type
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import (
    DatasetLoaderInterface,
    TraversedDatasetReference,
)
from megatron.energon.metadataset.metadataset import Metadataset
from megatron.energon.typed_converter import JsonParser


def load_dataset(
    path: Union[str, EPath, Path],
    **kwargs,
) -> DatasetLoaderInterface:
    """Loads a (meta)dataset."""

    if isinstance(path, dict):
        mds = load_config(
            path,
            default_type=Metadataset,
            default_kwargs=dict(path=EPath("/dict"), **kwargs),
        )
        return mds
    path = EPath(path)
    ds_type = get_dataset_type(path)
    if ds_type == EnergonDatasetType.METADATASET:
        mds = load_config(
            path,
            default_type=Metadataset,
            default_kwargs=dict(path=path, **kwargs),
        )
        mds.post_initialize()
        return mds
    elif ds_type in (
        EnergonDatasetType.WEBDATASET,
        EnergonDatasetType.JSONL,
        EnergonDatasetType.BINIDX,
        EnergonDatasetType.PARQUET,
    ):
        ds = DatasetLoader(path=path, **kwargs)
        ds.post_initialize()
        return ds
    elif ds_type == EnergonDatasetType.FILESYSTEM:
        raise ValueError("Filesystem datasets can only be used as auxiliary datasets.")
    else:
        raise ValueError(f"Invalid dataset at {path}")


def traverse_metadataset(
    path: Union[str, EPath, Path],
    *,
    split_part: str,
    **kwargs,
) -> list[TraversedDatasetReference]:
    """Traverse one metadataset split and return flattened leaf dataset references.

    This is the main public entrypoint for traversal-only inspection of a metadataset. It loads
    the root metadataset configuration, resolves nested metadatasets recursively, and returns the
    final leaf dataset references without constructing the intermediate scanned/traversed loader
    tree.

    Args:
        path: Path to the metadataset YAML file to traverse.
        split_part: Split to traverse, such as `\"train\"`, `\"val\"`, or `\"test\"`.
        **kwargs: Additional keyword arguments forwarded to `load_config()` while loading the root
            metadataset object.

    Returns:
        A flattened list of `TraversedDatasetReference` values describing the reachable leaf
        datasets for the requested split.

    Raises:
        AssertionError: If `path` does not point to a metadataset.
    """

    path = EPath(path)
    ds_type = get_dataset_type(path)
    assert ds_type == EnergonDatasetType.METADATASET, (
        f"traverse_metadataset only supports metadatasets, got {ds_type} at {path}"
    )
    mds = load_config(
        path,
        default_type=Metadataset,
        default_kwargs=dict(path=path, **kwargs),
    )
    return mds.traverse(split_part=split_part)


class MockJsonParser(JsonParser):
    """Json Parser, which translates unknown objects to a mock class."""

    def _resolve_object(
        self,
        module_name: str,
        object_name: str,
        cls: type,
        is_type: bool,
        is_callable: bool,
        is_instantiating_class: bool,
        is_calling_function: bool,
    ) -> Any:
        try:
            return super()._resolve_object(
                module_name,
                object_name,
                cls,
                is_type,
                is_callable,
                is_instantiating_class,
                is_calling_function,
            )
        except ModuleNotFoundError:

            class MockObject(cls):
                def __init__(self, *_, **__):
                    pass

            if is_type or is_instantiating_class:
                return MockObject
            elif is_callable or is_calling_function:
                return MockObject


def prepare_metadataset(path: EPath):
    from megatron.energon.dataset_config import load_config
    from megatron.energon.metadataset.metadataset import Metadataset

    meta_ds = load_config(
        path,
        default_type=Metadataset,
        default_kwargs=dict(path=path),
        parser=MockJsonParser(strict=True),
    )
    meta_ds.post_initialize()

    meta_ds.prepare()
