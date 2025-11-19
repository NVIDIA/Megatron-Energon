# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml
from megatron.energon.flavors import (
    BaseCoreDatasetFactory,
    CrudeSample,
    DefaultCrudeJsonlDatasetFactory,
    StandardWebdatasetFactory,
)
from megatron.energon.flavors.webdataset.config import MAIN_FOLDER_NAME
from megatron.energon.flavors.webdataset.metadata import EnergonDatasetType, get_dataset_type
from megatron.energon.typed_converter import JsonParser
from megatron.energon.worker import WorkerConfig

T = TypeVar("T")


def load_config(
    path: Union[EPath, Dict[str, Any]],
    *,
    default_type: Type[T],
    default_kwargs: Optional[Dict[str, Any]] = None,
    parser: JsonParser = JsonParser(strict=True),
) -> T:
    """
    Loads a config from a file or directly from a dictionary.

    Args:
        path: Path to the config to load or a dictionary containing the config.
        default_type: If set, this is the type to use if no type is specified in the config.
        default_kwargs: Default kwargs to use, will be overridden by the config.

    Returns:
        The instantiated type.
    """
    if isinstance(path, dict):
        data = path
    else:
        # Read the config from a file
        with path.open("rb") as f:
            data: dict = load_yaml(f)

    if default_kwargs is not None:
        new_data = default_kwargs.copy()
        new_data.update(data)
        data = new_data

    return parser.raw_to_instance(data, default_type)


T_sample = TypeVar("T_sample", covariant=True)


def get_dataset_from_config(
    path: Union[EPath, Path, str],
    *,
    dataset_config: str | None = None,
    split_config: str | None = None,
    split_part: str | None = None,
    training: bool = True,
    subflavors: Optional[Dict[str, Any]] = None,
    worker_config: WorkerConfig,
    sample_type: Optional[Type[T_sample]] = None,
    **kwargs,
) -> BaseCoreDatasetFactory[T_sample]:
    """
    Gets a dataset from a config path or path to a jsonl file.

    Args:
        path: Path to the folder where the `.nv-meta` folder is contained, or path to a jsonl file.
        dataset_config: Filename of the dataset config file (`path / '.nv-meta' / config`), or None for jsonl datasets.
        split_config: Filename of the split config file (`path / '.nv-meta' / split_config`), or None for jsonl datasets.
        split_part: Name of the split to load, or None for jsonl datasets.
        training: If true, apply training randomization and loop the dataset.
        subflavors: Merge-Override the __subflavors__ property of each sample.
        worker_config: If set, use this worker config instead of the default one.
        sample_type: Type of the samples to load, only used to ensure typing.
        **kwargs: Additional arguments to be passed to the dataset constructor.

    Returns:
        The instantiated dataset
    """
    path = EPath(path)
    dataset: BaseCoreDatasetFactory[T_sample]
    ds_type = get_dataset_type(path)
    if ds_type == EnergonDatasetType.JSONL:
        assert sample_type is CrudeSample or sample_type is None, (
            f"Sample type must be CrudeSample for jsonl datasets, but got {sample_type}"
        )
        assert dataset_config is None, (
            f"Dataset config must be None for jsonl datasets, but got {dataset_config}"
        )
        assert split_config is None, (
            f"Split config must be None for jsonl datasets, but got {split_config}"
        )
        # Note: We ignore split_part for jsonl datasets and always return the full dataset.

        dataset = DefaultCrudeJsonlDatasetFactory(
            path,
            training=training,
            subflavors=subflavors,
            worker_config=worker_config,
            **kwargs,
        )
    elif ds_type == EnergonDatasetType.WEBDATASET:
        if dataset_config is None:
            dataset_config = "dataset.yaml"
        if split_config is None:
            split_config = "split.yaml"
        if split_part is None:
            split_part = "train"

        dataset = load_config(
            path / MAIN_FOLDER_NAME / dataset_config,
            default_kwargs=dict(
                path=path,
                split_config=split_config,
                split_part=split_part,
                training=training,
                worker_config=worker_config,
                **kwargs,
            ),
            default_type=StandardWebdatasetFactory,
        )
    elif ds_type == EnergonDatasetType.FILESYSTEM:
        raise ValueError("Filesystem datasets are only supported as auxiliary datasets. ")
    else:
        raise ValueError(
            f"Path {path} does not contain a {MAIN_FOLDER_NAME}/.info.yaml or .info.json file nor is it a jsonl file. "
            f"Did you forget to prepare the dataset? Please check the documentation for an introduction to dataset "
            f"preparation."
        )
    if subflavors is not None:
        dataset.subflavors.update(subflavors)
    if sample_type is not None:
        assert issubclass(dataset.__sample_type__, sample_type), (
            f"Sample of type {dataset.__sample_type__} is not a subclass of {sample_type}."
        )
    return dataset
