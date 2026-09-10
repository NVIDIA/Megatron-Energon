# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

from megatron.energon.epathlib import EPath
from megatron.energon.eyaml import load_yaml
from megatron.energon.flavors import BaseCoreDatasetFactory
from megatron.energon.flavors.dataset_factory_resolver import get_dataset_factory_resolver
from megatron.energon.tags import resolve_tags
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

    return parser.raw_to_instance(data, default_type, default_kwargs=default_kwargs)


T_sample = TypeVar("T_sample", covariant=True)


def get_dataset_from_config(
    path: Union[EPath, Path, str],
    *,
    dataset_config: str | None = None,
    split_config: str | None = None,
    split_part: str | None = None,
    training: bool = True,
    tags: Optional[Dict[str, Any]] = None,
    subflavors: Optional[Dict[str, Any]] = None,
    worker_config: WorkerConfig,
    sample_type: Optional[Type[T_sample]] = None,
    filter_name: Optional[str] = None,
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
        tags: Merge-override the :attr:`Sample.__tags__` property of each sample.
        subflavors: Legacy alias for ``tags``. Specifying both raises an error.
        worker_config: If set, use this worker config instead of the default one.
        sample_type: Type of the samples to load, only used to ensure typing.
        filter_name: Name of the filter index sidecar to apply, if any.
        **kwargs: Additional arguments to be passed to the dataset constructor.

    Returns:
        The instantiated dataset
    """
    tags = resolve_tags(tags, subflavors)
    path = EPath(path)
    dataset = get_dataset_factory_resolver().get(
        path,
        dataset_config=dataset_config,
        split_config=split_config,
        split_part=split_part,
        training=training,
        tags=tags,
        worker_config=worker_config,
        sample_type=sample_type,
        filter_name=filter_name,
        **kwargs,
    )
    if tags is not None:
        dataset.tags.update(tags)
    if sample_type is not None:
        assert issubclass(dataset.__sample_type__, sample_type), (
            f"Sample of type {dataset.__sample_type__} is not a subclass of {sample_type}."
        )
    return dataset
