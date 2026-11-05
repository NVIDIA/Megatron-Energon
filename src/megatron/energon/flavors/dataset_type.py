# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum

from megatron.energon.deprecated import deprecated_dir, deprecated_getattr
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.paths import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME

__all__ = deprecated_dir(__name__, ["EnergonDatasetType", "get_dataset_type", "is_recipe"])


class EnergonDatasetType(Enum):
    RECIPE = "recipe"
    METADATASET = "recipe"
    MANIFEST_DATASET = "manifest_dataset"
    JSONL = "jsonl"
    BINIDX = "binidx"
    PARQUET = "parquet"
    FILESYSTEM = "filesystem"
    INVALID = "invalid"


def is_recipe(path: EPath) -> bool:
    return path.name.endswith(".yaml") and path.is_file()


def get_dataset_type(path: EPath) -> EnergonDatasetType:
    """Return the dataset type for a top-level dataset path."""

    if is_recipe(path):
        return EnergonDatasetType.RECIPE

    from megatron.energon.flavors.dataset_factory_resolver import get_dataset_factory_resolver

    dataset_type, _provider = get_dataset_factory_resolver().get_type(path)
    if dataset_type is not None:
        return dataset_type
    if (path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME).is_file():
        return EnergonDatasetType.FILESYSTEM
    return EnergonDatasetType.INVALID


def __getattr__(name: str):
    return deprecated_getattr(__name__, name)


def __dir__() -> list[str]:
    return deprecated_dir(__name__, __all__)
