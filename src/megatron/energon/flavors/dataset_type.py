# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.common.manifest.paths import INDEX_SQLITE_FILENAME, MAIN_FOLDER_NAME


class EnergonDatasetType(Enum):
    METADATASET = "metadataset"
    MANIFEST_DATASET = "manifest_dataset"
    JSONL = "jsonl"
    BINIDX = "binidx"
    PARQUET = "parquet"
    FILESYSTEM = "filesystem"
    INVALID = "invalid"


def is_metadataset(path: EPath) -> bool:
    return path.name.endswith(".yaml") and path.is_file()


def get_dataset_type(path: EPath) -> EnergonDatasetType:
    """Return the dataset type for a top-level dataset path."""

    if is_metadataset(path):
        return EnergonDatasetType.METADATASET

    from megatron.energon.flavors.dataset_factory_resolver import get_dataset_factory_resolver

    dataset_type, _provider = get_dataset_factory_resolver().get_type(path)
    if dataset_type is not None:
        return dataset_type
    if (path / MAIN_FOLDER_NAME / INDEX_SQLITE_FILENAME).is_file():
        return EnergonDatasetType.FILESYSTEM
    return EnergonDatasetType.INVALID
