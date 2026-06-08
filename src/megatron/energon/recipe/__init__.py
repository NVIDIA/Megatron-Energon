# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.deprecated import deprecated_dir, deprecated_getattr
from megatron.energon.recipe.dataset_loader import DatasetLoader
from megatron.energon.recipe.loader import (
    load_dataset,
    prepare_recipe,
    traverse_recipe,
)
from megatron.energon.recipe.loader_interface import DatasetLoaderInterface
from megatron.energon.recipe.recipe import MetadatasetV2, Recipe

__all__ = [
    "DatasetLoader",
    "load_dataset",
    "DatasetLoaderInterface",
    "Recipe",
    "MetadatasetV2",
    "prepare_recipe",
    "prepare_metadataset",
    "traverse_recipe",
    "traverse_metadataset",
]


def __getattr__(name: str):
    return deprecated_getattr(__name__, name)


def __dir__() -> list[str]:
    return deprecated_dir(__name__, __all__)
