# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Union

from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.flavors.dataset_type import (
    EnergonDatasetType,
    get_dataset_type,
    is_recipe,
)
from megatron.energon.recipe.dataset_loader import DatasetLoader
from megatron.energon.recipe.loader_interface import (
    DatasetLoaderInterface,
    TraversedDatasetReference,
)
from megatron.energon.recipe.recipe import Recipe
from megatron.energon.typed_converter import JsonParser


def load_dataset(
    path: Union[str, EPath, Path],
    **kwargs,
) -> DatasetLoaderInterface:
    """Loads a dataset or recipe."""

    if isinstance(path, dict):
        recipe = load_config(
            path,
            default_type=Recipe,
            default_kwargs=dict(path=EPath("/dict"), **kwargs),
        )
        return recipe
    path = EPath(path)
    if is_recipe(path):
        recipe = load_config(
            path,
            default_type=Recipe,
            default_kwargs=dict(path=path, **kwargs),
        )
        recipe.post_initialize()
        return recipe
    ds = DatasetLoader(path=path, **kwargs)
    ds.post_initialize()
    return ds


def traverse_recipe(
    path: Union[str, EPath, Path],
    *,
    split_part: str,
    **kwargs,
) -> list[TraversedDatasetReference]:
    """Traverse one recipe split and return flattened leaf dataset references.

    This is the main public entrypoint for traversal-only inspection of a recipe. It loads
    the root recipe configuration, resolves nested recipes recursively, and returns the
    final leaf dataset references without constructing the intermediate scanned/traversed loader
    tree.

    Args:
        path: Path to the recipe YAML file to traverse.
        split_part: Split to traverse, such as `\"train\"`, `\"val\"`, or `\"test\"`.
        **kwargs: Additional keyword arguments forwarded to `load_config()` while loading the root
            recipe object.

    Returns:
        A flattened list of `TraversedDatasetReference` values describing the reachable leaf
        datasets for the requested split.

    Raises:
        AssertionError: If `path` does not point to a recipe.
    """

    path = EPath(path)
    ds_type = get_dataset_type(path)
    assert ds_type == EnergonDatasetType.RECIPE, (
        f"traverse_recipe only supports recipes, got {ds_type} at {path}"
    )
    recipe = load_config(
        path,
        default_type=Recipe,
        default_kwargs=dict(path=path, **kwargs),
    )
    return recipe.traverse(split_part=split_part)


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


def prepare_recipe(path: EPath):
    from megatron.energon.dataset_config import load_config
    from megatron.energon.recipe.recipe import Recipe

    recipe = load_config(
        path,
        default_type=Recipe,
        default_kwargs=dict(path=path),
        parser=MockJsonParser(strict=True),
    )
    recipe.post_initialize()

    recipe.prepare()
