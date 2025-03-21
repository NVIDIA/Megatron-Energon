# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, Union

from megatron.energon.dataset_config import load_config
from megatron.energon.epathlib import EPath
from megatron.energon.metadataset.dataset_loader import DatasetLoader
from megatron.energon.metadataset.loader_interface import DatasetLoaderInterface
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
    if path.is_file():
        mds = load_config(
            path,
            default_type=Metadataset,
            default_kwargs=dict(path=path, **kwargs),
        )
        mds.post_initialize()
        return mds
    else:
        ds = DatasetLoader(path=path, **kwargs)
        ds.post_initialize()
        return ds


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
