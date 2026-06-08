# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Protocol, Type, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.dataset_type import EnergonDatasetType


class DatasetFactoryProvider(Protocol):
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None: ...

    @classmethod
    def from_path(cls, path: EPath, **kwargs: Any) -> BaseCoreDatasetFactory: ...


class DatasetFactoryResolver:
    def __init__(self) -> None:
        self._providers: list[Type[DatasetFactoryProvider]] = []

        # Manifest datasets are handled manually last.
        from megatron.energon.flavors.base_manifest_dataset import (
            BaseManifestShardListDatasetFactory,
        )

        self._manifest_provider: Type[DatasetFactoryProvider] = BaseManifestShardListDatasetFactory

    def register(self, provider: Type[DatasetFactoryProvider]) -> None:
        self._providers.append(provider)

    def get_type(
        self, path: EPath
    ) -> tuple[EnergonDatasetType, Type[DatasetFactoryProvider]] | tuple[None, None]:
        for provider in self._providers:
            dataset_type = provider.detect_path(path)
            if dataset_type is not None:
                return dataset_type, provider
        # Manifest datasets are handled manually last.
        dataset_type = self._manifest_provider.detect_path(path)
        if dataset_type is not None:
            return dataset_type, self._manifest_provider
        return None, None

    def get(
        self,
        path: EPath,
        **kwargs: Any,
    ) -> BaseCoreDatasetFactory:
        _, provider = self.get_type(path)
        if provider is not None:
            return provider.from_path(path, **kwargs)
        raise ValueError(
            f"Path {path} does not contain a {MAIN_FOLDER_NAME}/.info.yaml or .info.json file, "
            f"nor is it a registered dataset type. "
            f"Did you forget to prepare the dataset? Please check the documentation for an "
            f"introduction to dataset preparation."
        )


_DATASET_FACTORY_RESOLVER: DatasetFactoryResolver | None = None


T_Provider = TypeVar("T_Provider", bound=DatasetFactoryProvider)


def register_dataset_factory_provider(
    provider: Type[T_Provider],
) -> Type[T_Provider]:
    get_dataset_factory_resolver().register(provider)
    return provider


def get_dataset_factory_resolver() -> DatasetFactoryResolver:
    global _DATASET_FACTORY_RESOLVER
    if _DATASET_FACTORY_RESOLVER is None:
        _DATASET_FACTORY_RESOLVER = DatasetFactoryResolver()
    return _DATASET_FACTORY_RESOLVER
