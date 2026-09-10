# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Type, TypeVar, overload

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.flavors.common.manifest.paths import MAIN_FOLDER_NAME
from megatron.energon.flavors.dataset_type import EnergonDatasetType


class DatasetFactoryProvider(Protocol):
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None: ...

    @classmethod
    def from_path(cls, path: EPath, **kwargs: Any) -> BaseCoreDatasetFactory: ...


PRIORITY_FIRST: int = 0
PRIORITY_SINGLE_FILE: int = 100
PRIORITY_MANIFEST: int = 200


@dataclass(order=True, frozen=True, kw_only=True)
class _ProviderRegistration:
    priority: int
    provider: Type[DatasetFactoryProvider] = field(compare=False)


class DatasetFactoryResolver:
    def __init__(self) -> None:
        self._providers: list[_ProviderRegistration] = []

    def register(
        self,
        provider: Type[DatasetFactoryProvider],
        *,
        priority: int = PRIORITY_FIRST,
    ) -> None:
        """Register a dataset factory provider for path-based dataset detection.

        Args:
            provider: Dataset factory provider class to register.
            priority: Provider resolution priority. Lower values are checked first.
        """
        registration = _ProviderRegistration(priority=priority, provider=provider)
        index = bisect_right(self._providers, registration)
        self._providers.insert(index, registration)

    def get_type(
        self, path: EPath
    ) -> tuple[EnergonDatasetType, Type[DatasetFactoryProvider]] | tuple[None, None]:
        """Get the dataset type and provider for a path."""
        for registration in self._providers:
            provider = registration.provider
            dataset_type = provider.detect_path(path)
            if dataset_type is not None:
                return dataset_type, provider
        return None, None

    def get(
        self,
        path: EPath,
        **kwargs: Any,
    ) -> BaseCoreDatasetFactory:
        _, provider = self.get_type(path)
        if provider is None:
            raise ValueError(
                f"Path {path} does not contain a {MAIN_FOLDER_NAME}/.info.yaml or .info.json file, "
                f"nor is it a registered dataset type. "
                f"Did you forget to prepare the dataset? Please check the documentation for an "
                f"introduction to dataset preparation."
            )
        return provider.from_path(path, **kwargs)


_DATASET_FACTORY_RESOLVER: DatasetFactoryResolver | None = None


T_Provider = TypeVar("T_Provider", bound=DatasetFactoryProvider)


@overload
def register_dataset_factory_provider(
    provider: None = None,
    *,
    priority: int = PRIORITY_FIRST,
) -> Callable[[Type[T_Provider]], Type[T_Provider]]: ...


@overload
def register_dataset_factory_provider(
    provider: Type[T_Provider],
    *,
    priority: int = PRIORITY_FIRST,
) -> Type[T_Provider]: ...


def register_dataset_factory_provider(
    provider: Type[T_Provider] | None = None,
    *,
    priority: int = PRIORITY_FIRST,
) -> Type[T_Provider] | Callable[[Type[T_Provider]], Type[T_Provider]]:
    """Decorator for registering a dataset factory provider.

    Args:
        provider: Dataset factory provider class to register. When omitted, returns a
            decorator so keyword arguments such as ``priority`` can be supplied.
        priority: Provider resolution priority. Lower values are checked first. Defaults
            to ``PRIORITY_FIRST`` so custom providers take precedence over the built-in
            ``PRIORITY_SINGLE_FILE`` and ``PRIORITY_MANIFEST`` providers.
    """

    def decorator(provider: Type[T_Provider]) -> Type[T_Provider]:
        get_dataset_factory_resolver().register(provider, priority=priority)
        return provider

    if provider is None:
        return decorator
    return decorator(provider)


def get_dataset_factory_resolver() -> DatasetFactoryResolver:
    """Get the dataset factory resolver singleton."""
    global _DATASET_FACTORY_RESOLVER
    if _DATASET_FACTORY_RESOLVER is None:
        _DATASET_FACTORY_RESOLVER = DatasetFactoryResolver()
    return _DATASET_FACTORY_RESOLVER
