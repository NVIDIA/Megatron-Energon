# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from typing import Any

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import BaseCoreDatasetFactory
from megatron.energon.flavors.dataset_factory_resolver import (
    PRIORITY_FIRST,
    PRIORITY_MANIFEST,
    PRIORITY_SINGLE_FILE,
    DatasetFactoryResolver,
)
from megatron.energon.flavors.dataset_type import EnergonDatasetType


class _FirstProvider:
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        return EnergonDatasetType.JSONL

    @classmethod
    def from_path(cls, path: EPath, **kwargs: Any) -> BaseCoreDatasetFactory:
        raise NotImplementedError


class _SingleFileProvider:
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        return EnergonDatasetType.PARQUET

    @classmethod
    def from_path(cls, path: EPath, **kwargs: Any) -> BaseCoreDatasetFactory:
        raise NotImplementedError


class _LateProvider:
    @classmethod
    def detect_path(cls, path: EPath) -> EnergonDatasetType | None:
        return EnergonDatasetType.BINIDX

    @classmethod
    def from_path(cls, path: EPath, **kwargs: Any) -> BaseCoreDatasetFactory:
        raise NotImplementedError


class TestDatasetFactoryResolver(unittest.TestCase):
    def test_resolves_providers_by_priority(self) -> None:
        resolver = DatasetFactoryResolver()
        resolver.register(
            _SingleFileProvider,
            priority=PRIORITY_SINGLE_FILE,
        )
        resolver.register(_LateProvider, priority=PRIORITY_MANIFEST + 100)
        resolver.register(_FirstProvider)

        dataset_type, provider = resolver.get_type(EPath("/tmp/example"))

        self.assertEqual(dataset_type, EnergonDatasetType.JSONL)
        self.assertIs(provider, _FirstProvider)

    def test_provider_registrations_are_priority_sorted(self) -> None:
        resolver = DatasetFactoryResolver()
        resolver.register(
            _SingleFileProvider,
            priority=PRIORITY_SINGLE_FILE,
        )
        resolver.register(_LateProvider, priority=PRIORITY_MANIFEST)
        resolver.register(_FirstProvider)

        priorities = [registration.priority for registration in resolver._providers]

        self.assertEqual(priorities, sorted(priorities))
        self.assertEqual(priorities[0], PRIORITY_FIRST)
        self.assertIn(PRIORITY_SINGLE_FILE, priorities)
        self.assertIn(PRIORITY_MANIFEST, priorities)

    def test_same_priority_preserves_registration_order(self) -> None:
        resolver = DatasetFactoryResolver()
        resolver.register(_FirstProvider)
        resolver.register(
            _SingleFileProvider,
            priority=PRIORITY_FIRST,
        )

        _dataset_type, provider = resolver.get_type(EPath("/tmp/example"))

        self.assertIs(provider, _FirstProvider)


if __name__ == "__main__":
    unittest.main()
