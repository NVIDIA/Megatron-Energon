# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "AVData",
    "AVDecoder",
    "AVDecoderType",
    "BaseWebdatasetFactory",
    "DefaultDecoderWebdatasetFactory",
    "DefaultGenericWebdatasetFactory",
    "EmptyDatasetError",
    "ImageDecoderType",
    "SampleDecoder",
    "StandardWebdatasetFactory",
    "ShardListMeta",
]


def __getattr__(name: str):
    if name in {"AVData", "AVDecoder"}:
        from megatron.energon import av

        return getattr(av, name)
    if name == "ShardListMeta":
        from megatron.energon.flavors.common.manifest.io import ShardListMeta

        return ShardListMeta
    if name == "BaseWebdatasetFactory":
        from megatron.energon.flavors.webdataset.base_webdataset import BaseWebdatasetFactory

        return BaseWebdatasetFactory
    if name == "DefaultDecoderWebdatasetFactory":
        from megatron.energon.flavors.webdataset.default_decoder_webdataset import (
            DefaultDecoderWebdatasetFactory,
        )

        return DefaultDecoderWebdatasetFactory
    if name == "DefaultGenericWebdatasetFactory":
        from megatron.energon.flavors.webdataset.default_generic_webdataset import (
            DefaultGenericWebdatasetFactory,
        )

        return DefaultGenericWebdatasetFactory
    if name == "EmptyDatasetError":
        from megatron.energon.flavors.common.manifest.empty_dataset_error import EmptyDatasetError

        return EmptyDatasetError

    if name in {"AVDecoderType", "ImageDecoderType", "SampleDecoder"}:
        from megatron.energon import decoders

        return getattr(decoders, name)
    if name == "StandardWebdatasetFactory":
        from megatron.energon.flavors.webdataset.standard_webdataset import (
            StandardWebdatasetFactory,
        )

        return StandardWebdatasetFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
