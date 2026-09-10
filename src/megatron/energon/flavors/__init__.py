# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.decoders import AVDecoderType, ImageDecoderType, SampleDecoder
from megatron.energon.flavors.base_dataset import (
    BaseCoreDatasetFactory,
    PinMemoryMixin,
    Sample,
    SavableDataset,
)
from megatron.energon.flavors.base_manifest_dataset import (
    BaseManifestDatasetFactory,
    BaseManifestShardListDatasetFactory,
)
from megatron.energon.flavors.binidx import BinIdxDatasetFactory, DefaultBinIdxDatasetFactory
from megatron.energon.flavors.captioning import CaptioningSample, CaptioningWebdataset
from megatron.energon.flavors.common.filter_index import (
    FilterIndex,
    FilterIndexWriter,
    build_filter_index,
    build_filter_index_from_global_indexes,
    build_filter_index_from_shard_indexes,
)
from megatron.energon.flavors.common.manifest.io import ShardListMeta
from megatron.energon.flavors.crude import CrudeSample, CrudeWebdataset
from megatron.energon.flavors.image import ImageSample, ImageWebdataset
from megatron.energon.flavors.image_classification import (
    ImageClassificationSample,
    ImageClassificationWebdataset,
)
from megatron.energon.flavors.interleaved import InterleavedSample, InterleavedWebdataset
from megatron.energon.flavors.jsonl import (
    CrudeJsonlDatasetFactory,
    CrudeJsonlShardListDatasetFactory,
    DefaultCrudeJsonlDatasetFactory,
    DefaultCrudeJsonlShardListDatasetFactory,
)
from megatron.energon.flavors.multichoice_vqa import MultiChoiceVQASample, MultiChoiceVQAWebdataset
from megatron.energon.flavors.ocr import OCRSample, OCRWebdataset
from megatron.energon.flavors.parquet.dataset import (
    DefaultParquetDatasetFactory,
    DefaultParquetShardListDatasetFactory,
    ParquetDatasetFactory,
    ParquetShardListDatasetFactory,
)
from megatron.energon.flavors.similarity_interleaved import (
    SimilarityInterleavedSample,
    SimilarityInterleavedWebdataset,
)
from megatron.energon.flavors.text import TextSample, TextWebdataset
from megatron.energon.flavors.vid_qa import VidQASample, VidQAWebdataset
from megatron.energon.flavors.vqa import VQASample, VQAWebdataset
from megatron.energon.flavors.vqa_and_ocr import VQAOCRWebdataset
from megatron.energon.flavors.webdataset import (
    AVData,
    AVDecoder,
    BaseWebdatasetFactory,
    DefaultDecoderWebdatasetFactory,
    DefaultGenericWebdatasetFactory,
    EmptyDatasetError,
    StandardWebdatasetFactory,
)

__all__ = [
    "AVData",
    "AVDecoder",
    "AVDecoderType",
    "BaseCoreDatasetFactory",
    "BaseManifestDatasetFactory",
    "BaseManifestShardListDatasetFactory",
    "BaseWebdatasetFactory",
    "BinIdxDatasetFactory",
    "CaptioningSample",
    "CaptioningWebdataset",
    "CrudeJsonlDatasetFactory",
    "CrudeJsonlShardListDatasetFactory",
    "CrudeSample",
    "CrudeWebdataset",
    "DefaultBinIdxDatasetFactory",
    "DefaultCrudeJsonlDatasetFactory",
    "DefaultCrudeJsonlShardListDatasetFactory",
    "DefaultDecoderWebdatasetFactory",
    "DefaultGenericWebdatasetFactory",
    "DefaultParquetDatasetFactory",
    "DefaultParquetShardListDatasetFactory",
    "EmptyDatasetError",
    "FilterIndex",
    "FilterIndexWriter",
    "ImageClassificationSample",
    "ImageClassificationWebdataset",
    "ImageDecoderType",
    "ImageSample",
    "ImageWebdataset",
    "InterleavedSample",
    "InterleavedWebdataset",
    "MultiChoiceVQASample",
    "MultiChoiceVQAWebdataset",
    "OCRSample",
    "OCRWebdataset",
    "PinMemoryMixin",
    "ParquetDatasetFactory",
    "ParquetShardListDatasetFactory",
    "Sample",
    "SampleDecoder",
    "SavableDataset",
    "SimilarityInterleavedSample",
    "SimilarityInterleavedWebdataset",
    "StandardWebdatasetFactory",
    "TextSample",
    "TextWebdataset",
    "VidQASample",
    "VidQAWebdataset",
    "VQAOCRWebdataset",
    "VQASample",
    "VQAWebdataset",
    "ShardListMeta",
    "build_filter_index",
    "build_filter_index_from_global_indexes",
    "build_filter_index_from_shard_indexes",
]
