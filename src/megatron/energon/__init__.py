# Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from megatron.energon.flavors import (
    BaseCoreDataset,
    BaseWebdataset,
    CaptioningSample,
    CaptioningWebdataset,
    CrudeSample,
    CrudeWebdataset,
    DefaultDecoderWebdataset,
    DefaultGenericWebdataset,
    ImageClassificationSample,
    ImageClassificationWebdataset,
    ImageSample,
    ImageWebdataset,
    InterleavedSample,
    InterleavedWebdataset,
    MultiChoiceVQASample,
    MultiChoiceVQAWebdataset,
    OCRSample,
    OCRWebdataset,
    Sample,
    SavableDataset,
    SimilarityInterleavedSample,
    SimilarityInterleavedWebdataset,
    TextSample,
    TextWebdataset,
    VidQASample,
    VidQAWebdataset,
    VQAOCRWebdataset,
    VQASample,
    VQAWebdataset,
)
from megatron.energon.loader import get_loader, get_savable_loader
from megatron.energon.metadataset import (
    DatasetLoader,
    DatasetLoaderInterface,
    Metadataset,
    load_dataset,
)
from megatron.energon.savable_loader import SavableDataLoader
from megatron.energon.task_encoder import (
    AugmentTaskEncoder,
    Batch,
    DefaultTaskEncoder,
    TaskEncoder,
    batch_list,
    batch_pad_stack,
    batch_stack,
    generic_batch,
    get_train_dataset,
    get_val_dataset,
    get_val_datasets,
    stateless,
)
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BatchDataset,
    BlendDataset,
    ConcatDataset,
    EpochizeDataset,
    FilterDataset,
    GcDataset,
    GroupBatchDataset,
    IterMapDataset,
    LimitDataset,
    LogSampleDataset,
    MapDataset,
    MixBatchDataset,
    RepeatDataset,
    ShuffleBufferDataset,
    SkipSample,
    SliceBatchDataset,
    concat_pad,
    generic_concat,
    homogeneous_concat_mix,
)

__all__ = [
    "AugmentTaskEncoder",
    "BaseCoreDataset",
    "BaseWebdataset",
    "Batch",
    "BatchDataset",
    "BlendDataset",
    "CaptioningSample",
    "CaptioningWebdataset",
    "ConcatDataset",
    "Cooker",
    "CrudeSample",
    "CrudeWebdataset",
    "DatasetLoader",
    "DatasetLoaderInterface",
    "DefaultDecoderWebdataset",
    "DefaultGenericWebdataset",
    "DefaultTaskEncoder",
    "EpochizeDataset",
    "FilterDataset",
    "GcDataset",
    "GroupBatchDataset",
    "ImageClassificationSample",
    "ImageClassificationWebdataset",
    "ImageSample",
    "ImageWebdataset",
    "InterleavedSample",
    "InterleavedWebdataset",
    "IterMapDataset",
    "LimitDataset",
    "LogSampleDataset",
    "MapDataset",
    "Metadataset",
    "MixBatchDataset",
    "MultiChoiceVQASample",
    "MultiChoiceVQAWebdataset",
    "OCRSample",
    "OCRWebdataset",
    "RepeatDataset",
    "Sample",
    "SavableDataLoader",
    "SavableDataset",
    "SimilarityInterleavedSample",
    "SimilarityInterleavedWebdataset",
    "ShuffleBufferDataset",
    "SkipSample",
    "SliceBatchDataset",
    "TaskEncoder",
    "TextSample",
    "TextWebdataset",
    "VidQASample",
    "VidQAWebdataset",
    "VQASample",
    "VQAWebdataset",
    "VQAOCRWebdataset",
    "WorkerConfig",
    "basic_sample_keys",
    "batch_list",
    "batch_pad_stack",
    "batch_stack",
    "concat_pad",
    "generic_batch",
    "generic_concat",
    "get_loader",
    "get_savable_loader",
    "get_train_dataset",
    "get_val_dataset",
    "get_val_datasets",
    "homogeneous_concat_mix",
    "load_dataset",
    "stateless",
]
