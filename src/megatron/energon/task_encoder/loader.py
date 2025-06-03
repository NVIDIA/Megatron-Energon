# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, TypeVar, Union

from megatron.energon.epathlib import EPath
from megatron.energon.errors import warn_deprecated
from megatron.energon.flavors import BaseCoreDatasetFactory, SavableDataset
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
from megatron.energon.metadataset import load_dataset
from megatron.energon.task_encoder.base import DefaultTaskEncoder, TaskEncoder, WorkerConfig

T = TypeVar("T", covariant=True)


def _split_kwargs(kwargs: dict) -> dict:
    loader_kwargs = {}
    try:
        loader_kwargs["split_part"] = kwargs.pop("split_part")
    except KeyError:
        pass
    try:
        loader_kwargs["dataset_config"] = kwargs.pop("dataset_config")
    except KeyError:
        pass
    try:
        loader_kwargs["split_config"] = kwargs.pop("split_config")
    except KeyError:
        pass
    return loader_kwargs


def _split_deprecated_decoder_kwargs(kwargs: dict, task_encoder: TaskEncoder) -> None:
    """
    auto_decode: bool = True,
    image_decode: ImageDecoder = "torchrgb",
    ignore_decoder_errors: bool = False,
    av_decode: AVDecoder = "AVDecoder",
    video_decode_audio: bool = False,
    """
    auto_decode = True

    decoder_kwargs = {}
    if "auto_decode" in kwargs:
        auto_decode = kwargs.pop("auto_decode")
    if "image_decode" in kwargs:
        decoder_kwargs["image_decode"] = kwargs.pop("image_decode")
    if "av_decode" in kwargs:
        decoder_kwargs["av_decode"] = kwargs.pop("av_decode")
    if "video_decode_audio" in kwargs:
        decoder_kwargs["video_decode_audio"] = kwargs.pop("video_decode_audio")

    if not auto_decode:
        task_encoder.decoder = None
    elif len(decoder_kwargs) > 0:
        warn_deprecated(
            "The following decoder kwargs are deprecated and will be removed in a future version: "
            + ", ".join(decoder_kwargs.keys())
            + ". Instead, set the decoder directly in your task encoder."
        )

        assert (
            not hasattr(task_encoder, "decoder")
            or task_encoder.decoder is DefaultTaskEncoder.decoder
        ), "Task encoder already has a decoder, and setting using deprecated kwargs is not allowed."

        task_encoder.decoder = SampleDecoder(**decoder_kwargs)


def get_train_dataset(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["train"], str] = "train",
    worker_config: WorkerConfig,
    batch_size: Optional[int],
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    shuffle_buffer_size: Optional[int],
    max_samples_per_sequence: Optional[int],
    virtual_epoch_length: int = 0,
    shuffle_over_epochs_multiplier: Optional[int] = 1,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    repeat: bool = True,
    **kwargs,
) -> SavableDataset[T]:
    """
    Get training data loader with sensible defaults. See `get_dataset` for more details.

    The following recipe will be used:
      - :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
      - `task_encoder.encode_sample`
      - (:class:`megatron.energon.MixDataset` if mixing)
      - :class:`megatron.energon.BatchDataset` with `task_encoder.batch` for collation
      - `task_encoder.encode_batch`
      - :class:`megatron.energon.EpochizeDataset` (if `virtual_epoch_length` is set)

    Args:
        path: Path to the dataset.
        split_part: Default split part to use.
        worker_config: Worker configuration to use.
        batch_size: Size of a batch. If None, do not batch
        batch_drop_last: If true, drop the last batch if it is smaller than `batch_size`.
        shuffle_buffer_size: Size of the sample shuffle buffer (before task encoding).
        max_samples_per_sequence: If set, limit the number of samples per sample-sequence to this.
        virtual_epoch_length: If set, the dataset will be epochized to this length (=iterating
            will be suspended and the for-loop returns, next for-loop continues iterating).
            Otherwise, the dataset will loop indefinitely.
        shuffle_over_epochs_multiplier: Shuffle the shards over this many epochs.
        task_encoder: Task encoder to use.
        repeat: By default, the inner datasets will loop. If set to False, stop iteration after
            one epoch. Must only be set to False in conjunction with blend_epochized in the
            metadataset if one is used.
        cache_pool: If set, the cache pool to use for the dataset.
        **kwargs: Additional arguments to the dataset constructor.

    Returns:
        The dataloader.
    """

    loader = load_dataset(path, **_split_kwargs(kwargs))
    _split_deprecated_decoder_kwargs(kwargs, task_encoder)

    datasets = loader.get_datasets(
        training=True,
        split_part=split_part,
        worker_config=worker_config,
        max_samples_per_sequence=max_samples_per_sequence,
        shuffle_over_epochs_multiplier=shuffle_over_epochs_multiplier,
        decoder=task_encoder.decoder,
        **kwargs,
    )
    return task_encoder.build_train_datasets(
        datasets=datasets.datasets,
        worker_config=worker_config,
        batch_size=batch_size,
        batch_drop_last=batch_drop_last,
        packing_buffer_size=packing_buffer_size,
        virtual_epoch_length=virtual_epoch_length,
        shuffle_buffer_size=shuffle_buffer_size,
        blend_mode=datasets.blend_mode,
        repeat=repeat,
    )


def get_val_dataset(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["val", "test"], str] = "val",
    worker_config: WorkerConfig,
    batch_size: int,
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    limit: Optional[int] = None,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    **kwargs,
) -> SavableDataset[T]:
    """
    Get the validation/test dataset with sensible defaults. See `get_dataset` for more details.

    The following recipe will be used:
      - :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
      - `task_encoder.encode_sample`
      - (:class:`megatron.energon.MixDataset` if mixing)
      - :class:`megatron.energon.BatchDataset` with `task_encoder.batch` for collation
      - :class:`megatron.energon.LimitDataset` (if `limit` is set)
      - `task_encoder.encode_batch`

    Args:
        path: Path to the dataset.
        split_part: Default split part to use.
        worker_config: Worker configuration to use.
        batch_size: Size of a batch
        batch_drop_last: If true, drop the last batch if it is smaller than `batch_size`.
        limit: If set, limit the number of batches loaded from the dataset to this.
        task_encoder: Task encoder to use.
        **kwargs: Additional arguments to the dataset constructor.

    Returns:
        The loaded dataset.
    """
    _split_deprecated_decoder_kwargs(kwargs, task_encoder)
    loader = load_dataset(path, **_split_kwargs(kwargs))
    datasets = loader.get_datasets(
        training=False,
        split_part=split_part,
        worker_config=worker_config,
        decoder=task_encoder.decoder,
        **kwargs,
    )
    return task_encoder.build_val_datasets(
        datasets=datasets.datasets,
        worker_config=worker_config,
        batch_size=batch_size,
        batch_drop_last=batch_drop_last,
        packing_buffer_size=packing_buffer_size,
        limit=limit,
    )


def get_val_datasets(
    path: Union[str, EPath, Path],
    *,
    split_part: Union[Literal["val", "test"], str] = "val",
    worker_config: WorkerConfig,
    batch_size: int,
    batch_drop_last: bool = False,
    packing_buffer_size: Optional[int] = None,
    limit: Optional[int] = None,
    task_encoder: TaskEncoder[Any, Any, Any, T] = DefaultTaskEncoder(),
    **kwargs,
) -> List[Tuple[SavableDataset[T], BaseCoreDatasetFactory]]:
    """
    Get the validation/test dataset with sensible defaults. See `get_dataset` for more details.

    The following recipe will be used:
      - :func:`megatron.energon.dataset_config.get_dataset_from_config` (loads the raw dataset)
      - `task_encoder.encode_sample`
      - (:class:`megatron.energon.MixDataset` if mixing)
      - :class:`megatron.energon.BatchDataset` with `task_encoder.batch` for collation
      - :class:`megatron.energon.LimitDataset` (if `limit` is set)
      - `task_encoder.encode_batch`

    Args:
        path: Path to the dataset.
        split_part: Default split part to use.
        worker_config: Worker configuration to use.
        batch_size: Size of a batch
        batch_drop_last: If true, drop the last batch if it is smaller than `batch_size`.
        limit: If set, limit the number of batches loaded from the dataset to this.
        task_encoder: Task encoder to use.
        **kwargs: Additional arguments to the dataset constructor.

    Returns:
        The loaded val datasets, with the source datasets.
    """
    _split_deprecated_decoder_kwargs(kwargs, task_encoder)
    loader = load_dataset(path, **_split_kwargs(kwargs))
    datasets = loader.get_datasets(
        training=False,
        split_part=split_part,
        worker_config=worker_config,
        decoder=task_encoder.decoder,
        **kwargs,
    )
    return [
        (
            task_encoder.build_val_datasets(
                datasets=[dataset],
                worker_config=worker_config,
                batch_size=batch_size,
                batch_drop_last=batch_drop_last,
                packing_buffer_size=packing_buffer_size,
                limit=limit,
            ),
            dataset.dataset,
        )
        for dataset in datasets.datasets
    ]
