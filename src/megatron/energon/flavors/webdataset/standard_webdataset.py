# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Generic, Type, TypeVar

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.base_dataset import Sample
from megatron.energon.flavors.webdataset.default_decoder_webdataset import (
    DefaultDecoderWebdatasetFactory,
)

T_sample = TypeVar("T_sample", covariant=True)


class StandardWebdatasetFactory(DefaultDecoderWebdatasetFactory[T_sample], Generic[T_sample]):
    """
    This dataset sample loader factory uses the sample type e.g. given from a dataset.yaml, and applies the default
    loading logic, which includes decoding images, videos and containers.
    """

    def __init__(
        self,
        path: EPath,
        *,
        sample_type: Type[T_sample],
        **kwargs,
    ):
        """
        Factory for the standard webdataset sample loader.

        Args:
            path: Path to the dataset (passed to parent)
            sample_type: Type of the sample to be loaded
            auto_decode: If true, use the default webdataset sample decoder.
            image_decode: This defines the decoding results.
            ignore_decoder_errors: If true, ignore errors when decoding.
            subflavors: Subflavors dictionary to set for all loaded samples.
            field_map: Mapping from the webdataset fields to the sample fields.
            sample_loader: Function to load the sample from the webdataset fields. May be a string
                in order to load a function from a module, or a callable directly.
            part_filter: Filter for the parts to load. May be a string in order to load a function
                from a module, or a callable directly.
            split_part: Which part to load (e.g. 'train', 'val', 'test').
            training: If true, apply shuffling and loop the dataset.
            worker_config: Configuration for the workers.
            shuffle_over_epochs: Only effective if training=True.
                How many epochs to shuffle over if training.
                If = 1, every sample is seen exactly once per epoch.
                If > 1, samples (or rather shard slices) are shuffled within this number of epochs
                (i.e. randomly selected without replacement).
                If -1, the shards are effectively shuffle over infinite epochs (i.e. shard slices
                are drawn with replacement).
            parallel_shard_iters: Number of parallel opened shards per worker, shuffling between.
            max_samples_per_sequence: Maximum number of samples per sequence (=how many samples
                    will be sequentially iterated).
            split_config: Config file to use for shard split definitions.
            handler: Exception handler. Args: (exception, key).
        """
        self.__sample_type__ = sample_type
        assert issubclass(sample_type, Sample), (
            f"sample_type must be subclass of megatron.energon.Sample, got {sample_type.__qualname__}"
        )
        super().__init__(path, **kwargs)
