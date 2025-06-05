# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import inspect
from abc import ABC
from dataclasses import is_dataclass
from types import MethodType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from typing_extensions import ParamSpec

from megatron.energon.cache import CachePool, DecodeFileStore, FileStore
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors import (
    CrudeSample,
    CrudeWebdataset,
    PinMemoryMixin,
    Sample,
    SampleDecoder,
    SavableDataset,
)
from megatron.energon.flavors.base_dataset import ExtendableDataclassMixin
from megatron.energon.metadataset.loader_interface import DatasetBlendMode, LoadedDataset
from megatron.energon.rng import SystemRng
from megatron.energon.source_info import SourceInfo
from megatron.energon.task_encoder.cooking import Cooker
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers import (
    BatchDataset,
    BlendDataset,
    ConcatDataset,
    EpochizeDataset,
    GroupBatchDataset,
    LimitDataset,
    LogSampleDataset,
    MapDataset,
    PackingDataset,
    ShuffleBufferDataset,
)
from megatron.energon.wrappers.repeat_dataset import RepeatDataset

T = TypeVar("T")
V = TypeVar("V")
T_sample = TypeVar("T_sample", bound=Sample)
T_encoded_sample = TypeVar("T_encoded_sample")
T_raw_batch = TypeVar("T_raw_batch")
T_batch = TypeVar("T_batch")


FeatureBatcher = Callable[[List[Any]], Any]


def generic_batch(batch: List[Any]) -> Any:
    """Based on the types/shapes of the batch: Will either pad and stack, or return as list.
    Recurses structures (dict, dataclass, namedtuple) and applies the same logic to each field."""
    if isinstance(batch[0], torch.Tensor):
        return batch_pad_stack(batch)
    elif isinstance(batch[0], dict):
        return {key: generic_batch([sample[key] for sample in batch]) for key in batch[0].keys()}
    elif is_dataclass(batch[0]):
        if hasattr(batch[0], "from_samples"):
            # The dataclass defines a method for batching samples
            return batch[0].from_samples(batch)
        return type(batch[0])(
            **{
                field.name: generic_batch([getattr(sample, field.name) for sample in batch])
                for field in dataclasses.fields(batch[0])
            }
        )
    elif isinstance(batch[0], tuple) and hasattr(batch[0], "_fields"):
        # NamedTuple
        return type(batch[0])(
            **{
                field: generic_batch([getattr(sample, field) for sample in batch])
                for field in batch[0]._fields
            }
        )
    else:
        return batch_list(batch)


def batch_stack(batch: List[Any]) -> Any:
    """Stack a batch of tensors."""
    return torch.stack(batch, dim=0)


def batch_pad_stack(batch: List[Any]) -> Any:
    """Stack a batch of arbitrary-sized tensors padded with 0s."""
    max_size = [max(b.shape[dim] for b in batch) for dim in range(batch[0].ndim)]
    batch_tensor = batch[0].new_zeros((len(batch), *max_size))
    for i, b in enumerate(batch):
        batch_tensor[(i, *(slice(0, s) for s in b.shape))] = b
    # Pad all tensors to max_size
    return batch_tensor


def batch_list(batch: List[Any]) -> Any:
    """Stack a batch of tensors padded with 0s."""
    return batch


P = ParamSpec("P")


@overload
def stateless(*, restore_seeds: bool = False) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


@overload
def stateless(fn: Callable[P, T]) -> Callable[P, T]: ...


def stateless(
    fn: Optional[Callable[..., T]] = None,
    *,
    restore_seeds: bool = False,
    failure_tolerance: Optional[int] = None,
) -> Union[Callable[[Callable[..., T]], Callable[..., T]], Callable[..., T]]:
    """Decorator to mark a function of the task encoder as restorable.

    Args:
        fn: The function to decorate.
        restore_seeds: Whether to restore the seeds for the function. I.e. the seeds are set
            from the sample index and the worker seed, such that they can be restored when a sample
            is restored from that function.
        failure_tolerance: The number of consecutive exceptions that are handled, after which a `FatalSampleError` is
            raised for this function.

    Usage:

    .. code-block:: python

        @stateless
        def encode_sample(self, sample: T_sample) -> T_encoded_sample:
            ...

        # Or if randomness is used (e.g. for augmentations):
        @stateless(restore_seeds=True)
        def encode_sample(self, sample: T_sample) -> T_encoded_sample:
            ...

    """

    if fn is None:
        return lambda f: stateless(
            f, restore_seeds=restore_seeds, failure_tolerance=failure_tolerance
        )
    if restore_seeds:
        worker_seed = None

        @functools.wraps(fn)
        def seed_wrapper_generator(self, *args, **kwargs):
            nonlocal worker_seed
            if worker_seed is None:
                worker_seed = WorkerConfig.active_worker_config.worker_seed()

            # Save the RNG states and set the new seed
            outer_rng_state = SystemRng.save_state()

            # Before constructing the generator and before the first
            # iteration, set inner RNG based on seed computed
            # from worker_seed and current sample index
            SystemRng.seed_args(worker_seed, self.current_sample_index)

            it = iter(fn(self, *args, **kwargs))

            inner_rand_state = None

            while True:
                if inner_rand_state is not None:
                    # Restore inner random state before calling the generator
                    # This will not be done on the first iteration
                    SystemRng.restore_state(inner_rand_state)

                try:
                    # Now call the generator. This will yield the sample
                    # But note it may also throw an exception or a StopIteration
                    sample = next(it)

                    # Save inner random state after calling the generator
                    inner_rand_state = SystemRng.save_state()
                except StopIteration:
                    # We're stopping here, but the outer random state
                    # will be restored before returning (in finally below)
                    break
                finally:
                    # Restore outer rand state before yielding or when an exception was raised
                    SystemRng.restore_state(outer_rng_state)

                # Now yield the sample.
                # This will give control back to the caller who may
                # change the random state.
                yield sample

                # Save outer random state after yielding
                outer_rng_state = SystemRng.save_state()

        @functools.wraps(fn)
        def seed_wrapper(self, *args, **kwargs):
            nonlocal worker_seed
            if worker_seed is None:
                worker_seed = WorkerConfig.active_worker_config.worker_seed()

            # Save the RNG states and set the new seed
            rng_state = SystemRng.save_state()

            SystemRng.seed_args(worker_seed, self.current_sample_index)

            try:
                return fn(self, *args, **kwargs)
            finally:
                # Restore the RNGs
                SystemRng.restore_state(rng_state)

        if inspect.isgeneratorfunction(fn):
            setattr(seed_wrapper_generator, "__stateless__", True)
            return seed_wrapper_generator
        else:
            setattr(seed_wrapper, "__stateless__", True)
            return seed_wrapper

    setattr(fn, "__stateless__", True)
    setattr(fn, "__failure_tolerance__", failure_tolerance)
    return fn


def get_stateless(fn: Callable) -> bool:
    """Get whether a function is stateless."""
    return getattr(fn, "__stateless__", False)


def get_failure_tolerance(
    fn: Callable, default_failure_tolerance: Optional[int] = None
) -> Optional[int]:
    """Get the failure tolerance of a function."""
    return getattr(fn, "__failure_tolerance__", default_failure_tolerance)


@edataclass
class Batch(PinMemoryMixin, ExtendableDataclassMixin):
    """Base class for a batch dataclass. Provides a default implementation for pinning memory.
    Additionally, it provides a future safe implementation for creating an instance from another
    batch `Batch.derive_from`."""

    #: Uniquely identifies each sample in the dataset.
    __key__: list[str]
    #: Key for restoring the sample. This is used to restore the sample from a checkpoint. It
    # should be a (nested) tuple of strings and integers, which can be used to index the dataset.
    __restore_key__: Tuple[Union[str, int, tuple], ...]

    #: A dataset may define a subflavors to distinguish between samples of the same sample type.
    __subflavors__: Optional[list[Optional[Dict[str, Any]]]] = None

    #: Information about the source of the sample, i.e. where the data was loaded from.
    __sources__: Optional[tuple[SourceInfo, ...]] = None

    @classmethod
    def derive_from(cls: Type[T_batch], base_batch: "Batch", **kwargs) -> T_batch:
        """
        Uses the base fields of `Batch` from base_batch (i.e. __key__, __restore_key__, __subflavors__, __sources__)
        and creates a new batch with the kwargs as fields. This is useful for creating new batches, while keeping the
        metadata of the base batch.

        Use like::

        .. code-block:: python

            def encode_batch(batch: RawBatch) -> Batch:
                return Batch.derive_from(batch, field1=batch.field1 + 1)

        Args:
            base_batch: The base batch to copy the base fields / metadata from.
            kwargs: The fields of the new batch.

        Returns:
            The new batch.
        """
        base_kwargs = {
            field.name: getattr(base_batch, field.name) for field in dataclasses.fields(Batch)
        }
        return cls(
            **base_kwargs,
            **kwargs,
        )

    @classmethod
    def from_samples(cls: Type[T_batch], samples: Sequence[Sample], **kwargs) -> T_batch:
        """
        Creates a batch from samples to be batched. Tensors will be padded and stacked, other types will be put into
        lists. This is the default implementation for `Batch.from_samples`.

        Args:
            samples: The samples to batch.
            kwargs: Additional (overriding) fields of the batch.

        Returns:
            The constructed batch.
        """
        assert all(dataclasses.is_dataclass(scls) for scls in samples), (
            "Samples must be dataclasses"
        )
        # assert dataclasses.is_dataclass(cls), "Batch must be dataclass"
        init_args = {}
        fields = dataclasses.fields(cls)
        for field in fields:
            if field.name in kwargs:
                init_args[field.name] = kwargs[field.name]
            elif field.name == "__sources__":
                if any(sample.__sources__ is not None for sample in samples):
                    # Special handling, needs flattening
                    init_args[field.name] = tuple(
                        source
                        for sample in samples
                        if sample.__sources__
                        for source in sample.__sources__
                    )
            elif field.name == "__subflavors__":
                if any(sample.__subflavors__ is not None for sample in samples):
                    init_args[field.name] = [
                        sample.__subflavors__ for sample in samples if sample.__subflavors__
                    ]
            else:
                value = [getattr(sample, field.name) for sample in samples]
                if len(samples) > 0 and isinstance(samples[0], torch.Tensor):
                    value = batch_pad_stack(value)
                init_args[field.name] = value
        return cls(**init_args)


class TaskEncoder(ABC, Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch]):
    """
    Base class for task encoders.

    Task encoding follows these steps:
      0. Data comes from the dataset
      1. :meth:`megatron.energon.TaskEncoder.encode_sample` / :meth:`megatron.energon.TaskEncoder.preencode_sample` is called on each sample
      2. :meth:`megatron.energon.TaskEncoder.select_samples_to_pack` is called on the buffer of samples
      3. :meth:`megatron.energon.TaskEncoder.postencode_sample` is called on each sample of the current pack
      4. :meth:`megatron.energon.TaskEncoder.pack_selected_samples` is called on the selected sample pack
      5. :meth:`megatron.energon.TaskEncoder.batch` is called on the list of encoded samples
      6. :meth:`megatron.energon.TaskEncoder.encode_batch` is called on the batch
      7. yield to main process
      8. :meth:`megatron.energon.Batch.to_device` is called on the encoded batch
      9. resulting encoded batch is passed to the network
    """

    __default_failure_tolerance__: Optional[int] = 100

    cookers: Sequence[Cooker[T_sample]] = ()
    #: Internal: List of registered cookers. Will be the same as `cookers` after registering cookers.
    _registered_cookers: List[Cooker[T_sample]]

    #: The decoder to use for decoding samples. Set manually as needed to override options.
    decoder: Optional[SampleDecoder] = SampleDecoder()

    @stateless
    def cook_crude_sample(
        self,
        sample: Union[T_sample, CrudeSample],
        get_primary_aux: Callable[[], FileStore],
        **aux: FileStore,
    ) -> T_sample:
        """
        Cooks a crude sample.

        Args:
            sample: The sample to cook.
            get_primary_aux: A function that returns the (cached) primary auxiliary dataset.
            **aux: The auxiliary side dishes to use for cooking.

        Returns: The cooked sample.
        """
        if isinstance(sample, CrudeSample):
            for cooker in self.cookers:
                if cooker.is_match(sample):
                    assert get_stateless(cooker.cook), "Cooker must be stateless"
                    if not cooker.need_primary and not cooker.need_cache:
                        kwargs = aux
                    else:
                        kwargs: dict = {}
                        if cooker.need_primary:
                            kwargs["primary"] = get_primary_aux()
                        kwargs.update(aux)
                        if cooker.need_cache:
                            kwargs["cache"] = self.cache
                    return cooker.cook(sample, **kwargs)

            raise NotImplementedError(
                "You are using crude samples but not providing a way to cook them: "
                f"Sample key={sample['__key__']}, subflavors={sample['__subflavors__']}, "
                f"self.cookers={self.cookers}"
            )
        else:
            assert isinstance(sample, Sample), "Sample must be a complete Sample or a CrudeSample"
            return sample

    def _is_overridden(
        self, bound_method: Callable[..., Any], bases: Optional[Sequence[Type[Any]]] = None
    ) -> bool:
        """Check if a method is overridden by a subclass of the base class(es).
        By default, only TaskEncoder is used as a base class.
        This is mainly used for optimization purposes. If the default method
        is a no-op, we can skip it entirely unless the user has overridden it.

        Args:
            bound_method: The method to check.
            bases: The base classes to check against.
        Returns:
            True if the method is overridden outside of TaskEncoder, False otherwise.
        """

        if not isinstance(bound_method, MethodType):
            # If the method is not bound, it is always overridden
            return True

        # Get the underlying function
        func = bound_method.__func__

        # Check if the subclass method matches any of the base class methods
        if bases is None:
            bases = (TaskEncoder,)
        return not any(getattr(base, func.__name__) is func for base in bases)

    @stateless
    def encode_sample(
        self, sample: T_sample
    ) -> Union[T_encoded_sample, Generator[T_encoded_sample, None, None]]:
        """Encode a single sample. May raise :exc:`megatron.energon.SkipSample` to skip a sample.
        Alternatively, this can be a generator that yields (or ignores) new samples.
        If this is defined, :func:`preencode_sample` and :func:`postencode_sample` must not be defined.
        """
        return sample

    @stateless
    def preencode_sample(
        self, sample: T_sample
    ) -> Union[T_sample, Generator[T_sample, None, None]]:
        """Pre-encode a single sample. May raise :exc:`megatron.energon.SkipSample` to skip a sample.
        Alternatively, this can be a generator that yields (or ignores) new samples.
        Use in conjunction with packing and caching.
        If this is defined, :func:`encode_sample` must not be defined.
        """
        return sample

    @stateless
    def postencode_sample(
        self, sample: T_sample
    ) -> Union[T_encoded_sample, Generator[T_encoded_sample, None, None]]:
        """Post-encode a single sample. May raise :exc:`megatron.energon.SkipSample` to skip a sample.
        Alternatively, this can be a generator that yields (or ignores) new samples.
        Use in conjunction with packing and caching.
        If this is defined, :func:`encode_sample` must not be defined.
        """
        return sample

    @stateless
    def batch(self, samples: List[T_encoded_sample]) -> T_raw_batch:
        """Move a batch to a device. May raise :exc:`megatron.energon.SkipSample` to skip a batch."""
        return self._batch(samples, type(samples[0]))

    def batch_group_criterion(self, sample: T_encoded_sample) -> Tuple[Hashable, Optional[int]]:
        """
        Return a group criterion for the sample. Default implementation does not group
        (effectively, it returns a single value `(None, None)`, thus only one group is used).
        Returns the key of the bucket to put this sample into, and the size of the bucket (=batch size).
        The bucket size must always be the same for the same bucket key.

        May raise :exc:`megatron.energon.SkipSample` to skip a batch.
        """
        return None, None

    @stateless
    def encode_batch(self, batch: T_raw_batch) -> Union[T_batch, Generator[T_batch, None, None]]:
        """Encode a batch of samples. May raise :exc:`megatron.energon.SkipSample` to skip a batch.
        Alternatively, this can be a generator that yields (or ignores) new batches."""
        return batch

    def _batch(
        self,
        samples: List[T_encoded_sample],
        result_type: Type[T_raw_batch],
        actions: Optional[Dict[str, FeatureBatcher]] = None,
        default_action: FeatureBatcher = generic_batch,
    ) -> T_raw_batch:
        """
        Batch a list of samples.

        Args:
            samples: The samples to batch
            result_type: Type of the result (might be dict, dataclass, or namedtuple)
            actions: For each field (=key), may specify a specific batcher
            default_action: The batcher to apply to all fields not in `action`

        Returns:
            The batched result
        """
        if dataclasses.is_dataclass(result_type) and hasattr(result_type, "from_samples"):
            return result_type.from_samples(samples)

        # Get dict of samples
        if isinstance(samples[0], dict):
            list_samples = {key: [sample[key] for sample in samples] for key in samples[0].keys()}
        elif is_dataclass(samples[0]):
            list_samples = {
                field.name: [getattr(sample, field.name) for sample in samples]
                for field in dataclasses.fields(samples[0])
            }
        elif isinstance(samples[0], tuple) and hasattr(samples[0], "_fields"):
            # NamedTuple
            list_samples = {
                field: [getattr(sample, field) for sample in samples]
                for field in samples[0]._fields
            }
        else:
            raise ValueError("Unrecognized sample type.")
        # Convert each field
        if actions is not None:
            list_samples = {
                key: default_action(value) if key not in actions else actions[key](value)
                for key, value in list_samples.items()
            }
        else:
            list_samples = {key: default_action(value) for key, value in list_samples.items()}
        # Construct result
        if issubclass(result_type, dict):
            return list_samples
        elif dataclasses.is_dataclass(result_type) or issubclass(result_type, tuple):
            # DataClass or NamedTuple
            return result_type(**list_samples)
        else:
            raise ValueError("Unrecognized result type.")

    def select_samples_to_pack(
        self, samples: List[T_encoded_sample]
    ) -> List[List[T_encoded_sample]]:
        """
        For packing, selects the samples to be packed together.
        Packing is only active when packing_buffer_size is set.
        Internally this stage is called "pre_packing".

        Args:
            samples: The samples to pre-pack. A full buffer will be passed into the function.

        Returns: The pre-packed samples as a list of lists of samples.
        """
        raise NotImplementedError("Packing only effective when overridden.")

    def pack_selected_samples(self, samples: List[T_encoded_sample]) -> T_encoded_sample:
        """
        Given one set of samples to pack, returns the final packed sample.
        Packing is only active when packing_buffer_size is set.
        Internally this stage is called "final_packing".

        Args:
            samples: The samples to pack into a single sample

        Returns: The final packed sample.
        """
        raise NotImplementedError("Packing only effective when overridden.")

    def build_batch(
        self,
        dataset: SavableDataset[T_encoded_sample],
        *,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        worker_config: WorkerConfig,
    ) -> SavableDataset[T_raw_batch]:
        """Applies the batcher to the dataset."""

        dataset: SavableDataset[Any]

        if packing_buffer_size is not None:
            select_samples_to_pack_provided = self._is_overridden(self.select_samples_to_pack)
            pack_selected_samples_provided = self._is_overridden(self.pack_selected_samples)

            assert select_samples_to_pack_provided and pack_selected_samples_provided, (
                "Both select_samples_to_pack and pack_selected_samples methods must be provided in the TaskEncoder when using packing_buffer_size"
            )

            if self._is_overridden(self.postencode_sample):
                post_encode_fn = self.postencode_sample
            else:
                post_encode_fn = None

            dataset = PackingDataset(
                dataset,
                buffer_size=packing_buffer_size,
                pre_packer=self.select_samples_to_pack,
                final_packer=self.pack_selected_samples,
                final_packer_stateless=get_stateless(self.pack_selected_samples),
                sample_encoder=post_encode_fn,
                sample_encoder_stateless=True
                if post_encode_fn is None
                else get_stateless(post_encode_fn),
                worker_config=worker_config,
                pre_packer_failure_tolerance=get_failure_tolerance(
                    self.select_samples_to_pack, self.__default_failure_tolerance__
                ),
                final_packer_failure_tolerance=get_failure_tolerance(
                    self.pack_selected_samples, self.__default_failure_tolerance__
                ),
                sample_encoder_failure_tolerance=None
                if post_encode_fn is None
                else get_failure_tolerance(post_encode_fn, self.__default_failure_tolerance__),
            )
        elif self._is_overridden(self.postencode_sample):
            dataset = MapDataset(
                dataset,
                self.postencode_sample,
                worker_config=worker_config,
                stateless_map_fn=get_stateless(self.postencode_sample),
                failure_tolerance=get_failure_tolerance(
                    self.postencode_sample, self.__default_failure_tolerance__
                ),
            )

        if self._is_overridden(self.batch_group_criterion):
            dataset = GroupBatchDataset(
                dataset,
                fixed_batch_size=batch_size,
                sample_group_key=self.batch_group_criterion,
                batcher=self.batch,
                drop_last=batch_drop_last,
                worker_config=worker_config,
                failure_tolerance=get_failure_tolerance(
                    self.batch, self.__default_failure_tolerance__
                ),
            )

            if self._is_overridden(self.encode_batch):
                dataset = MapDataset(
                    dataset,
                    self.encode_batch,
                    worker_config=worker_config,
                    stateless_map_fn=get_stateless(self.encode_batch),
                    failure_tolerance=get_failure_tolerance(
                        self.encode_batch, self.__default_failure_tolerance__
                    ),
                )
        else:
            # No grouping is active

            if batch_size is not None:
                dataset = BatchDataset(
                    dataset,
                    batch_size=batch_size,
                    batcher=self.batch,
                    batcher_stateless=get_stateless(self.batch),
                    drop_last=batch_drop_last,
                    worker_config=worker_config,
                    failure_tolerance=get_failure_tolerance(
                        self.batch, self.__default_failure_tolerance__
                    ),
                )

                if self._is_overridden(self.encode_batch):
                    dataset = MapDataset(
                        dataset,
                        self.encode_batch,
                        worker_config=worker_config,
                        stateless_map_fn=get_stateless(self.encode_batch),
                        failure_tolerance=get_failure_tolerance(
                            self.encode_batch, self.__default_failure_tolerance__
                        ),
                    )

        return dataset

    def build_cook_crude_sample(
        self,
        dataset: SavableDataset[Union[T_sample, dict]],
        *,
        worker_config: WorkerConfig,
        subflavors: Dict[str, Any],
        get_primary_aux: Callable[[], FileStore],
        aux: Optional[Dict[str, FileStore]] = None,
    ) -> SavableDataset[T_sample]:
        """Applies the sample cooker to the dataset if we have cookers registered."""

        assert self.cookers, "No cookers registered, but got crude dataset."

        if aux is not None and self.decoder is not None:
            aux = {k: DecodeFileStore(v, decoder=self.decoder) for k, v in aux.items()}

        # Cache the primary auxiliary dataset for this dataset, i.e. construct it once when needed
        primary_aux = None

        def _get_primary_aux():
            nonlocal primary_aux
            if primary_aux is None:
                try:
                    if aux is not None:
                        primary_aux = aux.get("primary")
                    if primary_aux is None:
                        primary_aux = get_primary_aux()
                    assert primary_aux is not None, "Primary auxiliary dataset must always exist"
                    if self.decoder is not None:
                        primary_aux = DecodeFileStore(primary_aux, decoder=self.decoder)
                except Exception as e:
                    # Make the exception throw through for the sample being loaded
                    raise SystemError("Error getting primary auxiliary dataset") from e
            return primary_aux

        if aux is not None:
            cook_fn = functools.partial(
                self.cook_crude_sample, get_primary_aux=_get_primary_aux, **aux
            )
        else:
            cook_fn = functools.partial(self.cook_crude_sample, get_primary_aux=_get_primary_aux)

        return MapDataset(
            dataset,
            cook_fn,
            worker_config=worker_config,
            stateless_map_fn=True,
            map_fn_config=dict(
                cookers=[
                    dict(
                        cook=SavableDataset._function_config(cooker.cook),
                        has_subflavors=cooker.has_subflavors,
                    )
                    for cooker in self.cookers
                ],
                subflavors=subflavors,
            ),
            failure_tolerance=get_failure_tolerance(cook_fn, self.__default_failure_tolerance__),
        )

    def _load_dataset(
        self, dataset: LoadedDataset, worker_rotation_offset: int, worker_config: WorkerConfig
    ) -> SavableDataset[T_sample]:
        """Loads a train dataset, optionally cooking the samples."""
        if dataset.dataset.__sample_type__ == CrudeSample:
            return self.build_cook_crude_sample(
                dataset.dataset.build(worker_rotation_offset=worker_rotation_offset),
                worker_config=worker_config,
                subflavors=dataset.dataset.subflavors,
                get_primary_aux=dataset.dataset.as_file_store,
                aux=dataset.aux,
            )
        else:
            assert dataset.aux is None, "Aux is not supported for non-crude datasets."
            return dataset.dataset.build(worker_rotation_offset=worker_rotation_offset)

    def build_encode_sample(
        self,
        dataset: SavableDataset[T_sample],
        *,
        worker_config: WorkerConfig,
    ) -> SavableDataset[T_encoded_sample]:
        """Applies the sample encoder to the dataset."""
        if self._is_overridden(self.preencode_sample):
            pre_encode_fn = self.preencode_sample
            assert not self._is_overridden(
                self.encode_sample, bases=(TaskEncoder, DefaultTaskEncoder)
            ), "Cannot have both pre- and post-encode functions defined."
        elif self._is_overridden(self.encode_sample):
            pre_encode_fn = self.encode_sample
        else:
            pre_encode_fn = None
        if pre_encode_fn is not None:
            dataset = MapDataset(
                dataset,
                pre_encode_fn,
                worker_config=worker_config,
                stateless_map_fn=get_stateless(pre_encode_fn),
                failure_tolerance=get_failure_tolerance(
                    pre_encode_fn, self.__default_failure_tolerance__
                ),
            )
        return dataset

    def build_train_datasets(
        self,
        *,
        datasets: List[LoadedDataset],
        worker_config: WorkerConfig,
        batch_size: Optional[int],
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        virtual_epoch_length: int = 0,
        shuffle_buffer_size: Optional[int] = None,
        blend_mode: DatasetBlendMode = DatasetBlendMode.NONE,
        repeat: bool = True,
    ) -> SavableDataset[T_batch]:
        """Combines train datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset in datasets:
            if isinstance(dataset.dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        global_workers = max(1, worker_config.num_workers) * worker_config.world_size
        rotation_lengths = [len(dataset.dataset) for dataset in datasets]
        for i in range(1, len(rotation_lengths)):
            rotation_lengths[i] += rotation_lengths[i - 1]
        worker_rotation_offsets = [
            rotation_length % global_workers for rotation_length in [0] + rotation_lengths[:-1]
        ]

        if blend_mode == DatasetBlendMode.DATASET_WEIGHT:
            assert repeat, (
                "If repeat is False, the datasets can only be repeated or have no mode. Cannot blend with dataset weights."
            )
            inner_datasets = [
                (
                    RepeatDataset(
                        self._load_dataset(
                            dataset, worker_rotation_offset, worker_config=worker_config
                        ),
                        worker_config=worker_config,
                    ),
                    1.0 if dataset.weight is None else float(dataset.weight),
                )
                for dataset, worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]
            # Already repeating the inner datasets, so no need to repeat again
            repeat = False
        elif blend_mode == DatasetBlendMode.SAMPLE_REPETITIONS or (
            not repeat and blend_mode == DatasetBlendMode.NONE
        ):
            inner_datasets = [
                (
                    (
                        self._load_dataset(
                            dataset, worker_rotation_offset, worker_config=worker_config
                        )
                        if dataset.repetitions is None or dataset.repetitions == 1
                        else RepeatDataset(
                            self._load_dataset(
                                dataset, worker_rotation_offset, worker_config=worker_config
                            ),
                            repeats=dataset.repetitions,
                            worker_config=worker_config,
                        )
                    ),
                    len(dataset.dataset)
                    * (1 if dataset.repetitions is None else dataset.repetitions),
                )
                for dataset, worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]
        else:
            inner_datasets = [
                (
                    RepeatDataset(
                        self._load_dataset(
                            dataset, worker_rotation_offset, worker_config=worker_config
                        ),
                        worker_config=worker_config,
                    ),
                    1.0,
                )
                for dataset, worker_rotation_offset in zip(datasets, worker_rotation_offsets)
            ]
            # Already repeating the inner datasets, so no need to repeat again
            repeat = False

        if len(inner_datasets) > 1:
            # The worker offset for each dataset is the cumsum of the dataset lengths, but modulo the
            # global number of workers.
            dataset = BlendDataset(
                *[inner_dataset[:2] for inner_dataset in inner_datasets],
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = inner_datasets[0][0]
        else:
            raise ValueError("No datasets given.")
        if repeat:
            # Still need to repeat the dataset
            dataset = RepeatDataset(dataset, worker_config=worker_config)
        if shuffle_buffer_size is not None and shuffle_buffer_size > 1:
            dataset = ShuffleBufferDataset(
                dataset,
                size=shuffle_buffer_size,
                worker_config=worker_config,
            )
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if virtual_epoch_length > 0:
            dataset = EpochizeDataset(
                dataset,
                length=virtual_epoch_length,
                worker_config=worker_config,
            )
        if worker_config.should_log(level=1):
            dataset = LogSampleDataset(dataset, mode="train", worker_config=worker_config)

        return dataset

    def build_val_datasets(
        self,
        *,
        datasets: List[LoadedDataset],
        worker_config: WorkerConfig,
        batch_size: int,
        batch_drop_last: bool = False,
        packing_buffer_size: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> SavableDataset[T_batch]:
        """Combines val datasets to a single dataset."""

        # Check if there's a CrudeWebdataset but no cookers
        for dataset in datasets:
            if isinstance(dataset, CrudeWebdataset):
                assert self.cookers, "CrudeWebdataset found, but no cookers registered."

        global_workers = max(1, worker_config.num_workers) * worker_config.world_size
        rotation_lengths = [len(dataset.dataset) for dataset in datasets]
        for i in range(1, len(rotation_lengths)):
            rotation_lengths[i] += rotation_lengths[i - 1]
        worker_rotation_offsets = [
            rotation_length % global_workers for rotation_length in [0] + rotation_lengths[:-1]
        ]

        if len(datasets) > 1:
            dataset = ConcatDataset(
                *[
                    self._load_dataset(dataset, worker_rotation_offset, worker_config)
                    for dataset, worker_rotation_offset in zip(datasets, worker_rotation_offsets)
                ],
                worker_config=worker_config,
            )
        elif len(datasets) == 1:
            dataset = self._load_dataset(datasets[0], worker_rotation_offsets[0], worker_config)
        else:
            raise ValueError("No datasets given.")
        dataset = self.build_encode_sample(dataset, worker_config=worker_config)
        dataset = self.build_batch(
            dataset,
            batch_size=batch_size,
            batch_drop_last=batch_drop_last,
            packing_buffer_size=packing_buffer_size,
            worker_config=worker_config,
        )
        if limit is not None and limit > 0:
            dataset = LimitDataset(
                dataset,
                length=limit,
                worker_config=worker_config,
                reset_after_epoch=True,
            )

        if worker_config.should_log(level=2):
            dataset = LogSampleDataset(dataset, mode="val", worker_config=worker_config)

        return dataset

    @property
    def current_batch_index(self) -> int:
        """Returns the current index for the next batch yielded from the current worker. Each batch
        on the current rank will get a strictly increasing unique number. Counting happens on each
        rank separately (i.e. each rank will get the same numbers for same batch index)."""
        assert WorkerConfig.active_worker_config is not None, (
            "The batch_index can only be fetched within the worker, and to be usable, you must use the get_(savable_)loader methods provided from the package."
        )
        return WorkerConfig.active_worker_config.active_worker_batch_index

    @property
    def current_sample_index(self) -> int:
        """Returns the current index for the next sample yielded from the current routine (e.g.
        for `encode_sample`, `batch`, or `encode_batch`). Each routine will get a number
        representing the number of calls to that function. Across workers, this number will be
        unique, but it is not synced across workers, thus it may raise in different intervals (e.g.
        if batching does not work the same for all batches). When restoring a sample, this number is
        also restored and can be relied on for deterministic randomness reproduction of a sample."""
        assert WorkerConfig.active_worker_config is not None, (
            "The batch_index can only be fetched within the worker, and to be usable, you must use the get_(savable_)loader methods provided from the package."
        )
        return WorkerConfig.active_worker_config.active_worker_sample_index

    @property
    def cache(self) -> CachePool:
        """Returns the cache pool to use for caching out sample data to disk (for use with cookers / aux file stores).
        This is set and configured externally by the loader."""
        assert WorkerConfig.active_worker_config is not None, (
            "The cache can only be fetched within the worker, and to be usable, you must use the get_(savable_)loader methods provided from the package."
        )
        assert WorkerConfig.active_worker_config._cache_pool is not None, (
            "Cache pool must be set by the loader."
        )
        return WorkerConfig.active_worker_config._cache_pool


class DefaultTaskEncoder(
    TaskEncoder[T_sample, T_encoded_sample, T_raw_batch, T_batch],
    ABC,
    Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch],
):
    """The default task encoder supports automagically mapping to target types.
    You may override any methods to customize the behavior. By default, `encode_sample` is the
    identity function, `batch` calls `\\_batch` with the type of the first sample, and `encode\\_batch`
    is also the identity function. If you set any of `encoded_sample_type`, `raw_batch_type` or
    `batch_type`, the corresponding method return that type, where it automatically maps the fields
    (by name) to your new type.
    """

    _encoded_sample_type: Optional[Type[T_encoded_sample]]
    _raw_batch_type: Optional[Type[T_raw_batch]]
    _batch_type: Optional[Type[T_batch]]

    def __init__(
        self,
        *,
        encoded_sample_type: Optional[Type[T_encoded_sample]] = None,
        raw_batch_type: Optional[Type[T_raw_batch]] = None,
        batch_type: Optional[Type[T_batch]] = None,
    ):
        """
        Initialize the default task encoder.
        Types may be:
          * A `@dataclass` class: Return that typed dataclass. Field names must match the input
            fields.
          * A `NamedTuple` class: Return that typed namedtuple. Field names must match the input
            fields.
          * `dict`: Simply return the input as dict with field names as keys.

        Args:
            encoded_sample_type: Type of encoded samples (before batching)
            raw_batch_type: Type of the batched samples (after batching)
            batch_type: Type of the encoded batched samples
            cache: Cache pool to use for caching. If not provided, a no-op cache pool will be used.
        """
        self._encoded_sample_type = encoded_sample_type
        self._raw_batch_type = raw_batch_type
        self._batch_type = batch_type

    @stateless
    def encode_sample(
        self, sample: T_sample
    ) -> Union[T_encoded_sample, Generator[T_encoded_sample, None, None]]:
        """Encode a single sample. The default implementation converts to the
        _encoded_sample_type."""
        if self._encoded_sample_type is None or isinstance(sample, self._encoded_sample_type):
            return sample
        if is_dataclass(sample):
            fields = {
                field.name: getattr(sample, field.name) for field in dataclasses.fields(sample)
            }
        elif isinstance(sample, tuple) and hasattr(sample, "_fields"):
            fields = {field: getattr(sample, field) for field in sample._fields}
        elif isinstance(sample, dict):
            fields = sample
        else:
            raise ValueError("Unrecognized sample type.")
        if issubclass(self._encoded_sample_type, dict):
            return fields
        elif dataclasses.is_dataclass(self._encoded_sample_type) or issubclass(
            self._encoded_sample_type, tuple
        ):
            # DataClass or NamedTuple
            return self._encoded_sample_type(**fields)
        else:
            raise ValueError("Unrecognized encoded sample type.")

    @stateless
    def batch(self, samples: List[T_encoded_sample]) -> T_raw_batch:
        """Batch a list of samples. The default implementation uses default batching to convert
        to _batch_type."""
        actions = None
        if isinstance(samples[0], Sample):
            actions = {
                "__subflavors__": lambda x: x,
            }
        return self._batch(
            samples,
            type(samples[0]) if self._raw_batch_type is None else self._raw_batch_type,
            actions=actions,
        )

    @stateless
    def encode_batch(self, batch: T_raw_batch) -> Union[T_batch, Generator[T_batch, None, None]]:
        """Encode a batch of samples. The default implementation converts to the
        _encoded_batch_type."""
        if self._batch_type is None or self._raw_batch_type == self._batch_type:
            return batch
        if is_dataclass(batch):
            fields = {field.name: getattr(batch, field.name) for field in dataclasses.fields(batch)}
        elif isinstance(batch, tuple) and hasattr(batch, "_fields"):
            fields = {field: getattr(batch, field) for field in batch._fields}
        elif isinstance(batch, dict):
            fields = batch
        else:
            raise ValueError("Unrecognized sample type.")
        if issubclass(self._batch_type, dict):
            return fields
        elif dataclasses.is_dataclass(self._batch_type) or issubclass(self._batch_type, tuple):
            # DataClass or NamedTuple
            return self._batch_type(**fields)
        else:
            raise ValueError("Unrecognized encoded sample type.")


class AugmentTaskEncoder(
    TaskEncoder[T_sample, T_encoded_sample, T_raw_batch, T_batch],
    Generic[T_sample, T_encoded_sample, T_raw_batch, T_batch],
):
    """Augment a task encoder with additional functionality. By default, delegates everything to the
    original task encoder."""

    def __init__(self, task_encoder: TaskEncoder[T_sample, T_encoded_sample, T_raw_batch, T_batch]):
        """Initialize the augmenting task encoder.

        Args:
            task_encoder: The delegate task encoder. All calls will by default be forwarded to this.
        """
        self._task_encoder = task_encoder

    @property
    def decoder(self) -> SampleDecoder:
        return self._task_encoder.decoder

    def encode_sample(self, sample: T_sample) -> T_encoded_sample:
        return self._task_encoder.encode_sample(sample)

    def batch(self, samples: List[T_encoded_sample]) -> T_raw_batch:
        return self._task_encoder.batch(samples)

    def encode_batch(self, batch_data: T_raw_batch) -> T_batch:
        return self._task_encoder.encode_batch(batch_data)
