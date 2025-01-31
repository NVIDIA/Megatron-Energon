# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from torch.utils.data import IterableDataset

from megatron.energon.epathlib import EPath
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)


class PinMemoryMixin:
    """A mixin class providing a generic `pin_memory` function."""

    def _pin_memory(self, batch: T, device: Union[torch.device, str, None] = None) -> T:
        """Pin memory of a batch. Uses recursion to handle nested structures. Supports nested
        structures of dicts, dataclasses, namedtuples, lists and tuples."""
        if isinstance(batch, torch.Tensor):
            return batch.pin_memory(device)
        elif isinstance(batch, dict):
            return {key: self._pin_memory(value, device) for key, value in batch.items()}
        elif dataclasses.is_dataclass(batch):
            return type(batch)(
                **{
                    field.name: self._pin_memory(getattr(batch, field.name), device)
                    for field in dataclasses.fields(batch)
                }
            )
        elif isinstance(batch, (tuple, list)):
            if hasattr(batch, "_fields"):
                # NamedTuple
                return type(batch)(*[self._pin_memory(val, device) for val in batch])
            else:
                # list / tuple
                return type(batch)(self._pin_memory(val, device) for val in batch)
        else:
            return batch

    def pin_memory(self: T) -> T:
        return self._pin_memory(self)


class ExtendableDataclassMixin:
    """A mixin class providing a generic `extend` function for copying dataclasses."""

    @classmethod
    def extend(cls: Type[T], src, **kwargs) -> T:
        """
        Used for overridden dataclass instances. Example

        .. code-block:: python

            @dataclass
            class MyBaseClass:
                a: List[int]

            @dataclass
            class MyExtendedClass(MyBaseClass):
                # Add a new field `b` to the state
                b: List[int]

            base = MyBaseClass(a=[1, 2, 3])
            extended = MyExtendedClass.extend(base, b=[4, 5, 6])

        Args:
            src: The source dataclass instance to extend.
            **kwargs: The new fields to add to the instance to construct the new instance.

        Returns:
            The extended dataclass instance.
        """
        assert is_dataclass(cls), "Must be a dataclass"
        assert issubclass(cls, type(src)), "Cannot extend class of different type"

        for f in dataclasses.fields(src):
            if not f.init or f.type is ClassVar or typing.get_origin(f.type) is ClassVar:
                continue

            if f.name not in kwargs:
                kwargs[f.name] = getattr(src, f.name)
        return cls(**kwargs)


@dataclass(slots=True)
class Sample(ABC, PinMemoryMixin, ExtendableDataclassMixin):
    """An abstract base class for one element of a batch.
    Each task should derive a specific subclass as a `@dataclass`, like
    :class:`megatron.energon.CaptioningBatchSample`, and add the input and output fields as needed for
    training.
    """

    #: Uniquely identifies each sample in the dataset.
    __key__: str
    #: Key for restoring the sample. This is used to restore the sample from a checkpoint. It
    # should be a (nested) tuple of strings and integers, which can be used to index the dataset.
    __restore_key__: Tuple[Union[str, int, tuple], ...]

    #: A dataset may define a subflavor to distinguish between samples of the same sample type.
    __subflavor__: Optional[str]
    #: A dataset may define a subflavors to distinguish between samples of the same sample type.
    __subflavors__: Optional[Dict[str, Any]]

    @classmethod
    def derive_from(cls: Type[T_sample], base_sample: "Sample", **kwargs) -> T_sample:
        """
        Uses the base fields of `Sample` from base_sample (i.e. __key__, __restore_key__, __subflavor__, __subflavors__)
        and creates a new sample with the kwargs as fields. This is useful for creating new samples, while keeping the
        metadata of the base sample.

        Args:
            base_sample: The base sample to copy the base fields / metadata from.
            kwargs: The fields of the new sample.

        Returns:
            The new sample.
        """
        return cls(
            **{
                field.name: getattr(base_sample, field.name) for field in dataclasses.fields(Sample)
            },
            **kwargs,
        )

    @classmethod
    def from_joined(
        cls: Type[T_sample], *args: "Optional[Sample]", **kwargs: "Optional[Sample]"
    ) -> T_sample:
        """
        Creates a sample from joined samples. The samples are either passed as positional arguments or as keyword
        arguments. The first sample is the primary sample, which is used to initialize the key and subflavors.

        In the default implementation, the joined samples' fields will be joined together, such that latter joined
        samples will update the fields last (i.e. take precedence), except for the key and subflavors. The restore key
        is later set externally.

        Args:
            args: The samples to join (either this or kwargs is specified).
            kwargs: The samples to join (either this or args is specified). Not supported for the default
                implementation. Overwriting implementations may use this.

        Returns:
            The joined constructed sample.
        """
        assert (
            len(kwargs) == 0
        ), "Please specify joined datasets as list for the default joiner. Keyword arguments are confusing, because keys are ignored."
        excluded_fields = set(field.name for field in dataclasses.fields(Sample))
        init_args = {}
        if len(args) > 0:
            primary = args[0]
            assert primary is not None, "Primary sample must not be None."
            fields = dataclasses.fields(primary)
            for field in fields:
                init_args[field.name] = getattr(primary, field.name)
            for arg in args:
                if arg is None:
                    continue
                fields = dataclasses.fields(arg)
                for field in fields:
                    if field.name not in excluded_fields:
                        init_args[field.name] = getattr(arg, field.name)
        return cls(**init_args)


@dataclass(slots=True)
class State(ABC, ExtendableDataclassMixin):
    """An abstract base class for the state of a dataset. See :class:`megatron.energon.SavableDataset`.
    The state of a dataset is used to save and restore the dataset state (i.e. random generators,
    buffer states, file pointers, etc.).
    Each dataset should derive a specific subclass as a `@dataclass` and add the fields as needed
    for training.

    To extend subclasses, use the .extend method. Example:

    .. code-block:: python

        @dataclass
        class MyState(State):
            a: int

        @dataclass
        class MyExtendedState(MyState):
            # Add a new field `b` to the state
            b: int

        class MyStateSaver:
            def save_state(self) -> MyState:
                return MyState(a=42)

        class MyExtendedStateSaver(MyStateSaver):
            def save_state(self) -> MyExtendedState:
                # Fetch state from super class, which is already a complete instance (cannot add
                # new fields to it, type is fixed).
                state: MyState = super().save_state()

                # Now extend the state of the super class (of type `MyState`) with the new field
                # required to define `MyExtendedState`.
                return MyExtendedState.extend(state, b=21)
    """


@dataclass(slots=True)
class MergedState(ABC, ExtendableDataclassMixin):
    """An abstract base class for the merged state of a dataset. See :class:`SavableDataset`.
    The merged state is created in the :meth:`megatron.energon.SavableDataset.merge_states` method, and
    represents the merged state of all worker processes (only workers, not ranks). It is required
    to restore the state of the dataset in the :meth:`megatron.energon.SavableDataset.restore_state`
    method for all workers before the workers are started.
    Each dataset should derive a specific subclass as a `@dataclass` and add the fields as needed
    for training.

    To extend subclasses, use the .extend method. Example:
    Example (see :meth:`megatron.energon.State` to complete the example)

    .. code-block:: python

        @dataclass
        class MyMergedState(MergedState):
            a: List[int]

        @dataclass
        class MyExtendedMergedState(MyMergedState):
            # Add a new field `b` to the state
            b: List[int]

        class MyMergedStateSaver:
            def merge_state(self, states: List[MyState]) -> MyMergedState:
                return MyMergedState(a=[s.a for s in states])

        class MyExtendedMergedStateSaver(MyMergedStateSaver):
            def merge_state(self, states: List[MyExtendedState]) -> MyExtendedMergedState:
                # Fetch state from super class, which is already a complete instance (cannot add
                # new fields to it, type is fixed).
                state: MyMergedState = super().merge_state(states)

                # Now extend the state of the super class (of type `MyMergedState`) with the
                # new field required to define `MyExtendedMergedState`.
                return MyExtendedMergedState.extend(state, b=[s.b for s in states])
    """


class SavableDataset(IterableDataset[T_sample], Generic[T_sample], ABC):
    """A dataset that can be saved and restored (i.e. the random state, internal buffers, etc.).
    I.e. it can be resumed from a checkpoint.

    How dataset state saving works:

    1. The dataset state needs to be saved in all forked worker processes which contain a copy of
       the main dataset instance (see :class:`megatron.energon.SavableDataLoader`). Each worker returns
       only its own state.
    2. The main process merges the states via the :meth:`megatron.energon.SavableDataset.merge_states`
       method in the main process on the main dataset instance (which doesn't hold the worker states,
       as they were forked).
    3. The main process saves the merged state to the checkpoint.

    """

    worker_config: WorkerConfig

    def __init__(self, worker_config: WorkerConfig):
        self.worker_config = worker_config

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def save_state(self) -> State:
        """
        Saves the state of the dataset. This should include the random state, but not the data
        itself. Can only be called in a worker process.

        Returns:
            The state of the dataset as savable object (i.e. python basic types).
        """
        ...

    @abstractmethod
    def merge_states(self, states: List[Optional[State]]) -> MergedState:
        """
        Merges the states of all workers into one state. Restore state expects a merged state.

        Args:
            states: The states of the workers to merge.

        Returns:
            The merged state.
        """
        ...

    @abstractmethod
    def restore_state(self, state: Optional[MergedState]) -> None:
        """
        Restores the state of the dataset. This should include the random state, but not the data
        itself. Can only be called in a worker process.

        Args:
            state: The state of the dataset as savable object (i.e. python basic types) as saved by
                `save_state`. If None, restore initial state.
        """
        ...

    @abstractmethod
    def worker_has_samples(self) -> bool:
        """Returns True if the worker's split has samples. This is used to determine if this dataset
        yields anything."""
        ...

    @staticmethod
    def _function_config(fn: Callable) -> str:
        mod = inspect.getmodule(fn)
        if mod is not None:
            mod_name = mod.__name__
        else:
            mod_name = getattr(fn, "__module__", "<unknown>")
        return f"{mod_name}.{getattr(fn, '__qualname__', getattr(fn, '__name__', '<unknown>'))}"

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return a config dict that can be used to check if datasets have the same settings.
        Variables in dicts starting with "_" represent a possibly changable setting, like a full
        path which may be changed."""
        return {
            "type": type(self).__qualname__,
        }

    def can_restore_sample(self) -> bool:
        """Returns True if the dataset can restore a sample from a key."""
        return False

    def assert_can_restore(self) -> None:
        """Asserts that the dataset can restore a sample from a key."""
        assert self.can_restore_sample(), "This dataset cannot restore samples."

    def restore_sample(self, index: Tuple[Union[str, int, tuple], ...]) -> T_sample:
        """
        Generic key type, because it might be either an integer (for a core dataset), or something
        more complex (e.g. for blended datasets).

        Default raises an exception (assumed non-deterministic if not implemented, does not
        guarantee determinism).
        """
        raise NotImplementedError(
            "This dataset does not support indexing, because it is not safely deterministic."
        )


class BaseCoreDatasetFactory(Generic[T_sample], ABC):
    """Base type for an inner dataset sample loader. This factory can be used to construct a sample loader, or for
    joining in a joined dataset."""

    __sample_type__: Type[T_sample] = cast(Type[T_sample], None)
    paths: List[EPath]

    subflavor: Optional[str]
    subflavors: Dict[str, Any]

    @abstractmethod
    def build(self, worker_rotation_offset: int = 0) -> SavableDataset[T_sample]: ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset across all ranks."""
        ...


def add_sample_restore_key(
    sample: T_sample, *key: Union[int, str], src: Any, fail_otherwise: bool = False
) -> T_sample:
    """Adds a key to a sample. The sample must be a valid `Sample` or dict containing
    __restore_key__, which is a tuple of keys that can be used to restore the inner sample.
    This restore key is prepended with the `key`."""
    if isinstance(sample, Sample) or hasattr(sample, "__restore_key__"):
        try:
            sample.__restore_key__ = (type(src).__name__, *key, *sample.__restore_key__)
        except KeyError:
            pass
    elif isinstance(sample, dict) and "__restore_key__" in sample:
        sample["__restore_key__"] = (type(src).__name__, *key, *sample["__restore_key__"])
    elif fail_otherwise:
        raise RuntimeError(
            "Did not yield a sample with a restore key, but is marked stateless/deterministic."
        )
    return sample


def set_sample_restore_key(
    sample: T_sample, *key: Union[int, str], src: Any, fail_otherwise: bool = False
) -> T_sample:
    """Sets the restore key for a sample. The sample must be a valid `Sample` or dict containing
    __restore_key__, which is a tuple of keys that can be used to restore the inner sample.
    This restore key is prepended with the `key`."""
    if isinstance(sample, Sample) or hasattr(sample, "__restore_key__"):
        try:
            sample.__restore_key__ = (type(src).__name__, *key)
        except KeyError:
            pass
    elif isinstance(sample, dict) and "__restore_key__" in sample:
        sample["__restore_key__"] = (type(src).__name__, *key)
    elif fail_otherwise:
        raise RuntimeError(
            "Did not yield a sample with a restore key, but is marked stateless/deterministic."
        )
    return sample
