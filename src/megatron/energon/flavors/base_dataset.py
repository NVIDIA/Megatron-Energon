# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
import threading
import typing
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import is_dataclass
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
from typing_extensions import Self

import megatron.energon
from megatron.energon.cache import FileStore
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.savable import Savable
from megatron.energon.source_info import SourceInfo
from megatron.energon.state import FlexState
from megatron.energon.worker import WorkerConfig

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)


class PinMemoryMixin:
    """A mixin class providing a generic `pin_memory` function."""

    @classmethod
    def sample_pin_memory(cls, batch: T, device: Union[torch.device, str, None] = None) -> T:
        """Pin memory of a batch. Uses recursion to handle nested structures. Supports nested
        structures of dicts, dataclasses, namedtuples, lists and tuples."""
        if hasattr(batch, "pin_memory"):
            return batch.pin_memory(device)
        if isinstance(batch, dict):
            return {key: cls.sample_pin_memory(value, device) for key, value in batch.items()}
        elif dataclasses.is_dataclass(batch):
            return type(batch)(
                **{
                    field.name: cls.sample_pin_memory(getattr(batch, field.name), device)
                    for field in dataclasses.fields(batch)
                }
            )
        elif not isinstance(batch, (str, bytes)) and isinstance(batch, (tuple, list)):
            if hasattr(batch, "_fields"):
                # NamedTuple
                return type(batch)(*[cls.sample_pin_memory(val, device) for val in batch])
            else:
                # list / tuple
                return type(batch)(cls.sample_pin_memory(val, device) for val in batch)
        else:
            return batch

    def pin_memory(self: Self, device: torch.device | str | None = None) -> Self:
        assert dataclasses.is_dataclass(self), "Must be a dataclass"
        return type(self)(
            **{
                field.name: self.sample_pin_memory(getattr(self, field.name), device)
                for field in dataclasses.fields(self)
            }
        )


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
        # assert issubclass(cls, type(src)), "Cannot extend class of different type"

        for f in dataclasses.fields(src):
            if not f.init or f.type is ClassVar or typing.get_origin(f.type) is ClassVar:
                continue

            if f.name not in kwargs:
                kwargs[f.name] = getattr(src, f.name)
        return cls(**kwargs)


@edataclass
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
    # May be None in some cases, but it may then not be restorable.
    __restore_key__: "RestoreKey | None"

    #: A dataset may define a subflavors to distinguish between samples of the same sample type.
    __subflavors__: Optional[Dict[str, Any]] = None

    #: Information about the source of the sample, i.e. where the data was loaded from.
    __sources__: Optional[tuple[SourceInfo, ...]] = None

    @classmethod
    def derive_from(cls: Type[T_sample], base_sample: "Sample", **kwargs) -> T_sample:
        """
        Uses the base fields of `Sample` from base_sample (i.e. __key__, __restore_key__, __subflavors__, __sources__)
        and creates a new sample with the kwargs as fields. This is useful for creating new samples, while keeping the
        metadata of the base sample.

        Args:
            base_sample: The base sample to copy the base fields / metadata from.
            kwargs: The fields of the new sample.

        Returns:
            The new sample.
        """
        base_kwargs = {
            field.name: getattr(base_sample, field.name)
            for field in dataclasses.fields(Sample)
            if field.name not in kwargs
        }
        return cls(
            **base_kwargs,
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
        assert len(kwargs) == 0, (
            "Please specify joined datasets as list for the default joiner. Keyword arguments are confusing, because keys are ignored."
        )
        excluded_fields = set(field.name for field in dataclasses.fields(Sample))
        init_args = {}
        if len(args) > 0:
            primary = args[0]
            assert primary is not None, "Primary sample must not be None."
            fields = dataclasses.fields(primary)
            for field in fields:
                init_args[field.name] = getattr(primary, field.name)
            # Merge sources from all joined samples
            init_args["__sources__"] = (
                *(primary.__sources__ or ()),
                *(
                    src
                    for arg in args
                    if arg is not None and arg.__sources__ is not None
                    for src in arg.__sources__
                ),
            )
            for arg in args:
                if arg is None:
                    continue
                fields = dataclasses.fields(arg)
                for field in fields:
                    if field.name not in excluded_fields:
                        init_args[field.name] = getattr(arg, field.name)
        return cls(**init_args)


@edataclass
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


THREAD_SAFE = True


class SavableDataset(IterableDataset[T_sample], Savable, Generic[T_sample], ABC):
    """A dataset that can be saved and restored (i.e. the random state, internal buffers, etc.).
    I.e. it can be resumed from a checkpoint.

    How dataset state saving works:

    1. The dataset state needs to be saved in all forked worker processes which contain a copy of
       the main dataset instance (see :class:`megatron.energon.DataLoader`). Each worker returns
       only its own state.
    2. The main process merges the states via the :meth:`megatron.energon.SavableDataset.merge_states`
       method in the main process on the main dataset instance (which doesn't hold the worker states,
       as they were forked).
    3. The main process saves the merged state to the checkpoint.

    """

    worker_config: WorkerConfig

    #: List of names of the fields that are saved and restored in the state.
    _savable_fields: ClassVar[Tuple[str, ...]] = ()

    #: List of names of the fields that are not saved, but are still part of the state (i.e. not shared between workers).
    _state_fields: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, worker_config: WorkerConfig):
        self.worker_config = worker_config
        if THREAD_SAFE:
            self._thread_state = threading.local()

    @abstractmethod
    def len_worker(self, worker_idx: int | None = None) -> int:
        """Returns the length of the dataset for the current or a specific worker.
        The length is the number of different available samples.
        The number of actually yielded samples may be different (considering skipping samples or generator functions).

        Args:
            worker_idx: The index of the worker to return the length for.
                If None, the length of the current worker is returned (must be in worker context).
        """
        ...

    def len_rank(self) -> int:
        """Returns the length of the dataset for the current rank.
        The length is the number of different available samples.
        The number of actually yielded samples may be different (considering skipping samples or generator functions).
        """
        return sum(self.len_worker(i) for i in range(self.worker_config.num_workers or 1))

    def __len__(self) -> int:
        """Returns the length of the dataset for the current rank. Corresponds to `len_rank`."""
        return self.len_rank()

    def save_state(self) -> FlexState:
        """
        Saves the state of the dataset. This will save and return the state of all fields
        in the _savable_fields tuple.
        Can only be called in a worker process.
        """

        state = FlexState()
        state["__class__"] = type(self).__name__
        for key in self._savable_fields:
            attr = getattr(self, key)
            if isinstance(attr, Savable):
                state[key] = attr.save_state()
            else:
                # Check if this field is a simple python type or a user class

                if attr is not None and getattr(attr, "__module__", "builtins") != "builtins":
                    import warnings

                    warnings.warn(
                        f"The savable attribute {key} of class {type(self)} does "
                        "not inherit from Savable, nor it is a simple builtin type. Please double-check.",
                        UserWarning,
                    )

                state[key] = deepcopy(getattr(self, key))

        return state

    def restore_state(self, state: FlexState) -> None:
        """
        Restores the state of the dataset. This will restore the state of all fields
        in the _savable_fields tuple.
        Can only be called in a worker process.

        Args:
            state: The state of the dataset as savable object. If None, restore initial state.
        """
        assert state["__class__"] == type(self).__name__, (
            f"Class name mismatch: {state['__class__']} != {type(self).__name__}"
        )

        for key in self._savable_fields:
            assert key in state, f"Key {key} not in state {state}"
            value = state.get(key)

            assert hasattr(self, key), f"Savable field {key} not in dataset {self}"
            if isinstance(getattr(self, key), Savable):
                getattr(self, key).restore_state(value)
            else:
                setattr(self, key, value)

    def reset_state(self) -> None:
        """
        Resets the state of the dataset. Called at least once in the worker process before iterating.
        Recursively resets the state of all wrapped datasets as well.
        """
        pass

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

    def restore_sample(self, restore_key: "RestoreKey") -> T_sample:
        """
        Restores a sample from a restore key.

        Args:
            restore_key: The restore key to restore the sample from.

        Returns:
            The restored sample.
        """
        raise NotImplementedError(
            "This dataset does not support restoring, because it is not safely deterministic."
        )

    if THREAD_SAFE:

        def __getattribute__(self, name: str) -> Any:
            if name in ("_savable_fields", "_state_fields", "_thread_state", "worker_config"):
                return object.__getattribute__(self, name)
            elif name in self._savable_fields or name in self._state_fields:
                try:
                    return getattr(self._thread_state, name)
                except AttributeError:
                    return object.__getattribute__(self, name)
            else:
                return object.__getattribute__(self, name)

        def __delattr__(self, name: str) -> None:
            if name in self._savable_fields or name in self._state_fields:
                delattr(self._thread_state, name)
            else:
                object.__delattr__(self, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in ("_savable_fields", "_state_fields", "_thread_state", "worker_config"):
                object.__setattr__(self, name, value)
            elif name in self._savable_fields or name in self._state_fields:
                setattr(self._thread_state, name, value)
            else:
                object.__setattr__(self, name, value)


class BaseCoreDatasetFactory(Generic[T_sample], ABC):
    """Base type for an inner dataset sample loader. This factory can be used to construct a sample loader, or for
    joining in a joined dataset."""

    __sample_type__: Type[T_sample] = cast(Type[T_sample], None)
    paths: List[EPath]

    subflavors: Dict[str, Any]

    @abstractmethod
    def build(self, worker_rotation_offset: int = 0) -> SavableDataset[T_sample]:
        """Builds the dataset."""
        ...

    @abstractmethod
    def as_file_store(self) -> "FileStore":
        """Returns the dataset as a random access dataset."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset across all ranks."""
        ...


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class RestoreKey(ABC):
    """Base class for restore keys."""

    def _tupleify(self, value: Any) -> Any:
        if isinstance(value, (int, str, float, bool)):
            return value
        elif isinstance(value, RestoreKey):
            return value.as_tuple()
        elif isinstance(value, (list, tuple)):
            return tuple(self._tupleify(v) for v in value)
        else:
            return value

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.__class__.__name__,
            *(self._tupleify(getattr(self, field.name)) for field in dataclasses.fields(self)),
        )

    @staticmethod
    def _untupleify(value: Any) -> Any:
        if isinstance(value, (int, str, float, bool)):
            return value
        elif isinstance(value, RestoreKey):
            return value.from_tuple(value)
        elif isinstance(value, (list, tuple)):
            if isinstance(value[0], str) and hasattr(megatron.energon, value[0]):
                return getattr(megatron.energon, value[0]).from_tuple(value[1:])
            else:
                return tuple(RestoreKey._untupleify(v) for v in value)

    @staticmethod
    def from_tuple(json: tuple[Any, ...]) -> "RestoreKey":
        cls = getattr(megatron.energon, json[0])
        kwargs = {}
        for field in dataclasses.fields(cls):
            kwargs[field.name] = RestoreKey._untupleify(json[1:])
        return cls(**kwargs)


def set_sample_restore_key(
    sample: T_sample, restore_key: RestoreKey, fail_otherwise: bool = False
) -> T_sample:
    """Sets the restore key for a sample. The sample must be a valid `Sample` or dict containing
    __restore_key__, which is a tuple of keys that can be used to restore the inner sample.
    This restore key is prepended with the `key`."""
    if isinstance(sample, Sample) or hasattr(sample, "__restore_key__"):
        try:
            sample.__restore_key__ = restore_key
        except KeyError:
            pass
    elif isinstance(sample, dict) and "__restore_key__" in sample:
        sample["__restore_key__"] = restore_key
    elif fail_otherwise:
        raise RuntimeError(
            "Did not yield a sample with a restore key, but is marked stateless/deterministic."
        )
    return sample


def legacy_handler(
    handler: Union[
        Callable[[Exception, Optional[str], Optional[list[SourceInfo]]], None],
        Callable[[Exception, Optional[str]], None],
    ],
) -> Callable[[Exception, Optional[str], Optional[list[SourceInfo]]], None]:
    """Safely returns the new style three argument handler. If the handler takes 2 arguments, it wraps it."""
    import functools
    import inspect

    handler_sig = inspect.signature(handler)
    if len(handler_sig.parameters) != 3:
        original_handler = handler

        @functools.wraps(original_handler)
        def wrapped_handler(
            exc: Exception, key: Optional[str], source_infos: Optional[list[SourceInfo]]
        ) -> None:
            return original_handler(exc, key)

        return wrapped_handler
    else:
        return handler
