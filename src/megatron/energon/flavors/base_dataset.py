# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import inspect
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping
from copy import copy, deepcopy
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

from megatron.energon.cache import FileStore
from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath
from megatron.energon.savable import Savable
from megatron.energon.source_info import SourceInfo
from megatron.energon.state import FlexState
from megatron.energon.tags import SampleTagsAlias, TagsAlias, canonicalize_tag_kwargs
from megatron.energon.worker import WorkerConfig
from megatron.energon.worker_local import WorkerLocalFields

if typing.TYPE_CHECKING:
    from megatron.energon.flavors.common.manifest.types import ShardInfo

T_sample = TypeVar("T_sample", covariant=True)
T = TypeVar("T", covariant=True)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RestoreKey:
    """A typed address used to reconstruct a sample or batch."""

    def to_json_object(self) -> dict:
        """Return the key's type and fields for structured debug logging."""
        return {
            "type": type(self).__qualname__,
            **{
                field.name: _restore_key_json_value(getattr(self, field.name))
                for field in dataclasses.fields(self)
            },
        }


def _restore_key_json_value(value: Any) -> Any:
    if isinstance(value, RestoreKey):
        return value.to_json_object()
    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, slice):
        return {
            "type": "slice",
            "start": _restore_key_json_value(value.start),
            "stop": _restore_key_json_value(value.stop),
            "step": _restore_key_json_value(value.step),
        }
    if isinstance(value, (tuple, list)):
        return [_restore_key_json_value(item) for item in value]
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _restore_key_json_value(item) for key, item in value.items()}
        return {
            "type": "mapping",
            "entries": [
                [_restore_key_json_value(key), _restore_key_json_value(item)]
                for key, item in value.items()
            ],
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _restore_key_json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    # A custom packing slice may contain an opaque Python object.
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


class PinMemoryMixin:
    """A mixin class providing a generic `pin_memory` function."""

    def _pin_memory(self, batch: T, device: Union[torch.device, str, None] = None) -> T:
        """Pin memory of a batch. Uses recursion to handle nested structures. Supports nested
        structures of dicts, dataclasses, namedtuples, lists and tuples."""
        if isinstance(batch, torch.Tensor):
            return batch.pin_memory(device)
        elif isinstance(batch, RestoreKey):
            return batch
        elif getattr(type(batch), "pin_memory", None) is not PinMemoryMixin.pin_memory and callable(
            custom_pin := getattr(batch, "pin_memory", None)
        ):
            return custom_pin()
        elif isinstance(batch, Mapping):
            values = {key: self._pin_memory(value, device) for key, value in batch.items()}
            try:
                if isinstance(batch, MutableMapping):
                    result = copy(batch)
                    result.update(values)
                    return result
                return type(batch)(values)
            except TypeError:
                return values
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

    def pin_memory(self: Self) -> Self:
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
        kwargs = canonicalize_tag_kwargs(kwargs)
        assert is_dataclass(cls), "Must be a dataclass"
        assert issubclass(cls, type(src)), "Cannot extend class of different type"

        for f in dataclasses.fields(src):
            if not f.init or f.type is ClassVar or typing.get_origin(f.type) is ClassVar:
                continue

            if f.name not in kwargs:
                kwargs[f.name] = getattr(src, f.name)
        return cls(**kwargs)


@edataclass
class Sample(SampleTagsAlias, ABC, PinMemoryMixin, ExtendableDataclassMixin):
    """An abstract base class for one element of a batch.
    Each task should derive a specific subclass as a `@dataclass`, like
    :class:`megatron.energon.CaptioningBatchSample`, and add the input and output fields as needed for
    training.
    """

    #: Uniquely identifies each sample in the dataset.
    __key__: str
    #: Typed address for reconstructing this sample. None if the sample has no restore address.
    __restore_key__: RestoreKey | None

    #: A dataset may define tags to distinguish between samples of the same sample type.
    __tags__: Optional[Dict[str, Any]] = None

    #: Information about the source of the sample, i.e. where the data was loaded from.
    __sources__: Optional[tuple[SourceInfo, ...]] = None

    @classmethod
    def derive_from(cls: Type[T_sample], base_sample: "Sample", **kwargs) -> T_sample:
        """
        Uses the base fields of `Sample` from base_sample (i.e. __key__, __restore_key__, __tags__, __sources__)
        and creates a new sample with the kwargs as fields. This is useful for creating new samples, while keeping the
        metadata of the base sample.

        Args:
            base_sample: The base sample to copy the base fields / metadata from.
            kwargs: The fields of the new sample.

        Returns:
            The new sample.
        """
        kwargs = canonicalize_tag_kwargs(kwargs)
        base_kwargs = {
            field.name: getattr(base_sample, field.name)
            for field in dataclasses.fields(Sample)
            if field.name not in kwargs
        }
        return cls(
            **base_kwargs,
            **kwargs,
        )


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


class SavableDataset(WorkerLocalFields, IterableDataset[T_sample], Savable, Generic[T_sample], ABC):
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

    #: List of names of the fields that are saved and restored in the state.
    _savable_fields: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, worker_config: WorkerConfig):
        self.worker_config = worker_config

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

    def close(self) -> None:
        """Release resources owned by this dataset, such as open reader files.

        Leaf datasets that own readers should override this method. Dataset wrappers
        propagate closure to their children. Must be idempotent, and the dataset stays usable:
        readers reopen their files on the next access, so the dataset can be iterated again,
        e.g. by another loader.
        """

    def worker_init(self) -> None:
        """Initialize resources owned by the active loader worker."""

    def worker_close(self) -> None:
        """Release resources owned by the active loader worker."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

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

    @abstractmethod
    def reset_state_own(self) -> None:
        """Resets the state of the dataset to the initial state. Can only be called in a worker process."""
        ...

    def reset_state_deep(self) -> None:
        """Resets the state of the dataset to the initial state. Can only be called in a worker process."""
        self.reset_state_own()

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

    def restore_sample(self, restore_key: RestoreKey) -> T_sample:
        """
        Generic key type, because it might be either an integer (for a core dataset), or something
        more complex (e.g. for blended datasets).

        Default raises an exception (assumed non-deterministic if not implemented, does not
        guarantee determinism).
        """
        raise NotImplementedError(
            "This dataset does not support indexing, because it is not safely deterministic."
        )


class BaseCoreDatasetFactory(TagsAlias, Generic[T_sample], ABC):
    """Base type for an inner dataset sample loader."""

    __sample_type__: Type[T_sample] = cast(Type[T_sample], None)
    paths: List[EPath]

    tags: Dict[str, Any]

    @classmethod
    @abstractmethod
    def get_original_shards(cls, path: EPath) -> list["ShardInfo"]:
        """Return shards in source order, before splits, subsets, or filters.

        Each shard describes its name, primary file path, and original sample count.
        This must be the order in which filters will apply to the dataset.
        """
        ...

    @abstractmethod
    def build(
        self, worker_rotation_offset: int = 0, part_filter: Callable[[str], bool] | None = None
    ) -> SavableDataset[T_sample]:
        """Builds the dataset.

        Args:
            worker_rotation_offset: The offset of the worker rotation.
            part_filter: A function to filter the parts of the dataset.
        """
        ...

    @abstractmethod
    def as_file_store(self) -> "FileStore":
        """Returns the dataset as a random access dataset."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset across all ranks."""
        ...


def set_sample_restore_key(
    sample: T_sample, restore_key: RestoreKey | None, *, fail_otherwise: bool = False
) -> T_sample:
    """Attach a typed restore address to a sample or batch."""
    if isinstance(sample, dict) and "__restore_key__" in sample:
        sample["__restore_key__"] = restore_key
    elif hasattr(sample, "__restore_key__"):
        sample.__restore_key__ = restore_key
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
