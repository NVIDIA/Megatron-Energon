# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from copy import deepcopy
from threading import local
from typing import Any, Callable, ClassVar
from weakref import ref

_context = local()
_DELETED = object()


def in_thread_worker() -> bool:
    return getattr(_context, "fields", None) is not None


@contextmanager
def thread_worker_scope():
    """Give one loader thread its own dataset state and reader handles."""
    previous = getattr(_context, "fields", None)
    _context.fields = {}
    try:
        yield
    finally:
        _context.fields = previous


def _fields_for(fields, obj):
    """Index by identity so user-defined equality and hashing cannot alias workers' state."""
    key = id(obj)
    entry = fields.get(key)
    if entry is not None and entry[0]() is obj:
        return entry[1]

    def forget(reference):
        current = fields.get(key)
        if current is not None and current[0] is reference:
            del fields[key]

    values = {}
    fields[key] = (ref(obj, forget), values)
    return values


class WorkerLocalFields:
    """Share configuration while isolating mutable state in each loader thread.

    Declare field names in a tuple, or map names to factories for values that need
    explicit initialization. A field without a factory starts with its own copy
    of the instance default, unless the worker has already assigned a value.
    """

    _worker_local_fields: ClassVar[tuple[str, ...] | dict[str, Callable[[], Any] | None]] = {}
    _worker_field_factories: ClassVar[dict[str, Callable[[], Any] | None]] = {}
    _savable_fields: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        factories = {}
        for base in reversed(cls.__mro__):
            fields = vars(base).get("_worker_local_fields", ())
            factories.update(fields if isinstance(fields, dict) else dict.fromkeys(fields))
        cls._worker_field_factories = factories

    def __getattribute__(self, name):
        fields = getattr(_context, "fields", None)
        if fields is not None and name not in {"_worker_field_factories", "_savable_fields"}:
            factories = super().__getattribute__("_worker_field_factories")
            if name in factories or name in super().__getattribute__("_savable_fields"):
                values = _fields_for(fields, self)
                if name not in values:
                    factory = factories.get(name)
                    values[name] = (
                        factory()
                        if factory is not None
                        else deepcopy(super().__getattribute__(name))
                    )
                value = values[name]
                if value is _DELETED:
                    raise AttributeError(name)
                return value
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        fields = getattr(_context, "fields", None)
        if fields is not None and (
            name in self._worker_field_factories or name in self._savable_fields
        ):
            _fields_for(fields, self)[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        fields = getattr(_context, "fields", None)
        if fields is not None and (
            name in self._worker_field_factories or name in self._savable_fields
        ):
            # Check that the attribute exists, as normal deletion would.
            getattr(self, name)
            _fields_for(fields, self)[name] = _DELETED
        else:
            super().__delattr__(name)
