# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager
from threading import local
from weakref import WeakKeyDictionary

_context = local()


def in_thread_worker() -> bool:
    return getattr(_context, "fields", None) is not None


@contextmanager
def thread_worker_scope():
    """Give one loader thread its own dataset state and reader handles."""
    _context.fields = WeakKeyDictionary()
    try:
        yield
    finally:
        del _context.fields


class WorkerLocalFields:
    """Share dataset configuration while keeping the declared mutable fields worker-local."""

    _worker_local_fields = {}
    _savable_fields = ()

    def __getattribute__(self, name):
        fields = getattr(_context, "fields", None)
        if fields is not None and name not in {"_worker_local_fields", "_savable_fields"}:
            factories = super().__getattribute__("_worker_local_fields")
            if name in factories or name in super().__getattribute__("_savable_fields"):
                values = fields.setdefault(self, {})
                factory = factories.get(name)
                if name not in values and factory is not None:
                    values[name] = factory()
                if name in values:
                    return values[name]
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        fields = getattr(_context, "fields", None)
        if fields is not None and (
            name in self._worker_local_fields or name in self._savable_fields
        ):
            fields.setdefault(self, {})[name] = value
        else:
            super().__setattr__(name, value)
