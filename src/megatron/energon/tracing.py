from __future__ import annotations

import gc
import json
import os
import threading
import time
import weakref
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import torch

__all__ = [
    "TraceWriter",
    "Span",
    "AsyncSpan",
    "AsyncFlow",
    "ObjectTrace",
    "NoopTraceWriter",
]

T = TypeVar("T")

_JSON_OPEN = b"[\n"
_JSON_NEXT = b",\n"
_JSON_CLOSE = b"]\n"


def _timestamp_us() -> int:
    """Return current time in micro-seconds as int."""
    # Use time_ns, such that it's synchronized between processes.
    return time.time_ns() // 1_000  # convert ns -> µs


class JsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays, torch tensors, and dataclasses."""

    def default(self, o: Any) -> Any:
        # Handle numpy arrays directly
        if isinstance(o, (np.ndarray, torch.Tensor)):
            try:
                return o.tolist()
            except Exception:
                return str(o)[:250]

        # Handle dataclasses
        if is_dataclass(o):
            return {"__type__": type(o).__name__, **asdict(o)}

        return super().default(o)


class TraceWriter(AbstractContextManager):
    """Chrome-trace writer with live-streaming capabilities.

    This helper produces trace logs that follow the *Trace Event Format* as
    consumed by Chrome's ``chrome://tracing`` and the Perfetto UI.  We output
    the simplest JSON variant – a flat **array of event objects** – because it
    can be concatenated on the fly.

    The public surface consists of one generic :py:meth:`emit` method that
    serialises an *event dictionary* directly plus a set of convenience
    helpers – :py:meth:`span`, :py:meth:`instant`, :py:meth:`async_begin`,
    :py:meth:`flow_start`, :py:meth:`counter`, :py:meth:`object_new`, … – that
    wrap the *phase* field (``ph``) semantics defined in the Chromium spec:

    ============  =============================================================
    Phase (``ph``)  Helper(s)
    ------------  -------------------------------------------------------------
    ``B``/``E``     :py:meth:`span` (or :pyclass:`Span` ctx-mgr)
    ``i``           :py:meth:`instant`
    ``b``/``n``/``e`` :py:meth:`async_begin`, :py:meth:`async_instant`,
                      :py:meth:`async_end` and the :pyclass:`AsyncSpan`
                      context-manager
    ``s``/``t``/``f`` :py:meth:`flow_start`, :py:meth:`flow_step`,
                      :py:meth:`flow_end`
    ``C``           :py:meth:`counter`
    ``N``/``O``/``D`` :py:meth:`object_new`, :py:meth:`object_snapshot`,
                      :py:meth:`object_delete` and :pyclass:`ObjectTrace`
    ============  =============================================================

    For further background on each event family refer to the *Event
    Descriptions* section in the Trace-Event specification.
    """

    _write_lock: threading.Lock
    _pid: int
    _events: int
    _closed: bool
    _stream: IO[bytes]
    _own_stream: Optional[IO[bytes]]
    _flush_interval: int
    _pending: int
    _log_level: int

    _global_next_id_lock: ClassVar[threading.Lock] = threading.Lock()
    _global_next_id: ClassVar[int] = 0

    def __init__(
        self,
        stream: Union[str, Path, IO[bytes]],
        *,
        pid: int | None = None,
        log_level: int = 0,
    ) -> None:
        self._pid = pid if pid is not None else os.getpid()
        self._events = 0
        self._closed = False
        self._write_lock = threading.Lock()

        if isinstance(stream, (str, Path)):
            # Ensure parent directory exists when stream is a path.
            path = Path(stream)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = path.open("wb+")
            buffering = os.stat(stream).st_blksize
            self._own_stream = self._stream
            self._flush_interval = int(buffering * 0.8)
        else:
            self._stream = stream
            self._own_stream = None
            try:
                buffering = os.stat(stream).st_blksize
            except Exception:
                buffering = 4096
            self._flush_interval = int(buffering * 0.8)

        self._pending = 0

        # logging level (lower is more verbose) — default 0
        self._log_level = log_level

        # Initialise the JSON array with a closing bracket so the file is
        # syntactically complete right away.
        self._stream.write(_JSON_OPEN + _JSON_CLOSE)
        self._stream.flush()

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------

    @classmethod
    def _next_id(cls) -> int:
        """Return a new unique identifier."""
        with cls._global_next_id_lock:
            cls._global_next_id += 1
            return cls._global_next_id

    def _write_raw(self, json_event: bytes, *, flush: bool = False) -> None:
        """Write raw *json_event* bytes keeping the trace JSON valid. Flushes the stream if needed.

        Args:
            json_event: A fully-serialised event as UTF-8 encoded JSON bytes.
            flush: If *True* the underlying stream is flushed after the write.
        """
        with self._write_lock:
            self._stream.seek(-len(_JSON_CLOSE), os.SEEK_END)
            if self._events > 0:
                json_event = _JSON_NEXT + json_event + _JSON_CLOSE
            else:
                json_event = json_event + _JSON_CLOSE
            self._stream.write(json_event)
            self._pending += len(json_event)
            if flush or self._pending >= self._flush_interval:
                self._stream.flush()
                self._pending = 0

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with self._write_lock:
                self._stream.flush()
                if self._own_stream is not None:
                    self._own_stream.close()
                    self._own_stream = None

    def flush(self) -> None:
        with self._write_lock:
            self._stream.flush()

    def _emit(self, event: Dict[str, Any]) -> None:
        """Serialize *event* mapping and append it to the trace.

        Args:
            event: A dictionary that already fulfills the Trace-Event schema
                expectations.
        """
        json_event = json.dumps(
            event, separators=(",", ":"), ensure_ascii=False, cls=JsonEncoder
        ).encode("utf-8")
        self._write_raw(json_event)
        self._events += 1

    # Convenience helpers --------------------------------------------------

    def duration_begin(
        self,
        name: str,
        *,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Emit a *duration* event pair.

        Args:
            name: Displayed slice name.
            cat: Optional comma-separated category list.
            args: Extra arguments object to attach to both *B* and *E* events.
            level: Logging level.
        """
        if level < self._log_level:
            return
        event = {
            "name": name,
            "ph": "B",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def duration_end(
        self,
        name: str,
        *,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Emit the *end* of a *duration* event pair (``ph='E'``).

        Args:
            name: Displayed slice name.
            cat: Optional comma-separated category list.
            args: Extra arguments object to attach to both *B* and *E* events.
            level: Logging level.
        """
        if level < self._log_level:
            return
        event = {
            "name": name,
            "ph": "E",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def span(
        self,
        name: str,
        *,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "Span":
        """Return a context manager capturing a *duration* event pair.

        Args:
            name: Displayed slice name.
            cat: Optional comma-separated category list.
            args: Extra arguments object to attach to both *B* and *E* events.
            level: Logging level.

        Returns:
            Span – a context manager emitting matching ``B``/``E`` events.
        """
        if level < self._log_level:
            return _NOOP_SPAN
        return Span(self, name=name, cat=cat, args=args)

    def instant(
        self,
        name: str,
        *,
        cat: str | None = None,
        scope: str = "t",
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Emit a zero-duration *instant* event (``ph='i'``).

        Args:
            name: Display name.
            cat: Optional categories.
            scope: Trace-viewer scope selector – ``t`` (thread), ``p`` (process)
                or ``g`` (global).
            args: Optional arguments payload.
            level: Logging level.
        """
        if level < self._log_level:
            return
        event = {
            "name": name,
            "ph": "i",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
            "s": scope,
        }
        if cat is not None:
            event["cat"] = cat
        if args:
            event["args"] = dict(args)
        self._emit(event)

    # Async events --------------------------------------------------------

    def async_begin(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> Union[int, str]:
        """Start a *nestable async* chain (``ph='b'``).

        Args:
            name: Event display name.
            id: Correlation identifier (int or str).
            cat: Optional categories.
            scope: Extra scope string to avoid id collisions.
            args: Optional argument object.
            level: Logging level.
        """
        if level < self._log_level:
            return
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "b",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if scope is not None:
            event["scope"] = scope  # avoid clash with "s" used by instant events
        if args:
            event["args"] = dict(args)
        self._emit(event)
        return id

    def async_instant(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Emit an *instant* step for a nestable async chain (``ph='n'``).

        Args:
            name: Event name.
            id: Correlation identifier.
            cat: Categories.
            scope: Optional scope string.
            args: Additional arguments.
            level: Logging level.
        """
        if level < self._log_level:
            return
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "n",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if scope is not None:
            event["scope"] = scope
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def async_end(
        self,
        name: str,
        *,
        id: Union[int, str],
        cat: str | None = None,
        scope: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Finish a *nestable async* chain (``ph='e'``).

        Args:
            id: Correlation identifier.
            cat: Categories.
            scope: Optional scope string.
            args: Additional arguments.
            level: Logging level.
        """
        if level < self._log_level:
            return
        event = {
            "ph": "e",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if scope is not None:
            event["scope"] = scope
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def async_span(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "AsyncSpan":
        """Return an *AsyncSpan* context-manager for a nestable async chain.

        Args:
            name: Display name.
            id: Correlation identifier to keep events together.
            cat: Categories.
            scope: Optional scope string.
            args: Arguments attached to the begin event.
            level: Logging level.

        Returns:
            AsyncSpan context manager.
        """
        if level < self._log_level:
            return _NOOP_ASYNC_SPAN
        if id is None:
            id = self._next_id()

        return AsyncSpan(
            self,
            name=name,
            id=id,
            cat=cat,
            scope=scope,
            args=args,
        )

    def async_flow(
        self,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        level: int = 0,
    ) -> "AsyncFlow":
        """Return an *AsyncFlow* context-manager for a nestable async chain.

        Args:
            id: Correlation identifier.
            cat: Categories.
            scope: Optional scope string.
            level: Logging level.
        """
        if level < self._log_level:
            return _NOOP_ASYNC_FLOW
        if id is None:
            id = self._next_id()

        return AsyncFlow(
            self,
            id=id,
            cat=cat,
            scope=scope,
        )

    # Counter events ------------------------------------------------------

    def counter(
        self,
        name: str,
        value: Union[int, float, Dict[str, Union[int, float]]],
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        level: int = 0,
    ) -> None:
        """Emit a numerical *counter* (``ph='C'``).

        Args:
            name: Counter track name.
            value: Either a single numeric value or a mapping of series-name to
                numeric value.
            id: Optional counter identifier (name+id pair becomes counter key).
            cat: Categories.
            level: Logging level.
        """
        if level < self._log_level:
            return
        if isinstance(value, Mapping):
            args_field = value
        else:
            args_field = {"value": value}
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "C",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
            "args": args_field,
        }
        if id is not None:
            event["id"] = id
        if cat is not None:
            event["cat"] = cat
        self._emit(event)

    # Object events -------------------------------------------------------

    def object_new(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        level: int = 0,
    ) -> None:
        """Emit an object creation event (``ph='N'``).

        Args:
            name: Object type/name displayed in UI.
            id: Unique identifier (e.g. pointer address or GUID).
            cat: Categories.
            scope: Optional scope string to avoid id clashes.
            level: Logging level.
        """
        if level < self._log_level:
            return
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "N",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if scope is not None:
            event["scope"] = scope
        self._emit(event)
        return id

    def object_snapshot(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        snapshot: Dict[str, Any],
        cat: str | None = None,
        scope: str | None = None,
        level: int = 0,
    ) -> None:
        """Emit an object *snapshot* (``ph='O'``).

        Args:
            name: Object name.
            id: Identifier matching a previously created object.
            snapshot: Arbitrary JSON-serialisable state payload.
            cat: Categories.
            scope: Optional scope string.
            level: Logging level.
        """
        if level < self._log_level:
            return
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "O",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
            "args": {"snapshot": dict(snapshot)},
        }
        if cat is not None:
            event["cat"] = cat
        if scope is not None:
            event["scope"] = scope
        self._emit(event)

    def object_delete(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        level: int = 0,
    ) -> None:
        """Emit an object deletion event (``ph='D'``).

        Args:
            name: Object name.
            id: Identifier.
            cat: Categories.
            scope: Optional scope string.
            level: Logging level.
        """
        if level < self._log_level:
            return
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "D",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": threading.get_ident(),
        }
        if cat is not None:
            event["cat"] = cat
        if scope is not None:
            event["scope"] = scope
        self._emit(event)

    # Helper --------------------------------------------------------------

    def object_trace(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        scope: str | None = None,
        snapshot: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "ObjectTrace":
        """Create an :class:`ObjectTrace` helper.

        Args:
            name: Object type/name.
            id: Identifier to correlate with future snapshots/deletion.
            cat: Categories.
            scope: Optional scope string.
            snapshot: Optional initial snapshot emitted right after ``N``.
            level: Logging level.

        Returns:
            ObjectTrace instance.
        """
        if level < self._log_level:
            return _NOOP_OBJECT_TRACE
        if id is None:
            id = self._next_id()

        return ObjectTrace(
            self,
            name=name,
            id=id,
            cat=cat,
            scope=scope,
            initial_snapshot=snapshot,
        )

    def trace_object(
        self,
        obj: Any,
        name: str,
        *,
        cat: str | None = None,
        scope: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "ObjectTrace":
        """Attach tracing to an existing Python *obj* until GC.

        Args:
            obj: Target instance to monitor.
            name: Trace-viewer object name.
            cat: Categories.
            scope: Optional scope string.
            level: Logging level.

        Returns:
            ObjectTrace handle.
        """
        if not gc.is_tracked(obj):
            raise ValueError("Object is not tracked by the garbage collector")
        if level < self._log_level:
            return _NOOP_OBJECT_TRACE
        trace = self.object_trace(name, id=id(obj), cat=cat, scope=scope)
        weakref.finalize(obj, trace.delete)
        if args:
            trace.snapshot(args)
        return trace

    # Metadata ------------------------------------------------------------

    def metadata(
        self: "TraceWriter",
        name: str,
        *,
        args: Dict[str, Any],
        pid: int | None = None,
        tid: int | None = None,
    ) -> None:
        """Emit a generic *metadata* event (``ph='M'``).

        Args:
            name: Metadata event name (e.g. ``process_name``).
            args: Arguments dict as required by the spec.
            pid: Override process id; defaults to writer.pid.
            tid: Thread id; required for thread metadata.
        """
        event = {
            "name": name,
            "ph": "M",
            "pid": pid if pid is not None else self._pid,
        }
        if tid is not None:
            event["tid"] = tid
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def metadata_process_name(self, name: str, *, pid: int | None = None) -> None:
        self.metadata("process_name", args=dict(name=name), pid=pid)

    def metadata_process_labels(self, labels: str, *, pid: int | None = None) -> None:
        self.metadata("process_labels", args=dict(labels=labels), pid=pid)

    def metadata_process_sort_index(self, sort_index: int, *, pid: int | None = None) -> None:
        self.metadata("process_sort_index", args=dict(sort_index=sort_index), pid=pid)

    def metadata_thread_name(
        self, name: str, *, tid: int | None = None, pid: int | None = None
    ) -> None:
        self.metadata(
            "thread_name", args=dict(name=name), tid=tid or threading.get_ident(), pid=pid
        )

    def metadata_thread_sort_index(
        self, sort_index: int, *, tid: int | None = None, pid: int | None = None
    ) -> None:
        self.metadata(
            "thread_sort_index",
            args=dict(sort_index=sort_index),
            tid=tid or threading.get_ident(),
            pid=pid,
        )

    # Context management ---------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401, N802
        self.close()
        return False

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        status = "closed" if self._closed else f"{self._events} events"
        return f"<TraceWriter {status} pid={self._pid}>"


class Span(AbstractContextManager):
    """Context manager for *duration* events.

    See :py:meth:`TraceWriter.span`.
    """

    __slots__ = ("_writer", "_name", "_cat", "_args", "_begin_ts")
    _writer: TraceWriter
    _name: str
    _cat: Optional[str]
    _args: Dict[str, Any] | None

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._writer = writer
        self._name = name
        self._cat = cat
        self._args = args

    def begin(self) -> None:
        self._writer.duration_begin(self._name, cat=self._cat, args=self._args)
        self._args = None

    def update_args(self, args: Dict[str, Any]) -> None:
        if self._args is None:
            self._args = args
        else:
            self._args.update(args)

    def end(self) -> None:
        self._writer.duration_end(self._name, cat=self._cat, args=self._args or None)
        self._args = None

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self):  # noqa: D401
        self.begin()
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401, N802
        self.end()


class AsyncSpan(AbstractContextManager):
    """Context manager for *nestable async* events.

    Use :py:meth:`instant` for ``n`` events inside the span.
    """

    __slots__ = (
        "_writer",
        "_name",
        "_id",
        "_cat",
        "_scope",
        "_args",
    )

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        id: Union[int, str],
        cat: str | None = None,
        scope: str | None = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._writer = writer
        self._name = name
        self._id = id
        self._cat = cat
        self._scope = scope
        self._args = args

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def begin(self) -> None:
        self._writer.async_begin(
            self._name, id=self._id, cat=self._cat, scope=self._scope, args=self._args or None
        )
        self._args = None

    def __enter__(self):  # noqa: D401
        self.begin()
        return self

    def update_args(self, args: Dict[str, Any]) -> None:
        if self._args is None:
            self._args = args
        else:
            self._args.update(args)

    def end(self, args: Optional[Dict[str, Any]] = None) -> None:
        if self._args and args:
            self._args.update(args)
        self._writer.async_end(
            self._name, id=self._id, cat=self._cat, scope=self._scope, args=self._args or None
        )
        self._args = None

    def __exit__(self, exc_type, exc, tb):  # noqa: D401, N802
        self.end()


class AsyncFlow(AbstractContextManager):
    """Context manager for *nestable async* events."""

    __slots__ = (
        "_writer",
        "_id",
        "_cat",
        "_scope",
    )

    def __init__(
        self,
        writer: TraceWriter,
        *,
        id: Union[int, str],
        cat: str | None = None,
        scope: str | None = None,
    ) -> None:
        self._writer = writer
        self._id = id
        self._cat = cat
        self._scope = scope

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self):  # noqa: D401
        return self

    def instant(self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0) -> None:
        """Emit an async *instant* (``ph='n'``) event within this async flow."""
        self._writer.async_instant(
            name, id=self._id, cat=self._cat, scope=self._scope, args=args, level=level
        )

    def start(self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0) -> None:
        """Emit an async *start* (``ph='b'``) event within this async flow."""
        self._writer.async_begin(
            name, id=self._id, cat=self._cat, scope=self._scope, args=args, level=level
        )

    def end(self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0) -> None:
        """Emit an async *end* (``ph='e'``) event within this async flow."""
        self._writer.async_end(
            name, id=self._id, cat=self._cat, scope=self._scope, args=args, level=level
        )

    def span(
        self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0
    ) -> AsyncSpan:
        """Emit an async *span* (``ph='s'``) event within this async flow."""
        return self._writer.async_span(
            name, id=self._id, cat=self._cat, scope=self._scope, args=args, level=level
        )

    def iterable(self, iterable: Iterable[T], *, name: str, level: int = 0) -> Iterable[T]:
        """Wrap an iterable to emit trace events for each `next` call."""
        if level < self._writer._log_level:
            return iterable
        return IterableNextWrapper(iterable, span=lambda: self.span(name))

    def exception(self, exc: Exception, *, name: str, level: int = 0) -> None:
        """Emit an exception event."""
        self._writer.exception(
            exc, name=name, id=self._id, cat=self._cat, scope=self._scope, level=level
        )

    def __exit__(self, exc_type, exc, tb):  # noqa: D401, N802
        pass


class IterableNextWrapper(Iterator[T], Generic[T]):
    """A wrapper for an iterable that emits trace events for each `next` call."""

    __slots__ = (
        "_iterable",
        "_name",
    )

    _iterator: Iterator[T]
    _span: Callable[[], ContextManager]

    def __init__(self, iterable: Iterable[T], *, span: Callable[[], ContextManager]):
        self._iterator = iter(iterable)
        self._span = span

    def __iter__(self):
        return self

    def __next__(self):
        with self._span():
            return next(self._iterator)


class ObjectTrace(AbstractContextManager):
    """Lifecycle helper for Trace-Event objects.

    Emits ``N`` on construction, :py:meth:`snapshot` for ``O`` and ``D`` upon
    deletion, context exit, or garbage collection.
    """

    __slots__ = (
        "_writer",
        "_name",
        "_id",
        "_cat",
        "_scope",
        "_deleted",
    )

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        id: Union[int, str],
        cat: str | None = None,
        scope: str | None = None,
        initial_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._writer = writer
        self._name = name
        self._id = id
        self._cat = cat
        self._scope = scope
        self._deleted = False

        # Emit object creation event
        self._writer.object_new(name, id=id, cat=cat, scope=scope)

        if initial_snapshot is not None:
            self.snapshot(initial_snapshot)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def snapshot(self, data: Dict[str, Any], *, level: int = 0) -> None:
        """Emit snapshot for current state of the object."""
        if self._deleted:
            raise RuntimeError("Cannot snapshot deleted traced object")
        self._writer.object_snapshot(
            self._name,
            id=self._id,
            snapshot=data,
            cat=self._cat,
            scope=self._scope,
            level=level,
        )

    def delete(self) -> None:
        """Emit delete event if not already emitted."""
        if not self._deleted:
            self._writer.object_delete(
                self._name,
                id=self._id,
                cat=self._cat,
                scope=self._scope,
            )
            self._deleted = True

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401, N802
        self.delete()
        # Do not suppress exceptions
        return False

    def __del__(self):  # noqa: D401
        # Ensure deletion event when object garbage-collected
        try:
            self.delete()
        except Exception:
            pass


# ------------------------------------------------------------------
# Noop implementations
# ------------------------------------------------------------------


class _NoopSpan(AbstractContextManager):
    def begin(self, *args, **kwargs) -> None:
        pass

    def update_args(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


_NOOP_SPAN = cast(Span, _NoopSpan())


class NoopAsyncSpan(AbstractContextManager):
    def begin(self, *args, **kwargs) -> None:
        pass

    def update_args(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


_NOOP_ASYNC_SPAN = cast(AsyncSpan, NoopAsyncSpan())


class NoopAsyncFlow(AbstractContextManager):
    def instant(self, *args, **kwargs) -> None:
        pass

    def async_start(self, *args, **kwargs) -> None:
        pass

    def async_end(self, *args, **kwargs) -> None:
        pass

    def span(self, *args, **kwargs) -> AsyncSpan:
        return _NOOP_ASYNC_SPAN
    
    def iterable(self, iterable, *args, **kwargs) -> Iterable:
        return iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


_NOOP_ASYNC_FLOW = cast(AsyncFlow, NoopAsyncFlow())


class NoopObjectTrace(AbstractContextManager):
    def snapshot(self, *args, **kwargs) -> None:
        pass

    def delete(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


_NOOP_OBJECT_TRACE = cast(ObjectTrace, NoopObjectTrace())


class NoopTraceWriter:
    """A trace writer that does nothing. Used when tracing is disabled."""

    def duration_begin(self, *args, **kwargs) -> None:
        pass

    def duration_end(self, *args, **kwargs) -> None:
        pass

    def span(self, *args, **kwargs) -> "Span":
        return _NOOP_SPAN

    def instant(self, *args, **kwargs) -> None:
        pass

    def async_begin(self, *args, **kwargs) -> None:
        pass

    def async_instant(self, *args, **kwargs) -> None:
        pass

    def async_end(self, *args, **kwargs) -> None:
        pass

    def async_span(self, *args, **kwargs) -> "AsyncSpan":
        return _NOOP_ASYNC_SPAN

    def async_flow(self, *args, **kwargs) -> "AsyncFlow":
        return _NOOP_ASYNC_FLOW

    def flow_start(self, *args, **kwargs) -> None:
        pass

    def flow_step(self, *args, **kwargs) -> None:
        pass

    def flow_end(self, *args, **kwargs) -> None:
        pass

    def counter(self, *args, **kwargs) -> None:
        pass

    def object_new(self, *args, **kwargs) -> None:
        pass

    def object_snapshot(self, *args, **kwargs) -> None:
        pass

    def object_delete(self, *args, **kwargs) -> None:
        pass

    def object_trace(self, *args, **kwargs) -> "ObjectTrace":
        return _NOOP_OBJECT_TRACE

    def trace_object(self, *args, **kwargs) -> "ObjectTrace":
        return _NOOP_OBJECT_TRACE

    def metadata(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def __repr__(self) -> str:
        return "<NoopTraceWriter>"


NOOP_TRACE_WRITER: TraceWriter = cast(TraceWriter, NoopTraceWriter())
