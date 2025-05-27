from __future__ import annotations

import gc
import json
import os
import threading
import time
import traceback
import weakref
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import (
    IO,
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    Iterable,
    Literal,
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
    "AsyncContext",
    "Flow",
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


def _cur_thread_id() -> int:
    """Return current thread id as int."""
    tid = threading.get_ident()
    while tid > 0xFFFFFFFF:
        tid = (tid & 0xFFFFFFFF) ^ (tid >> 32)
    return tid


class JsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays, torch tensors, and dataclasses."""

    def default(self, o: Any) -> Any:
        # Handle numpy arrays directly
        if isinstance(o, (np.ndarray, torch.Tensor)):
            try:
                return o.tolist()
            except Exception:
                return str(o)[:250]

        # Handle dataclass *instances* (exclude dataclass *types*).
        if is_dataclass(o) and not isinstance(o, type):
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
        if level > self._log_level:
            return
        event = {
            "name": name,
            "ph": "B",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
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
        if level > self._log_level:
            return
        event = {
            "name": name,
            "ph": "E",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
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
        if level > self._log_level:
            return _NOOP_SPAN
        return Span(self, name=name, cat=cat, args=args)

    def iterable(
        self,
        iterable: Iterable[T],
        *,
        name: Optional[str] = None,
        next: Optional[Callable[[], ContextManager]] = None,
        level: int = 0,
    ) -> Iterable[T]:
        """Wrap an iterable to emit trace events for each `next` call."""
        if level > self._log_level:
            return iterable
        assert (name is not None) != (next is not None), "Either name xor next must be provided"
        if name is not None:
            return iterable_wrapper(iterable, span=lambda: self.span(name))
        else:
            assert next is not None
            return iterable_wrapper(iterable, span=next)

    def instant(
        self,
        name: str,
        *,
        cat: str | None = None,
        scope: Optional[Literal["t", "p", "g"]] = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Emit a zero-duration *instant* event (``ph='i'``).

        Args:
            name: Display name.
            cat: Optional categories.
            scope: Trace-viewer scope selector – ``t`` (thread), ``p`` (process)
                or ``g`` (global). Defaults to ``t``.
            args: Optional arguments payload.
            level: Logging level.
        """
        if level > self._log_level:
            return
        event = {
            "name": name,
            "ph": "i",
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if scope is not None:
            event["s"] = scope
        if cat is not None:
            event["cat"] = cat
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def generator(
        self,
        name: str,
        *,
        cat: str | None = None,
        next_args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "GeneratorContext":
        if level > self._log_level:
            return _NOOP_GENERATOR_CONTEXT
        return GeneratorContext(self, name=name, cat=cat, next_args=next_args)

    # Async events --------------------------------------------------------

    def async_begin(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> Union[int, str]:
        """Start a *nestable async* chain (``ph='b'``).

        Args:
            name: Event display name.
            id: Correlation identifier (int or str).
            cat: Optional categories.
            args: Optional argument object.
            level: Logging level.
        """
        if id is None:
            id = self._next_id()
        if level > self._log_level:
            return id

        event = {
            "name": name,
            "ph": "b",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if cat is not None:
            event["cat"] = cat
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
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Emit an *instant* step for a nestable async chain (``ph='n'``).

        Args:
            name: Event name.
            id: Correlation identifier.
            cat: Categories.
            args: Additional arguments.
            level: Logging level.
        """
        if level > self._log_level:
            return
        if id is None:
            id = self._next_id()

        event = {
            "name": name,
            "ph": "n",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if cat is not None:
            event["cat"] = cat
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def async_end(
        self,
        name: str,
        *,
        id: Union[int, str],
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> None:
        """Finish a *nestable async* chain (``ph='e'``).

        Args:
            id: Correlation identifier.
            cat: Categories.
            args: Additional arguments.
            level: Logging level.
        """
        if level > self._log_level:
            return
        event = {
            "ph": "e",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if cat is not None:
            event["cat"] = cat
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def async_span(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "AsyncSpan":
        """Return an *AsyncSpan* context-manager for a nestable async chain.

        Args:
            name: Display name.
            id: Correlation identifier to keep events together.
            cat: Categories.
            args: Arguments attached to the begin event.
            level: Logging level.

        Returns:
            AsyncSpan context manager.
        """
        if level > self._log_level:
            return _NOOP_ASYNC_SPAN
        if id is None:
            id = self._next_id()

        return AsyncSpan(
            self,
            name=name,
            id=id,
            cat=cat,
            args=args,
        )

    def async_flow(
        self,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        level: int = 0,
    ) -> "AsyncContext":
        """Return an *AsyncFlow* context-manager for a nestable async chain.

        Args:
            id: Correlation identifier.
            cat: Categories.
            level: Logging level.
        """
        if level > self._log_level:
            return _NOOP_ASYNC_CONTEXT
        if id is None:
            id = self._next_id()

        return AsyncContext(
            self,
            id=id,
            cat=cat,
        )

    def async_generator(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        next_args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "AsyncGeneratorContext":
        """Emit an async *generator* (``ph='g'``) event within this async flow."""
        if level > self._log_level:
            return _NOOP_ASYNC_GENERATOR_CONTEXT
        if id is None:
            id = self._next_id()
        return AsyncGeneratorContext(
            self,
            name=name,
            id=id,
            cat=cat,
            next_args=next_args,
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
        if level > self._log_level:
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
            "tid": _cur_thread_id(),
            "args": args_field,
        }
        if id is not None:
            event["id"] = id
        if cat is not None:
            event["cat"] = cat
        self._emit(event)

    def async_object_trace(
        self,
        name: str,
        *,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        snapshot: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "ObjectTrace":
        """Create an :class:`ObjectTrace` helper.

        Args:
            name: Object type/name.
            id: Identifier to correlate with future snapshots/deletion.
            cat: Categories.
            snapshot: Optional initial snapshot emitted right after ``N``.
            level: Logging level.

        Returns:
            AsyncObjectTrace instance.
        """
        if level > self._log_level:
            return _NOOP_OBJECT_TRACE
        if id is None:
            id = self._next_id()

        return ObjectTrace(
            self,
            name=name,
            id=id,
            cat=cat,
            initial_snapshot=snapshot,
        )

    def trace_object_async(
        self,
        obj: Any,
        name: str,
        *,
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
        level: int = 0,
    ) -> "ObjectTrace":
        """Attach tracing to an existing Python *obj* until GC.

        Args:
            obj: Target instance to monitor.
            name: Trace-viewer object name.
            cat: Categories.
            level: Logging level.

        Returns:
            AsyncObjectTrace handle.
        """
        if not gc.is_tracked(obj):
            raise ValueError("Object is not tracked by the garbage collector")
        if level > self._log_level:
            return _NOOP_OBJECT_TRACE
        trace = self.async_object_trace(name, id=id(obj), cat=cat, snapshot=args)
        weakref.finalize(obj, trace.delete)
        return trace

    # Metadata ------------------------------------------------------------

    def metadata(
        self: "TraceWriter",
        name: str,
        *,
        args: Dict[str, Any],
        tid: int | None = None,
    ) -> None:
        """Emit a generic *metadata* event (``ph='M'``).

        Args:
            name: Metadata event name (e.g. ``process_name``).
            args: Arguments dict as required by the spec.
            tid: Thread id; required for thread metadata.
        """
        event = {
            "name": name,
            "ph": "M",
            "pid": self._pid,
        }
        if tid is not None:
            event["tid"] = tid
        if args:
            event["args"] = dict(args)
        self._emit(event)

    def metadata_process_name(self, name: str) -> None:
        """Set the current process name."""
        self.metadata("process_name", args=dict(name=name))

    def metadata_thread_name(self, name: str) -> None:
        """Set the current thread name."""
        self.metadata("thread_name", args=dict(name=name), tid=_cur_thread_id())

    # Flow events --------------------------------------------------------

    def flow_start(
        self,
        name: str,
        *,
        id: Optional[int] = None,
        cat: str | None = None,
        level: int = 0,
    ) -> Union[int, str]:
        """Emit a *flow start* (``ph='s'``) event. The flow is bound to the enclosing slice.

        Args:
            name: Display name.
            id: Correlation identifier.
            cat: Categories.
            args: Additional arguments.
            level: Logging level.
        """
        if id is None:
            id = self._next_id()
        if level > self._log_level:
            return id

        event: Dict[str, Any] = {
            "name": name,
            "ph": "s",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if cat is not None:
            event["cat"] = cat
        self._emit(event)
        return id

    def flow_step(
        self,
        name: str,
        *,
        id: int,
        cat: str | None = None,
        level: int = 0,
    ) -> None:
        """
        Emit a *flow step* (``ph='t'``) event. The flow is bound to the enclosing slice.

        Args:
            name: The name of the flow.
            id: The id of the flow.
            cat: The category of the flow.
            level: The level of the flow.
        """
        if level > self._log_level:
            return

        event: Dict[str, Any] = {
            "name": name,
            "ph": "t",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if cat is not None:
            event["cat"] = cat
        self._emit(event)

    def flow_end(
        self,
        name: str,
        *,
        id: int,
        cat: str | None = None,
        bind_enclosing_slice: bool = False,
        level: int = 0,
    ) -> None:
        """Emit a *flow end* (``ph='f'``) event. The flow is finished either in the enclosing slice or at the next slice.

        Args:
            name: The name of the flow.
            id: The id of the flow.
            cat: The category of the flow.
            bind_enclosing_slice: If *True*, adds ``bp='e'`` to bind to the
                enclosing slice (see Trace Event Format), otherwise binds to the next slice.
            level: The level of the flow.
        """
        if level > self._log_level:
            return

        event: Dict[str, Any] = {
            "name": name,
            "ph": "f",
            "id": id,
            "ts": _timestamp_us(),
            "pid": self._pid,
            "tid": _cur_thread_id(),
        }
        if cat is not None:
            event["cat"] = cat
        if bind_enclosing_slice:
            event["bp"] = "e"
        self._emit(event)

    def flow(
        self,
        name: str,
        *,
        id: Optional[int] = None,
        cat: str | None = None,
        level: int = 0,
    ) -> "Flow":
        """Emit a *flow* event."""
        if level > self._log_level:
            return _NOOP_FLOW
        if id is None:
            id = self._next_id()
        return Flow(self, name=name, id=id, cat=cat)

    def resume_flow(self, saved_flow: dict) -> "Flow":
        """Resume a flow from a dictionary."""
        if len(saved_flow) == 0:
            return _NOOP_FLOW
        return Flow(self, **saved_flow, resuming=True)

    # Exception ---------------------------------------------------------

    def async_exc(
        self,
        *,
        name: str,
        id: Union[int, str, None] = None,
        cat: str | None = None,
        level: int = 0,
    ) -> None:
        """Emit an *exception* event as an async instant (``ph='n'``).

        This is primarily used by :class:`AsyncFlow` to surface
        exceptions that happened inside a flow.
        """

        if id is None:
            id = self._next_id()
        if level > self._log_level:
            return

        # Represent exception as string to keep JSON serialisable.
        exc_repr = traceback.format_exc().splitlines()

        self.async_instant(name, id=id, cat=cat, args={"exception": exc_repr}, level=level)

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
    _writer: Optional[TraceWriter]
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
        self._writer.duration_begin(self._name, cat=self._cat, args=args or None)
        self._args = None

    def update_args(self, args: Dict[str, Any]) -> None:
        if self._args is None:
            self._args = args
        else:
            self._args.update(args)

    def end(self, args: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is None:
            return
        if self._args and args:
            self._args.update(args)
        self._writer.duration_end(self._name, cat=self._cat, args=self._args or args or None)
        self._args = None
        self._writer = None

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------

    def __enter__(self):  # noqa: D401
        return self

    def __exit__(self, exc_type, exc, tb):  # noqa: D401, N802
        self.end()


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


class AsyncSpan(AbstractContextManager):
    """Context manager for *nestable async* events.

    Use :py:meth:`instant` for ``n`` events inside the span.
    """

    __slots__ = (
        "_writer",
        "_name",
        "_id",
        "_cat",
        "_args",
    )

    _writer: Optional[TraceWriter]
    _name: str
    _id: Union[int, str]
    _cat: Optional[str]
    _args: Optional[Dict[str, Any]]

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        id: Union[int, str],
        cat: str | None = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._writer = writer
        self._name = name
        self._id = id
        self._cat = cat
        self._args = None

        self._writer.async_begin(self._name, id=self._id, cat=self._cat, args=args or None)

    def update_args(self, args: Dict[str, Any]) -> None:
        if self._args is None:
            self._args = args
        else:
            self._args.update(args)

    def end(self, args: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is None:
            return
        if self._args and args:
            self._args.update(args)
        self._writer.async_end(self._name, id=self._id, cat=self._cat, args=self._args or None)
        self._args = None
        self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.end()


class NoopAsyncSpan(AbstractContextManager):
    def update_args(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


_NOOP_ASYNC_SPAN = cast(AsyncSpan, NoopAsyncSpan())


class AsyncContext:
    """Context manager for *nestable async* events with the same id."""

    __slots__ = (
        "_writer",
        "_id",
        "_cat",
    )

    _writer: TraceWriter
    _id: Union[int, str]
    _cat: Optional[str]

    def __init__(
        self,
        writer: TraceWriter,
        *,
        id: Union[int, str],
        cat: str | None = None,
    ) -> None:
        self._writer = writer
        self._id = id
        self._cat = cat

    def instant(self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0) -> None:
        """Emit an async *instant* (``ph='n'``) event within this async flow."""
        self._writer.async_instant(name, id=self._id, cat=self._cat, args=args, level=level)

    def start(self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0) -> None:
        """Emit an async *start* (``ph='b'``) event within this async flow."""
        self._writer.async_begin(name, id=self._id, cat=self._cat, args=args, level=level)

    def end(self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0) -> None:
        """Emit an async *end* (``ph='e'``) event within this async flow."""
        self._writer.async_end(name, id=self._id, cat=self._cat, args=args, level=level)

    def span(
        self, name: str, *, args: Optional[Dict[str, Any]] = None, level: int = 0
    ) -> AsyncSpan:
        """Emit an async *span* (``ph='s'``) event within this async flow."""
        return self._writer.async_span(name, id=self._id, cat=self._cat, args=args, level=level)

    def generator(
        self, name: str, *, next_args: Optional[Dict[str, Any]] = None, level: int = 0
    ) -> "AsyncGeneratorContext":
        """Get a generator context for the given name.

        This is used to trace all code being executed between yields of a generator.

        Usage::

            with async_ctx.generator(name="my_generator", next_args={"item_idx": 0}) as ctx:
                for item_idx, item in enumerate(iterable):
                    ctx.instant("item", args={"item": item})
                    with ctx.yield_(next_args={"item_idx": item_idx + 1}):
                        yield item
        """
        return self._writer.async_generator(
            name, id=self._id, cat=self._cat, next_args=next_args, level=level
        )

    def iterable(self, iterable: Iterable[T], *, name: str, level: int = 0) -> Iterable[T]:
        """Wrap an iterable to emit trace events for each `next` call."""
        if level > self._writer._log_level:
            return iterable
        return iterable_wrapper(iterable, span=lambda: self.span(name))

    def exc(self, *, name: str, level: int = 0) -> None:
        """Emit an exception event."""
        self._writer.async_exc(name=name, id=self._id, cat=self._cat, level=level)


class NoopAsyncContext:
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

    def exc(self, *args, **kwargs) -> None:
        pass


_NOOP_ASYNC_CONTEXT = cast(AsyncContext, NoopAsyncContext())


class AsyncGeneratorContext(AbstractContextManager):
    """Context manager for a generator context, that interrupts when yielding.

    Use like this::

        with writer.async_generator_context(name="my_generator", next_args={"item_idx": 0}) as ctx:
            for item_idx, item in enumerate(iterable):
                ctx.instant("item", args={"item": item})
                with ctx.yield_(next_args={"item_idx": item_idx + 1}):
                    yield item
    """

    __slots__ = (
        "_writer",
        "_name",
        "_id",
        "_cat",
        "_active_scope",
    )

    _writer: Optional[TraceWriter]
    _name: str
    _id: Union[int, str]
    _cat: Optional[str]

    _active_scope: Optional[AsyncSpan]

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        id: Union[int, str],
        cat: Optional[str] = None,
        next_args: Optional[Dict[str, Any]] = None,
    ):
        self._writer = writer
        self._name = name
        self._id = id
        self._cat = cat

        self._active_scope = self._writer.async_span(name, id=id, cat=cat, args=next_args)

    @contextmanager
    def yield_(
        self,
        *,
        last_args: Optional[Dict[str, Any]] = None,
        next_args: Optional[Dict[str, Any]] = None,
    ):
        if self._writer is None:
            return
        assert self._active_scope is not None
        self._active_scope.end(args=last_args)
        self._active_scope = None
        try:
            yield self
        finally:
            assert self._active_scope is None
            self._active_scope = self._writer.async_span(
                self._name, id=self._id, cat=self._cat, args=next_args
            )

    def yield_from(
        self,
        iterable: Iterable[T],
        *,
        last_args: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[T]:
        """Wrap an iterable to emit trace events for each `next` call."""
        for item in iterable:
            with self.yield_(last_args=last_args, next_args=args):
                last_args = None
                yield item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._active_scope is not None
        self._active_scope.end()
        self._active_scope = None
        self._writer = None


class DummyAsyncGeneratorContext(AbstractContextManager):
    @contextmanager
    def yield_(self, *args, **kwargs):
        yield self

    def yield_from(self, iterable: Iterable[T], **kwargs) -> Iterable[T]:
        return iterable

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


_NOOP_ASYNC_GENERATOR_CONTEXT = cast(AsyncGeneratorContext, DummyAsyncGeneratorContext())


class GeneratorContext(AbstractContextManager):
    """Context manager for a generator context, that interrupts when yielding.

    Use like this::

        with writer.generator_context(name="my_generator", next_args={"item_idx": 0}) as ctx:
            for item_idx, item in enumerate(iterable):
                with ctx.yield_(next_args={"item_idx": item_idx + 1}):
                    yield item
    """

    __slots__ = (
        "_writer",
        "_name",
        "_cat",
        "_active_scope",
    )

    _writer: Optional[TraceWriter]
    _name: str
    _cat: Optional[str]

    _active_scope: Optional[Span]

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        cat: Optional[str] = None,
        next_args: Optional[Dict[str, Any]] = None,
    ):
        self._writer = writer
        self._name = name
        self._cat = cat

        self._active_scope = self._writer.span(name, cat=cat, args=next_args)

    @contextmanager
    def yield_(
        self,
        *,
        last_args: Optional[Dict[str, Any]] = None,
        next_args: Optional[Dict[str, Any]] = None,
    ):
        if self._writer is None:
            return
        assert self._active_scope is not None
        self._active_scope.end(args=last_args)
        self._active_scope = None
        try:
            yield self
        finally:
            assert self._active_scope is None
            self._active_scope = self._writer.span(self._name, cat=self._cat, args=next_args)

    def yield_from(
        self,
        iterable: Iterable[T],
        *,
        last_args: Optional[Dict[str, Any]] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> Iterable[T]:
        """Wrap an iterable to emit trace events for each `next` call."""
        for item in iterable:
            with self.yield_(last_args=last_args, next_args=args):
                last_args = None
                yield item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        assert self._active_scope is not None
        self._active_scope.end()
        self._active_scope = None
        self._writer = None


class DummyGeneratorContext(AbstractContextManager):
    @contextmanager
    def yield_(self, *args, **kwargs):
        yield self

    def yield_from(self, iterable: Iterable[T], **kwargs) -> Iterable[T]:
        return iterable

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


_NOOP_GENERATOR_CONTEXT = cast(GeneratorContext, DummyGeneratorContext())


def iterable_wrapper(iterable: Iterable[T], *, span: Callable[[], ContextManager]) -> Iterable[T]:
    """A wrapper for an iterable that emits trace events for each `next` call."""
    ctx = span()
    ctx.__enter__()
    try:
        for item in iterable:
            ctx.__exit__(None, None, None)
            yield item
            ctx = span()
            ctx.__enter__()
    finally:
        ctx.__exit__(None, None, None)


class ObjectTrace:
    """Lifecycle helper for Trace-Event objects, using async events to trace the object.

    Emits ``N`` on construction, :py:meth:`snapshot` for ``O`` and ``D`` upon
    deletion, context exit, or garbage collection.
    """

    __slots__ = (
        "_writer",
        "_name",
        "_id",
        "_cat",
    )

    _writer: Optional[TraceWriter]
    _name: str
    _id: Union[int, str]
    _cat: Optional[str]

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        id: Union[int, str],
        cat: str | None = None,
        initial_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._writer = writer
        self._name = name
        self._id = id
        self._cat = cat

        # Emit object creation event
        self._writer.async_begin(name, id=id, cat=cat, args=initial_snapshot)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def snapshot(self, data: Dict[str, Any], *, level: int = 0) -> None:
        """Emit snapshot for current state of the object."""
        if self._writer is None:
            raise RuntimeError("Cannot snapshot deleted traced object")
        self._writer.async_instant(
            self._name,
            id=self._id,
            args=data,
            cat=self._cat,
            level=level,
        )

    def delete(self) -> None:
        """Emit delete event if not already emitted."""
        if self._writer is None:
            return
        self._writer.async_end(
            self._name,
            id=self._id,
            cat=self._cat,
        )
        self._writer = None


class NoopObjectTrace:
    def snapshot(self, *args, **kwargs) -> None:
        pass

    def delete(self, *args, **kwargs) -> None:
        pass


_NOOP_OBJECT_TRACE = cast(ObjectTrace, NoopObjectTrace())


class NoopTraceWriter:
    """A trace writer that does nothing. Used when tracing is disabled."""

    def close(self) -> None:
        pass

    def flush(self) -> None:
        pass

    def duration_begin(self, *args, **kwargs) -> None:
        pass

    def duration_end(self, *args, **kwargs) -> None:
        pass

    def span(self, *args, **kwargs) -> "Span":
        return _NOOP_SPAN

    def instant(self, *args, **kwargs) -> None:
        pass

    def iterable(self, iterable: Iterable[T], *args, **kwargs) -> Iterable[T]:
        return iterable

    def generator(self, *args, **kwargs) -> "GeneratorContext":
        return _NOOP_GENERATOR_CONTEXT

    def async_begin(self, *args, **kwargs) -> None:
        pass

    def async_instant(self, *args, **kwargs) -> None:
        pass

    def async_end(self, *args, **kwargs) -> None:
        pass

    def async_span(self, *args, **kwargs) -> "AsyncSpan":
        return _NOOP_ASYNC_SPAN

    def async_flow(self, *args, **kwargs) -> "AsyncContext":
        return _NOOP_ASYNC_CONTEXT

    def async_generator(self, *args, **kwargs) -> "AsyncGeneratorContext":
        return _NOOP_ASYNC_GENERATOR_CONTEXT

    def flow_start(self, *args, **kwargs) -> None:
        pass

    def flow_step(self, *args, **kwargs) -> None:
        pass

    def flow_end(self, *args, **kwargs) -> None:
        pass

    def flow(self, *args, **kwargs) -> "Flow":
        return _NOOP_FLOW

    def resume_flow(self, saved_flow: dict) -> "Flow":
        return _NOOP_FLOW

    def counter(self, *args, **kwargs) -> None:
        pass

    def async_object_trace(self, *args, **kwargs) -> "ObjectTrace":
        return _NOOP_OBJECT_TRACE

    def trace_object_async(self, *args, **kwargs) -> "ObjectTrace":
        return _NOOP_OBJECT_TRACE

    def metadata(self, *args, **kwargs) -> None:
        pass

    def metadata_process_name(self, name: str) -> None:
        pass

    def metadata_thread_name(self, name: str) -> None:
        pass

    def async_exc(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def __repr__(self) -> str:
        return "<NoopTraceWriter>"


NOOP_TRACE_WRITER: TraceWriter = cast(TraceWriter, NoopTraceWriter())


# ------------------------------------------------------------------
# Flow context manager
# ------------------------------------------------------------------


class Flow:
    """Context manager for *flow* events (``ph='s'``/``'t'``/``'f'``).

    Use :py:meth:`step` for intermediate *t* events inside the flow.
    """

    __slots__ = (
        "_writer",
        "_name",
        "_id",
        "_cat",
    )

    _writer: Optional[TraceWriter]
    _name: str
    _id: Union[int, str]
    _cat: Optional[str]

    def __init__(
        self,
        writer: TraceWriter,
        *,
        name: str,
        id: int,
        cat: str | None = None,
        resuming: bool = False,
    ) -> None:
        self._writer = writer
        self._name = name
        self._id = id
        self._cat = cat

        # Emit flow *start* event.
        if not resuming:
            self._writer.flow_start(
                self._name,
                id=self._id,
                cat=self._cat,
            )

    def step(self, *, level: int = 0) -> None:
        """Emit a *flow step* (``ph='t'``) event. The flow is bound to the enclosing slice."""
        writer = self._writer
        if writer is None:
            return
        writer.flow_step(
            self._name,
            id=self._id,
            cat=self._cat,
            level=level,
        )

    def end(
        self,
        *,
        level: int = 0,
        bind_enclosing_slice: bool = False,
    ) -> None:
        """
        Emit the *flow end* (``ph='f'``) event. The flow is finished either in the enclosing slice or in the next slice.

        Args:
            name: The name of the flow.
            level: The level of the flow.
            bind_enclosing_slice: Whether to bind the flow to the enclosing slice (otherwise bind to the next slice).
        """
        writer = self._writer
        if writer is None:
            return

        writer.flow_end(
            self._name,
            id=self._id,
            cat=self._cat,
            bind_enclosing_slice=bind_enclosing_slice,
            level=level,
        )
        # Mark as closed to avoid further emissions.
        self._writer = None

    def save(self) -> dict:
        """Return a dictionary representation of the flow, allowing resuming the flow in another process."""
        return {
            "name": self._name,
            "id": self._id,
            "cat": self._cat,
        }


class NoopFlow:
    def step(self, *args, **kwargs) -> None:
        pass

    def end(self, *args, **kwargs) -> None:
        pass

    def save(self) -> dict:
        return {}


_NOOP_FLOW = cast(Flow, NoopFlow())
