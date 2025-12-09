# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
"""Provides methods for converting typed objects to json objects and vice versa."""

import dataclasses
import importlib
import inspect
import typing
from typing import Any, Callable, Literal, Optional, Tuple, Type, TypeVar, Union

from megatron.energon.edataclass import edataclass
from megatron.energon.epathlib import EPath


class JsonValueError(ValueError):
    """Internal error class, for better error messages."""

    def __init__(
        self, msg: str, expected_type: Type, value: Any, path: str, stage: Tuple[int, ...]
    ) -> None:
        """New error"""
        super().__init__(msg)
        self.expected_type = expected_type
        self.value = value
        self.path = path
        self.stage = stage


class _MissingValue:
    """Internal class representing a missing value (i.e. a field which has not been specified,
    opposed to explicit `None`) for better error messages."""

    def __repr__(self) -> str:
        return "missing value"


_missing_value = _MissingValue()

TType = TypeVar("TType")


def _check_instance_type(cls, inst_type: Type) -> bool:
    """Check if a class is an instance of a type."""
    if inst_type is None:
        return True
    elif typing.get_origin(inst_type) is not None:
        org = typing.get_origin(inst_type)
        if org is Union:
            for check_type in typing.get_args(inst_type):
                if _check_instance_type(cls, check_type):
                    return True
        elif isinstance(org, type) and issubclass(cls, org):
            return True
        return False
    elif inst_type is Any:
        return True
    else:
        return not issubclass(inst_type, type) or issubclass(cls, inst_type)


@edataclass
class JsonParser:
    strict: bool = True
    #: If true, parse '__module__' and '__class__/__function__' attributes to allow explicit instantiation of types
    allow_imports: bool = True

    def _resolve_object(
        self,
        module_name: str,
        object_name: str,
        cls: Type[TType],
        is_type: bool,
        is_callable: bool,
        is_instantiating_class: bool,
        is_calling_function: bool,
    ) -> TType:
        """Resolve an object reference to the object."""
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:  # pragma: no cover
            try_energon_module = importlib.import_module("megatron.energon", package=None)
            if hasattr(try_energon_module, object_name):
                module = try_energon_module
            else:
                raise
        try:
            return getattr(module, object_name)
        except AttributeError:  # pragma: no cover
            raise ModuleNotFoundError(f"Object {object_name} not found in {module_name}")

    def raw_to_instance(
        self,
        kwargs: dict,
        inst_type: Type[TType],
        _path: str = "root",
        _stage: Tuple[int, ...] = (),
    ) -> TType:
        """
        Try to import and instantiate a class from a dict with "__module__" and "__class__"/"__function__" keys.

        Args:
            kwargs: The dict to parse
            inst_type: Expected return type, used if type is not specified in the kwargs
            strict: If true, don't allow additional attributes
            _path: (internal for recursive call) The path to the object being converted from the root
            _stage: (internal for recursive call) Numbers representing the position of the current
                object being converted from the root

        Returns:
            Instantiated class
        """
        kwargs = kwargs.copy()
        module_name = kwargs.pop("__module__", None)
        # Check if this is a type of Type[...] or just a class. Type[...] will return the class instead
        # of instantiating it.
        is_type = typing.get_origin(inst_type) is type
        is_callable = typing.get_origin(inst_type) is typing.get_origin(Callable)
        is_calling_function = False
        is_instantiating_class = False
        if is_type:
            inst_type = typing.get_args(inst_type)[0]
            object_name = kwargs.pop("__class__", None)
            if module_name is None or object_name is None:
                raise JsonValueError(
                    f"Expected __module__ and __class__ for Type[{inst_type}], got {kwargs}",
                    inst_type,
                    (module_name, object_name),
                    _path,
                    _stage,
                )
        elif is_callable:
            object_name = kwargs.pop("__function__", None)
            if module_name is None or object_name is None:
                raise JsonValueError(
                    f"Expected __module__ and __function__ for {inst_type}, got {kwargs}",
                    inst_type,
                    (module_name, object_name),
                    _path,
                    _stage,
                )
        else:
            if "__class__" in kwargs:
                object_name = kwargs.pop("__class__", None)
                is_instantiating_class = True
                is_calling_function = False
            elif "__function__" in kwargs:
                object_name = kwargs.pop("__function__", None)
                is_instantiating_class = False
                is_calling_function = True
            # Else case: It's a plain type, and nothing was passed, use the default cls
        if module_name is None or object_name is None:
            cls = inst_type
        else:
            cls = self._resolve_object(
                module_name,
                object_name,
                inst_type,
                is_type,
                is_callable,
                is_instantiating_class,
                is_calling_function,
            )

            if is_type:
                if isinstance(inst_type, type) and (
                    not isinstance(cls, type) or not issubclass(cls, inst_type)
                ):
                    raise JsonValueError(
                        f"Expected Type[{inst_type}], got {cls}", inst_type, cls, _path, _stage
                    )
            elif is_callable:
                if not callable(cls):
                    raise JsonValueError(
                        f"Expected a callable, got {cls}", inst_type, cls, _path, _stage
                    )
            elif is_instantiating_class:
                if not isinstance(cls, type) or not _check_instance_type(cls, inst_type):
                    raise JsonValueError(
                        f"Expected {inst_type}, got {cls}", inst_type, cls, _path, _stage
                    )
            else:
                assert is_calling_function
                if not callable(cls):
                    raise JsonValueError(
                        f"Expected {inst_type}, got {cls}", inst_type, cls, _path, _stage
                    )
        if is_type or is_callable:
            inst = cls
        else:
            # Do not assert the other cases, we fallback to the passed cls
            inst = self.safe_call_function(kwargs, cls)
            assert not isinstance(cls, type) or _check_instance_type(type(inst), inst_type), (
                f"Expected {inst_type}, got {cls}"
            )
        return inst

    def raw_to_typed(  # noqa: C901
        self,
        raw_data: Union[dict, list, str, int, bool, float, None],
        inst_type: Type[TType],
        _path: str = "root",
        _stage: Tuple[int, ...] = (),
    ) -> TType:
        """
        Converts raw data (i.e. dicts, lists and primitives) to typed objects (like
        `NamedTuple` or `dataclasses.dataclass`). Validates that python typing matches.

        Usage::

            class MyNamedTuple(NamedTuple):
                x: int
                y: str

            assert raw_to_typed({'x': int, 'y': "foo"}, MyNamedTuple) == MyNamedTuple(x=5, y="foo")

        Args:
            raw_data: The raw (e.g. json) data to be made as `inst_type`
            inst_type: The type to return
            _path: (internal for recursive call) The path to the object being converted from the root
            _stage: (internal for recursive call) Numbers representing the position of the current
                object being converted from the root

        Returns:
            The input data as `inst_type`.
        """
        type_name = getattr(inst_type, "__name__", repr(inst_type))
        if raw_data is _missing_value:  # pragma: no cover
            raise JsonValueError(
                f"Missing value at {_path}",
                inst_type,
                raw_data,
                _path,
                _stage,
            )
        elif inst_type in (str, int, float, bool, None, type(None)):
            # Literal types or missing data
            if not isinstance(raw_data, inst_type) and not (
                isinstance(raw_data, int) and inst_type is float
            ):  # pragma: no cover
                raise JsonValueError(
                    f"Type does not match, expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return raw_data
        elif inst_type is Any:
            if (
                self.allow_imports
                and isinstance(raw_data, dict)
                and "__module__" in raw_data
                and ("__class__" in raw_data or "__function__" in raw_data)
            ):
                return self.raw_to_instance(raw_data, inst_type, _path=_path, _stage=_stage)
            # Any
            return raw_data
        elif typing.get_origin(inst_type) is Literal:
            # Literal[value[, ...]]
            values = typing.get_args(inst_type)
            if raw_data not in values:  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return raw_data
        elif typing.get_origin(inst_type) is Union:
            # Union[union_types[0], union_types[1], ...]
            union_types = typing.get_args(inst_type)
            if None in union_types:
                # Fast Optional path
                if raw_data is None:
                    return None
            best_inner_error: Optional[JsonValueError] = None
            inner_exceptions = []
            for subtype in union_types:
                try:
                    return self.raw_to_typed(
                        raw_data,
                        subtype,
                        f"{_path} -> {getattr(subtype, '__name__', repr(subtype))}",
                        _stage + (1,),
                    )
                except JsonValueError as err:
                    if best_inner_error is None or len(err.stage) > len(best_inner_error.stage):
                        best_inner_error = err
                        inner_exceptions.clear()
                        inner_exceptions.append(err)
                    elif len(err.stage) == len(best_inner_error.stage):
                        inner_exceptions.append(err)
                    continue
            if len(inner_exceptions) > 0:
                cur_exc = inner_exceptions[0]
                for next_exc in inner_exceptions[1:]:
                    try:
                        raise next_exc from cur_exc
                    except JsonValueError as e:
                        cur_exc = e
                raise cur_exc
            else:  # pragma: no cover
                raise JsonValueError(
                    f"Expected {inst_type} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
        elif (
            self.allow_imports
            and isinstance(raw_data, dict)
            and "__module__" in raw_data
            and ("__class__" in raw_data or "__function__" in raw_data)
        ):
            return self.raw_to_instance(raw_data, inst_type, _path=_path, _stage=_stage)
        elif (
            isinstance(inst_type, type)
            and issubclass(inst_type, tuple)
            and hasattr(inst_type, "__annotations__")
        ):
            # class MyClass(NamedTuple): ...
            if not isinstance(raw_data, dict):
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            if getattr(inst_type, "__dash_keys__", "False"):
                raw_data = {key.replace("-", "_"): val for key, val in raw_data.items()}
            defaults = getattr(inst_type, "_field_defaults", {})
            kwargs = {
                field_name: self.raw_to_typed(
                    raw_data.get(field_name, defaults.get(field_name, _missing_value)),
                    field_type,
                    f"{_path} -> {type_name}:{field_name}",
                    _stage + (idx,),
                )
                for idx, (field_name, field_type) in enumerate(inst_type.__annotations__.items())
            }
            if self.strict and not set(raw_data).issubset(inst_type.__annotations__):
                raise JsonValueError(
                    f"Additional attributes for {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            try:
                return inst_type(**kwargs)
            except BaseException:
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
        elif dataclasses.is_dataclass(inst_type):
            # dataclass
            if not isinstance(raw_data, dict):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )

            def get_field_value(field: dataclasses.Field, idx: int) -> Any:
                value = raw_data.get(field.name, _missing_value)
                if value is _missing_value:
                    # Use the factory value directly, without going through the conversion
                    if field.default_factory is not dataclasses.MISSING:
                        return field.default_factory()
                    elif field.default is not dataclasses.MISSING:
                        return field.default
                return self.raw_to_typed(
                    value,
                    field.type,
                    f"{_path} -> {type_name}:{field.name}",
                    _stage + (idx,),
                )

            kwargs = {
                field.name: get_field_value(field, idx)
                for idx, field in enumerate(dataclasses.fields(inst_type))
                if field.init
            }
            if self.strict and not set(raw_data).issubset(
                field.name for field in dataclasses.fields(inst_type) if field.init
            ):
                raise JsonValueError(
                    f"Additional attributes for {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            try:
                return inst_type(**kwargs)
            except BaseException:  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
        elif typing.get_origin(inst_type) is list:
            # List[inner_type]
            (inner_type,) = typing.get_args(inst_type)
            if not isinstance(raw_data, list):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return [
                self.raw_to_typed(val, inner_type, f"{_path} -> {idx}", _stage + (idx,))
                for idx, val in enumerate(raw_data)
            ]
        elif typing.get_origin(inst_type) is set:
            # Set[inner_type]
            (inner_type,) = typing.get_args(inst_type)
            if not isinstance(raw_data, list):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            res = set(
                self.raw_to_typed(val, inner_type, f"{_path} -> {idx}", _stage + (idx,))
                for idx, val in enumerate(raw_data)
            )
            if len(res) != len(raw_data):  # pragma: no cover
                raise JsonValueError(
                    f"Duplicate element at {_path}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return res
        elif typing.get_origin(inst_type) is tuple:
            # Tuple[inner_types[0], inner_types[1], ...] or Tuple[inner_types[0], Ellipsis/...]
            inner_types = typing.get_args(inst_type)
            if not isinstance(raw_data, (list, tuple)):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            if len(inner_types) == 2 and inner_types[1] is Ellipsis:
                # Tuple of arbitrary length, all elements same type
                # Tuple[inner_types[0], Ellipsis/...]
                return tuple(
                    self.raw_to_typed(val, inner_types[0], f"{_path} -> {idx}", _stage + (idx,))
                    for idx, val in enumerate(raw_data)
                )
            else:
                # Fixed size/typed tuple
                # Tuple[inner_types[0], inner_types[1], ...]
                if len(raw_data) != len(inner_types):  # pragma: no cover
                    raise JsonValueError(
                        f"Expected {type_name} at {_path}, got {raw_data!r}",
                        inst_type,
                        raw_data,
                        _path,
                        _stage,
                    )
                return tuple(
                    self.raw_to_typed(val, inner_type, f"{_path} -> {idx}", _stage + (idx,))
                    for idx, (val, inner_type) in enumerate(zip(raw_data, inner_types))
                )
        elif typing.get_origin(inst_type) is dict:
            # Dict[str, value_type]
            key_type, value_type = typing.get_args(inst_type)
            assert key_type is str
            if not isinstance(raw_data, dict):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return {
                key: self.raw_to_typed(val, value_type, f"{_path} -> {key!r}", _stage + (idx,))
                for idx, (key, val) in enumerate(raw_data.items())
            }
        elif inst_type in (dict, list):
            # dict, list (no subtyping)
            if not isinstance(raw_data, inst_type):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return raw_data
        elif inst_type is EPath:
            if isinstance(raw_data, str):
                return EPath(raw_data)
            elif not isinstance(raw_data, EPath):  # pragma: no cover
                raise JsonValueError(
                    f"Expected {type_name} at {_path}, got {raw_data!r}",
                    inst_type,
                    raw_data,
                    _path,
                    _stage,
                )
            return raw_data
        else:
            return raw_data

    def safe_call_function(
        self,
        raw_data: Union[dict, list, str, int, bool, float, None],
        fn: Callable[..., TType],
    ) -> TType:
        """
        Converts raw data (i.e. dicts, lists and primitives) to typed call arguments.
        Validates that python typing matches.

        Usage::

            def fn(arg1: float, arg2: MyType, arg3) -> Any:
                assert isinstance(arg1, float)
                assert isinstance(arg2, MyType)

            fn(3.141, MyType(), None)

        Args:
            raw_data: The raw (e.g. json) data to be made as `inst_type`
            fn: The function to call with the converted data
            strict: If true, don't allow additional attributes

        Returns:
            The return value of `fn`
        """
        parameters = list(inspect.signature(fn).parameters.items())
        if inspect.isclass(fn):
            init_sig = getattr(fn, "__init__", None)
            if init_sig is not None:
                parameters = list(inspect.signature(init_sig).parameters.items())[1:]
        args = []
        kwargs = {}
        if isinstance(raw_data, dict):
            unused_args = raw_data.copy()
            for idx, (key, param) in enumerate(parameters):
                t = Any if param.annotation is inspect.Parameter.empty else param.annotation
                if param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    if param.default is inspect.Parameter.empty and key not in unused_args:
                        raise ValueError(f"Missing required argument {key!r} for {fn}")
                    kwargs[key] = self.raw_to_typed(
                        unused_args.pop(key, param.default),
                        t,
                        _path=key,
                        _stage=(idx,),
                    )
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    for arg_key, arg_val in unused_args.items():
                        kwargs[arg_key] = self.raw_to_typed(arg_val, t, _path=key, _stage=(idx,))
                    unused_args.clear()
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # No way to pass positional arguments
                    pass
                elif param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    # No way to pass positional arguments
                    raise RuntimeError(f"Unsupported positional only argument {key!r}")
                else:
                    raise RuntimeError(f"Unknown parameter kind {param.kind!r}")
            if self.strict and len(unused_args) > 0:
                raise ValueError(f"Unexpected arguments: {unused_args!r}")
        elif isinstance(raw_data, list):  # pragma: no cover
            unused_args = raw_data.copy()
            for idx, (key, param) in enumerate(parameters):
                t = Any if param.annotation is inspect.Parameter.empty else param.annotation
                if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    if param.default is inspect.Parameter.empty and len(unused_args) == 0:
                        raise ValueError(
                            f"Missing required positional-only argument {key!r} at index {idx}"
                        )
                    args.append(self.raw_to_typed(unused_args.pop(), t, _path=key, _stage=(idx,)))
                elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    if param.default is inspect.Parameter.empty and len(unused_args) == 0:
                        raise ValueError(
                            f"Missing required positional argument {key!r} at index {idx}"
                        )
                    if len(unused_args) == 0:
                        arg_val = param.default
                    else:
                        arg_val = unused_args.pop()
                    args.append(self.raw_to_typed(arg_val, t, _path=key, _stage=(idx,)))
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    for arg_val in unused_args:
                        args.append(self.raw_to_typed(arg_val, t, _path=key, _stage=(idx,)))
                    unused_args.clear()
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    # No way to pass keyword arguments
                    pass
                elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                    raise RuntimeError(f"Unsupported keyword-only argument {key!r}")
                else:
                    raise RuntimeError(f"Unknown parameter kind {param.kind!r}")
            if self.strict and len(unused_args) > 0:
                raise ValueError(f"Unexpected arguments: {unused_args!r}")
        else:  # pragma: no cover
            raise ValueError(
                f"Cannot call function with raw data of type {type(raw_data)!r}, require list or dict"
            )
        return fn(*args, **kwargs)


def to_json_object(obj: Any) -> Any:
    """
    Converts the given object to a json object.

    Args:
        obj: The object to convert

    Returns:
        The json-like object.
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        # Literal types
        return obj
    elif isinstance(obj, tuple) and hasattr(obj, "__annotations__"):
        # class MyClass(NamedTuple): ...
        return {
            field_name: to_json_object(getattr(obj, field_name))
            for field_name in obj.__annotations__.keys()
        }
    elif isinstance(obj, type):
        return {
            "__module__": obj.__module__,
            "__class__": obj.__name__,
        }
    elif isinstance(obj, Callable):
        return {
            "__module__": obj.__module__,
            "__function__": obj.__name__,
        }
    elif dataclasses.is_dataclass(obj):
        # dataclass
        return {
            field.name: to_json_object(getattr(obj, field.name))
            for field in dataclasses.fields(obj)
            if field.init
        }
    elif isinstance(obj, (list, tuple, set)):
        return [to_json_object(val) for val in obj]
    elif isinstance(obj, dict):
        return {key: to_json_object(val) for key, val in obj.items()}
    else:
        raise RuntimeError(f"Unknown type {type(obj)}")


def _isinstance_deep(val: Any, tp_chk: Type) -> bool:
    """Verifies if the given value is an instance of the tp_chk, allowing for typing extensions."""
    if tp_chk is Any:
        return True
    elif typing.get_origin(tp_chk) is Literal:
        values = typing.get_args(tp_chk)
        return val in values
    elif typing.get_origin(tp_chk) is list:
        (inner_type,) = typing.get_args(tp_chk)
        return isinstance(val, list) and all(_isinstance_deep(v, inner_type) for v in val)
    elif typing.get_origin(tp_chk) is tuple:
        inner_types = typing.get_args(tp_chk)
        if len(inner_types) == 2 and inner_types[1] == Ellipsis:
            return isinstance(val, tuple) and all(_isinstance_deep(v, inner_types[0]) for v in val)
        else:
            return (
                isinstance(val, tuple)
                and len(val) == len(inner_types)
                and all(_isinstance_deep(v, inner_type) for v, inner_type in zip(val, inner_types))
            )
    elif typing.get_origin(tp_chk) is dict:
        key_type, value_type = typing.get_args(tp_chk)
        return isinstance(val, dict) and all(
            _isinstance_deep(k, key_type) and _isinstance_deep(v, value_type)
            for k, v in val.items()
        )
    else:
        return isinstance(val, tp_chk)
