# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Set, Tuple, Type, Union

import pytest

from megatron.energon import typed_converter


class MyNamedTuple(NamedTuple):
    x: int
    y: str


@dataclasses.dataclass
class MyDataClass:
    a: int
    b: str = "default"


class ComplexNamedTuple(NamedTuple):
    id: int
    name: str
    active: bool


@dataclasses.dataclass
class NestedDataClass:
    value: float
    metadata: Dict[str, Any]
    tags: List[str]


@dataclasses.dataclass
class ExtendedNestedDataClass(NestedDataClass):
    extra: str


def my_call(a: int, b: str) -> NestedDataClass:
    return NestedDataClass(value=a, metadata={"b": b}, tags=[b])


def my_call_extended(a: int, b: str) -> ExtendedNestedDataClass:
    return ExtendedNestedDataClass(value=a, metadata={"b": b}, tags=[b], extra=f"extra_{b}")


@dataclasses.dataclass
class ComprehensiveDataClass:
    # Primitive types
    string_field: str
    int_field: int
    float_field: float
    bool_field: bool

    # Optional types
    optional_string: Optional[str] = None
    optional_int: Optional[int] = None

    # Union types
    union_field: Union[str, int] = "default"
    union_optional: Union[str, None] = None

    # List types
    string_list: List[str] = dataclasses.field(default_factory=list)
    int_list: List[int] = dataclasses.field(default_factory=list)
    nested_list: List[List[str]] = dataclasses.field(default_factory=list)

    # Dict types
    string_dict: Dict[str, str] = dataclasses.field(default_factory=dict)
    mixed_dict: Dict[str, Any] = dataclasses.field(default_factory=dict)
    nested_dict: Dict[str, Dict[str, int]] = dataclasses.field(default_factory=dict)

    # Tuple types
    fixed_tuple: Tuple[str, int, bool] = ("default", 0, False)
    variable_tuple: Tuple[str, ...] = ("single",)

    # Set types
    set_field: Set[int] = dataclasses.field(default_factory=set)

    # Literal types
    status: Literal["active", "inactive", "pending"] = "pending"
    priority: Literal[1, 2, 3, 4, 5] = 3

    # Nested dataclass
    nested: Optional[NestedDataClass] = None

    # Referencing a type
    type_ref: Type[NestedDataClass] = NestedDataClass

    # Referencing a function
    function_ref: Callable[[int, str], NestedDataClass] = my_call

    # NamedTuple
    named_tuple: Optional[ComplexNamedTuple] = None

    # Any type
    any_field: Any = None


def test_raw_to_typed_namedtuple():
    parser = typed_converter.JsonParser()
    raw = {"x": 42, "y": "foo"}
    result = parser.raw_to_typed(raw, MyNamedTuple)
    assert isinstance(result, MyNamedTuple)
    assert result.x == 42
    assert result.y == "foo"


def test_raw_to_typed_dataclass():
    parser = typed_converter.JsonParser()
    raw = {"a": 7, "b": "bar"}
    result = parser.raw_to_typed(raw, MyDataClass)
    assert isinstance(result, MyDataClass)
    assert result.a == 7
    assert result.b == "bar"


def test_raw_to_typed_dataclass_default():
    parser = typed_converter.JsonParser()
    raw = {"a": 5}
    result = parser.raw_to_typed(raw, MyDataClass)
    assert result.a == 5
    assert result.b == "default"


def test_raw_to_typed_union():
    parser = typed_converter.JsonParser()
    assert parser.raw_to_typed(123, Union[int, str]) == 123
    assert parser.raw_to_typed("abc", Union[int, str]) == "abc"
    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(1.5, Union[int, str])


def test_raw_to_typed_optional():
    parser = typed_converter.JsonParser()
    assert parser.raw_to_typed(None, Optional[int]) is None
    assert parser.raw_to_typed(10, Optional[int]) == 10


def test_raw_to_typed_list():
    parser = typed_converter.JsonParser()
    raw = [1, 2, 3]
    result = parser.raw_to_typed(raw, List[int])
    assert result == [1, 2, 3]


def test_raw_to_typed_dict():
    parser = typed_converter.JsonParser()
    raw = {"foo": 1, "bar": 2}
    result = parser.raw_to_typed(raw, Dict[str, int])
    assert result == {"foo": 1, "bar": 2}


def test_raw_to_typed_set():
    parser = typed_converter.JsonParser()
    raw = [1, 2, 3]
    result = parser.raw_to_typed(raw, Set[int])
    assert result == {1, 2, 3}


def test_raw_to_typed_literal():
    parser = typed_converter.JsonParser()
    assert parser.raw_to_typed("yes", Literal["yes", "no"]) == "yes"
    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed("maybe", Literal["yes", "no"])


def test_to_json_object_namedtuple():
    obj = MyNamedTuple(x=1, y="abc")
    json_obj = typed_converter.to_json_object(obj)
    assert json_obj == {"x": 1, "y": "abc"}


def test_to_json_object_dataclass():
    obj = MyDataClass(a=2, b="xyz")
    json_obj = typed_converter.to_json_object(obj)
    assert json_obj == {"a": 2, "b": "xyz"}


def test_to_json_object_list():
    obj = [1, 2, 3]
    json_obj = typed_converter.to_json_object(obj)
    assert json_obj == [1, 2, 3]


def test_to_json_object_dict():
    obj = {"foo": 1, "bar": 2}
    json_obj = typed_converter.to_json_object(obj)
    assert json_obj == {"foo": 1, "bar": 2}


def test_isinstance_deep():
    assert typed_converter._isinstance_deep(1, int)
    assert not typed_converter._isinstance_deep(1, str)
    assert not typed_converter._isinstance_deep(1, float)
    assert not typed_converter._isinstance_deep("1", int)
    assert not typed_converter._isinstance_deep("1", float)
    assert typed_converter._isinstance_deep([1, 2], List[int])
    assert not typed_converter._isinstance_deep([1, "a"], List[int])
    assert typed_converter._isinstance_deep({"a": 1}, Dict[str, int])
    assert not typed_converter._isinstance_deep({"a": "b"}, Dict[str, int])


def test_missing_value_error():
    parser = typed_converter.JsonParser()
    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(typed_converter._missing_value, int)


def test_strict_extra_keys():
    parser = typed_converter.JsonParser(strict=True)
    raw = {"a": 1, "b": "foo", "extra": 123}
    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(raw, MyDataClass)


def test_non_strict_extra_keys():
    parser = typed_converter.JsonParser(strict=False)
    raw = {"a": 1, "b": "foo", "extra": 123}
    result = parser.raw_to_typed(raw, MyDataClass)
    assert result.a == 1
    assert result.b == "foo"


def test_comprehensive_dataclass():
    """Test a complex dataclass with all supported types."""
    parser = typed_converter.JsonParser()

    # Create comprehensive raw data
    raw_data = {
        "string_field": "test_string",
        "int_field": 42,
        "float_field": 3.14159,
        "bool_field": True,
        "optional_string": "optional_value",
        "optional_int": 100,
        "union_field": 123,  # Using int instead of string
        "union_optional": "union_string",
        "string_list": ["item1", "item2", "item3"],
        "int_list": [1, 2, 3, 4, 5],
        "nested_list": [["a", "b"], ["c", "d"]],
        "string_dict": {"key1": "value1", "key2": "value2"},
        "mixed_dict": {"str_key": "string", "int_key": 42, "bool_key": True},
        "nested_dict": {"outer1": {"inner1": 1, "inner2": 2}, "outer2": {"inner3": 3}},
        "fixed_tuple": ["tuple_string", 99, True],
        "variable_tuple": ["var1", "var2", "var3"],
        "set_field": [1, 2, 3],
        "status": "active",
        "priority": 5,
        "nested": {
            "value": 2.71828,
            "metadata": {"nested_key": "nested_value", "count": 42},
            "tags": ["tag1", "tag2"],
        },
        "named_tuple": {"id": 123, "name": "test_name", "active": False},
        "any_field": {"arbitrary": "data", "number": 999},
    }

    # Convert raw data to typed object
    result = parser.raw_to_typed(raw_data, ComprehensiveDataClass)

    # Verify all fields
    assert result.string_field == "test_string"
    assert result.int_field == 42
    assert result.float_field == 3.14159
    assert result.bool_field is True
    assert result.optional_string == "optional_value"
    assert result.optional_int == 100
    assert result.union_field == 123
    assert result.union_optional == "union_string"
    assert result.string_list == ["item1", "item2", "item3"]
    assert result.int_list == [1, 2, 3, 4, 5]
    assert result.nested_list == [["a", "b"], ["c", "d"]]
    assert result.string_dict == {"key1": "value1", "key2": "value2"}
    assert result.mixed_dict == {"str_key": "string", "int_key": 42, "bool_key": True}
    assert result.nested_dict == {"outer1": {"inner1": 1, "inner2": 2}, "outer2": {"inner3": 3}}
    assert result.fixed_tuple == ("tuple_string", 99, True)
    assert result.variable_tuple == ("var1", "var2", "var3")
    assert result.set_field == {1, 2, 3}
    assert result.status == "active"
    assert result.priority == 5

    # Verify nested dataclass
    assert isinstance(result.nested, NestedDataClass)
    assert result.nested.value == 2.71828
    assert result.nested.metadata == {"nested_key": "nested_value", "count": 42}
    assert result.nested.tags == ["tag1", "tag2"]

    # Verify NamedTuple
    assert isinstance(result.named_tuple, ComplexNamedTuple)
    assert result.named_tuple.id == 123
    assert result.named_tuple.name == "test_name"
    assert result.named_tuple.active is False

    # Verify Any field
    assert result.any_field == {"arbitrary": "data", "number": 999}

    # Test conversion back to JSON
    json_obj = typed_converter.to_json_object(result)

    # Verify JSON conversion preserves data
    assert json_obj["string_field"] == "test_string"
    assert json_obj["int_field"] == 42
    assert json_obj["float_field"] == 3.14159
    assert json_obj["bool_field"] is True
    assert json_obj["optional_string"] == "optional_value"
    assert json_obj["optional_int"] == 100
    assert json_obj["union_field"] == 123
    assert json_obj["union_optional"] == "union_string"
    assert json_obj["string_list"] == ["item1", "item2", "item3"]
    assert json_obj["int_list"] == [1, 2, 3, 4, 5]
    assert json_obj["nested_list"] == [["a", "b"], ["c", "d"]]
    assert json_obj["string_dict"] == {"key1": "value1", "key2": "value2"}
    assert json_obj["mixed_dict"] == {"str_key": "string", "int_key": 42, "bool_key": True}
    assert json_obj["nested_dict"] == {
        "outer1": {"inner1": 1, "inner2": 2},
        "outer2": {"inner3": 3},
    }
    assert json_obj["fixed_tuple"] == ["tuple_string", 99, True]
    assert json_obj["variable_tuple"] == ["var1", "var2", "var3"]
    assert json_obj["set_field"] == [1, 2, 3]
    assert json_obj["status"] == "active"
    assert json_obj["priority"] == 5
    assert json_obj["nested"]["value"] == 2.71828
    assert json_obj["nested"]["metadata"] == {"nested_key": "nested_value", "count": 42}
    assert json_obj["nested"]["tags"] == ["tag1", "tag2"]
    assert json_obj["named_tuple"]["id"] == 123
    assert json_obj["named_tuple"]["name"] == "test_name"
    assert json_obj["named_tuple"]["active"] is False
    assert json_obj["any_field"] == {"arbitrary": "data", "number": 999}
    assert json_obj["function_ref"]["__module__"] == my_call.__module__
    assert json_obj["function_ref"]["__function__"] == my_call.__name__
    assert json_obj["type_ref"]["__module__"] == NestedDataClass.__module__
    assert json_obj["type_ref"]["__class__"] == NestedDataClass.__name__


def test_comprehensive_dataclass_with_defaults():
    """Test comprehensive dataclass with minimal data using defaults."""
    parser = typed_converter.JsonParser()

    # Minimal raw data - only required fields
    raw_data = {"string_field": "minimal", "int_field": 1, "float_field": 1.0, "bool_field": False}

    result = parser.raw_to_typed(raw_data, ComprehensiveDataClass)

    # Verify required fields
    assert result.string_field == "minimal"
    assert result.int_field == 1
    assert result.float_field == 1.0
    assert result.bool_field is False

    # Verify defaults
    assert result.optional_string is None
    assert result.optional_int is None
    assert result.union_field == "default"
    assert result.union_optional is None
    assert result.string_list == []
    assert result.int_list == []
    assert result.nested_list == []
    assert result.string_dict == {}
    assert result.mixed_dict == {}
    assert result.nested_dict == {}
    assert result.fixed_tuple == ("default", 0, False)
    assert result.variable_tuple == ("single",)
    assert result.set_field == set()
    assert result.status == "pending"
    assert result.priority == 3
    assert result.nested is None
    assert result.named_tuple is None
    assert result.any_field is None


def test_comprehensive_dataclass_error_cases():
    """Test error cases for comprehensive dataclass."""
    parser = typed_converter.JsonParser()

    # Test invalid literal value
    raw_data = {
        "string_field": "test",
        "int_field": 1,
        "float_field": 1.0,
        "bool_field": False,
        "status": "invalid_status",  # Should fail
    }

    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(raw_data, ComprehensiveDataClass)

    # Test invalid union type
    raw_data = {
        "string_field": "test",
        "int_field": 1,
        "float_field": 1.0,
        "bool_field": False,
        "union_field": 1.5,  # Should fail - not str or int
    }

    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(raw_data, ComprehensiveDataClass)

    # Test invalid list element type
    raw_data = {
        "string_field": "test",
        "int_field": 1,
        "float_field": 1.0,
        "bool_field": False,
        "string_list": ["valid", 123, "also_valid"],  # Should fail - int in string list
    }

    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(raw_data, ComprehensiveDataClass)

    # Test invalid tuple length
    raw_data = {
        "string_field": "test",
        "int_field": 1,
        "float_field": 1.0,
        "bool_field": False,
        "fixed_tuple": ["only", "two"],  # Should fail - needs 3 elements
    }

    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(raw_data, ComprehensiveDataClass)


def test_comprehensive_dataclass_strict_mode():
    """Test comprehensive dataclass in strict mode with extra keys."""
    parser = typed_converter.JsonParser(strict=True)

    raw_data = {
        "string_field": "test",
        "int_field": 1,
        "float_field": 1.0,
        "bool_field": False,
        "extra_field": "should_fail",  # Should fail in strict mode
        "nested": {
            "__module__": my_call_extended.__module__,
            "__function__": my_call_extended.__name__,
            "a": 42,
            "b": "Hello",
        },
        "function_ref": {
            "__module__": my_call_extended.__module__,
            "__function__": my_call_extended.__name__,
        },
        "type_ref": {
            "__module__": ExtendedNestedDataClass.__module__,
            "__class__": ExtendedNestedDataClass.__name__,
        },
    }

    with pytest.raises(typed_converter.JsonValueError):
        parser.raw_to_typed(raw_data, ComprehensiveDataClass)

    # Test non-strict mode allows extra keys
    parser_non_strict = typed_converter.JsonParser(strict=False)
    result = parser_non_strict.raw_to_typed(raw_data, ComprehensiveDataClass)
    assert result.string_field == "test"
    assert result.int_field == 1
    assert result.nested == ExtendedNestedDataClass(
        value=42, metadata={"b": "Hello"}, tags=["Hello"], extra="extra_Hello"
    )
    assert result.function_ref == my_call_extended
    assert result.type_ref == ExtendedNestedDataClass
