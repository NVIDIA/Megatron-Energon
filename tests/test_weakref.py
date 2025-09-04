# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause
import multiprocessing
from dataclasses import dataclass

from megatron.energon.fork_hook import (
    DataclassForkMixin,
    ForkMixin,
    WeakCallbacks,
    after_in_child_fork_hook,
    after_in_parent_fork_hook,
    before_fork_hook,
)


def test_weakcallbacks():
    # Just test the internal WeakCallbacks class.
    a_called = 0
    fn_called = 0

    class A:
        def method(self):
            nonlocal a_called
            a_called += 1

    def fn():
        nonlocal fn_called
        fn_called += 1

    a = A()

    registry = WeakCallbacks()

    registry.add_hook(a.method)
    registry.add_hook(fn)
    registry.add_hook(a.method)

    registry.run()

    assert a_called == 1, a_called
    assert fn_called == 1, fn_called

    assert len(registry._hooks) == 2, len(registry._hooks)

    del a

    assert len(registry._hooks) == 1, len(registry._hooks)

    registry.run()

    assert a_called == 1, a_called
    assert fn_called == 2, fn_called

    del fn

    assert len(registry._hooks) == 0, len(registry._hooks)

    registry.run()

    assert a_called == 1, a_called
    assert fn_called == 2, fn_called

    assert len(registry._hooks) == 0, len(registry._hooks)


def test_fork_weakref():
    # Verify that the fork hooks are called correctly, and that gc works correctly.

    _a_before_fork_called = 0
    _a_after_in_child_fork_called = 0
    _a_after_in_parent_fork_called = 0

    class A(ForkMixin):
        def __before_fork__(self):
            nonlocal _a_before_fork_called
            _a_before_fork_called += 1

        def __after_in_child_fork__(self):
            nonlocal _a_after_in_child_fork_called
            _a_after_in_child_fork_called += 1

        def __after_in_parent_fork__(self):
            nonlocal _a_after_in_parent_fork_called
            _a_after_in_parent_fork_called += 1

    _b_before_fork_called = 0
    _b_after_in_child_fork_called = 0
    _b_after_in_parent_fork_called = 0

    @dataclass
    class B(DataclassForkMixin):
        def __before_fork__(self):
            nonlocal _b_before_fork_called
            _b_before_fork_called += 1

        def __after_in_child_fork__(self):
            nonlocal _b_after_in_child_fork_called
            _b_after_in_child_fork_called += 1

        def __after_in_parent_fork__(self):
            nonlocal _b_after_in_parent_fork_called
            _b_after_in_parent_fork_called += 1

    a = A()
    b = B()

    _before_fork_called = 0
    _after_in_child_fork_called = 0
    _after_in_parent_fork_called = 0

    def before_fork():
        nonlocal _before_fork_called
        _before_fork_called += 1

    def after_in_child_fork():
        nonlocal _after_in_child_fork_called
        _after_in_child_fork_called += 1

    def after_in_parent_fork():
        nonlocal _after_in_parent_fork_called
        _after_in_parent_fork_called += 1

    before_fork_hook(before_fork)
    after_in_child_fork_hook(after_in_child_fork)
    after_in_parent_fork_hook(after_in_parent_fork)

    multiprocessing.set_start_method("fork", force=True)

    def process_verify_fork_hooks_1():
        # Verify in the process that the fork hooks were called
        assert _before_fork_called == 1, _before_fork_called
        assert _after_in_child_fork_called == 1, _after_in_child_fork_called
        # This was not called in the child process
        assert _after_in_parent_fork_called == 0, _after_in_parent_fork_called

        assert _a_before_fork_called == 1, _a_before_fork_called
        assert _a_after_in_child_fork_called == 1, _a_after_in_child_fork_called
        assert _a_after_in_parent_fork_called == 0, _a_after_in_parent_fork_called

        assert _b_before_fork_called == 1, _b_before_fork_called
        assert _b_after_in_child_fork_called == 1, _b_after_in_child_fork_called
        assert _b_after_in_parent_fork_called == 0, _b_after_in_parent_fork_called

    p1 = multiprocessing.Process(target=process_verify_fork_hooks_1)
    p1.start()
    p1.join()
    assert p1.exitcode == 0, p1.exitcode

    assert _before_fork_called == 1, _before_fork_called
    assert _after_in_child_fork_called == 0, _after_in_child_fork_called
    assert _after_in_parent_fork_called == 1, _after_in_parent_fork_called

    assert _a_before_fork_called == 1, _a_before_fork_called
    assert _a_after_in_child_fork_called == 0, _a_after_in_child_fork_called
    assert _a_after_in_parent_fork_called == 1, _a_after_in_parent_fork_called

    assert _b_before_fork_called == 1, _b_before_fork_called
    assert _b_after_in_child_fork_called == 0, _b_after_in_child_fork_called
    assert _b_after_in_parent_fork_called == 1, _b_after_in_parent_fork_called

    _a_before_fork_called = 0
    _a_after_in_child_fork_called = 0
    _a_after_in_parent_fork_called = 0

    _b_before_fork_called = 0
    _b_after_in_child_fork_called = 0
    _b_after_in_parent_fork_called = 0

    _before_fork_called = 0
    _after_in_child_fork_called = 0
    _after_in_parent_fork_called = 0

    del a
    del b
    del before_fork
    del after_in_child_fork
    del after_in_parent_fork

    def process_verify_fork_hooks_2():
        assert _before_fork_called == 0, _before_fork_called
        assert _after_in_child_fork_called == 0, _after_in_child_fork_called
        assert _after_in_parent_fork_called == 0, _after_in_parent_fork_called

        assert _a_before_fork_called == 0, _a_before_fork_called
        assert _a_after_in_child_fork_called == 0, _a_after_in_child_fork_called
        assert _a_after_in_parent_fork_called == 0, _a_after_in_parent_fork_called

        assert _b_before_fork_called == 0, _b_before_fork_called
        assert _b_after_in_child_fork_called == 0, _b_after_in_child_fork_called
        assert _b_after_in_parent_fork_called == 0, _b_after_in_parent_fork_called

    p2 = multiprocessing.Process(target=process_verify_fork_hooks_2)
    p2.start()
    p2.join()
    assert p2.exitcode == 0, p2.exitcode

    assert _before_fork_called == 0, _before_fork_called
    assert _after_in_child_fork_called == 0, _after_in_child_fork_called
    assert _after_in_parent_fork_called == 0, _after_in_parent_fork_called

    assert _a_before_fork_called == 0, _a_before_fork_called
    assert _a_after_in_child_fork_called == 0, _a_after_in_child_fork_called
    assert _a_after_in_parent_fork_called == 0, _a_after_in_parent_fork_called

    assert _b_before_fork_called == 0, _b_before_fork_called
    assert _b_after_in_child_fork_called == 0, _b_after_in_child_fork_called
    assert _b_after_in_parent_fork_called == 0, _b_after_in_parent_fork_called
