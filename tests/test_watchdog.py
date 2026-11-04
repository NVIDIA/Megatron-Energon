# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import torch

from megatron.energon.watchdog import PRINT_LOCAL_MAX_LENGTH, Watchdog, repr_short


class Unprintable:
    def __repr__(self):
        raise RuntimeError("Cannot format this value")


class UninspectableTensor(torch.Tensor):
    @property
    def is_cuda(self):
        raise RuntimeError("Cannot inspect this tensor")


class TestWatchdog(unittest.TestCase):
    def test_unprintable_locals(self):
        for value in (Unprintable(), [Unprintable()], {"sample": Unprintable()}):
            with self.subTest(value_type=type(value).__name__):
                self.assertEqual(
                    repr_short(value),
                    f"<unrepresentable {type(value).__name__}: RuntimeError>",
                )

    def test_tensor_inspection_failure(self):
        value = torch.empty(0).as_subclass(UninspectableTensor)
        self.assertEqual(repr_short(value), "<unrepresentable UninspectableTensor: RuntimeError>")

    def test_existing_representations(self):
        for value in (None, 42, {"sample": "text"}, torch.tensor([1, 2])):
            self.assertEqual(repr_short(value), repr(value))
        value = "x" * PRINT_LOCAL_MAX_LENGTH
        representation = repr(value)
        self.assertEqual(
            repr_short(value),
            representation[: PRINT_LOCAL_MAX_LENGTH // 2]
            + "..."
            + representation[-PRINT_LOCAL_MAX_LENGTH // 2 :],
        )

    def test_timeout_continues_after_unprintable_local(self):
        def stalled_worker():
            broken = Unprintable()
            following = "still visible"
            yield broken, following

        worker = stalled_worker()
        next(worker)
        callback = Mock()
        watchdog = Watchdog(timeout=60, callback=callback, enabled=False)
        output = io.StringIO()
        try:
            with (
                patch("sys._current_frames", return_value={-1: worker.gi_frame}),
                contextlib.redirect_stdout(output),
            ):
                watchdog._on_timeout()
        finally:
            watchdog.finish()
            worker.close()
        callback.assert_called_once_with()
        self.assertIn("<unrepresentable Unprintable: RuntimeError>", output.getvalue())
        self.assertIn("following: 'still visible'", output.getvalue())


if __name__ == "__main__":
    unittest.main()
