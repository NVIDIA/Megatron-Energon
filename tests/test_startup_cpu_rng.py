# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from megatron.energon.rng import SystemRng
from megatron.energon.task_encoder.base import stateless
from megatron.energon.worker import WorkerConfig


class OptimizationsTest(unittest.TestCase):
    def test_cpu_rng_matches_legacy(self):
        for seed in (0, 1, 42, 2**32 - 1):
            SystemRng.seed(seed)
            expected = (
                torch.rand(100),
                np.random.rand(100),
                [random.random() for _ in range(100)],
            )
            with patch.object(
                torch, "manual_seed", side_effect=AssertionError("accelerator seeding")
            ):
                SystemRng.seed(seed)
            actual = (
                torch.rand(100),
                np.random.rand(100),
                [random.random() for _ in range(100)],
            )
            self.assertTrue(torch.equal(expected[0], actual[0]))
            np.testing.assert_array_equal(expected[1], actual[1])
            self.assertEqual(expected[2], actual[2])
        with patch.object(torch, "manual_seed", wraps=torch.manual_seed) as seed:
            SystemRng.seed(42, seed_accelerators=True)
            seed.assert_called_once_with(42)

    def test_stateless_cpu_rng_function_generator_and_exception(self):
        class Encoder:
            current_sample_index = 7

            @stateless(restore_seeds=True)
            def normal(self):
                return (torch.rand(5), np.random.rand(5), random.random())

            @stateless(restore_seeds=True)
            def generate(self):
                yield self.normal()
                yield self.normal()

            @stateless(restore_seeds=True)
            def fail(self):
                self.normal()
                raise ValueError("expected")

        with patch.object(
            WorkerConfig,
            "active_worker_config",
            SimpleNamespace(worker_seed=lambda: 42),
        ):
            encoder = Encoder()
            before = repr(SystemRng.save_state())
            with patch.object(
                torch, "manual_seed", side_effect=AssertionError("accelerator seeding")
            ):
                first, second = encoder.normal(), encoder.normal()
                self.assertTrue(torch.equal(first[0], second[0]))
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(first[2], second[2])
                self.assertEqual(before, repr(SystemRng.save_state()))
                generator = encoder.generate()
                next(generator)
                self.assertEqual(before, repr(SystemRng.save_state()))
                next(generator)
                self.assertEqual(before, repr(SystemRng.save_state()))
                with self.assertRaises(StopIteration):
                    next(generator)
                with self.assertRaisesRegex(ValueError, "expected"):
                    encoder.fail()
                self.assertEqual(before, repr(SystemRng.save_state()))


if __name__ == "__main__":
    unittest.main()
