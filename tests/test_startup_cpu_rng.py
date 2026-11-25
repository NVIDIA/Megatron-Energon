# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import random
import unittest
from threading import local
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from megatron.energon.rng import SystemRng
from megatron.energon.task_encoder.base import DefaultTaskEncoder, stateless
from megatron.energon.worker import WorkerConfig


class OptimizationsTest(unittest.TestCase):
    def test_cpu_rng_matches_legacy(self):
        for seed in (0, 1, 42, 2**32 - 1):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
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

    def test_stateless_cpu_rng_function_generator_and_exception(self):
        class Encoder:
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
            SimpleNamespace(worker_seed=lambda: 42, active_worker_invocation_index=7),
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

    def test_task_encoder_reuses_lazy_rng_storage(self):
        class Encoder(DefaultTaskEncoder):
            @stateless(restore_task_encoder_seeds=True)
            def draw(self):
                return self.rng.random.random()

        config = WorkerConfig(rank=0, world_size=1, num_workers=0)
        config.worker_activate(sample_index=7)
        self.addCleanup(config.worker_deactivate)
        with patch("megatron.energon.task_encoder.base.local", wraps=local) as allocate:
            encoder = Encoder()
            allocate.assert_not_called()
            self.assertEqual(encoder.draw(), encoder.draw())
            allocate.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
