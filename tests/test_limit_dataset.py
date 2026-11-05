# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from types import SimpleNamespace
from unittest import mock

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.limit_dataset import LimitDataset


class CountingDataset(SavableDataset[int]):
    _savable_fields = ("cursor",)

    def __init__(self, worker_config, length=10):
        super().__init__(worker_config)
        self.length = length
        self.cursor = 0
        self.reads = 0
        self.resets = 0

    def __iter__(self):
        while self.cursor < self.length:
            sample = self.cursor
            self.cursor += 1
            self.reads += 1
            yield sample

    def len_worker(self, worker_idx=None):
        return self.length

    def worker_has_samples(self):
        return self.length > 0

    def reset_state_own(self):
        self.cursor = 0
        self.resets += 1

    def config(self):
        return {"type": type(self).__qualname__, "length": self.length}


class TestLimitDataset(unittest.TestCase):
    def setUp(self):
        self.worker_config = WorkerConfig(rank=0, world_size=1, num_workers=0)

    def make_dataset(self, limit, reset_after_epoch=False, source_length=10):
        source = CountingDataset(self.worker_config, source_length)
        limited = LimitDataset(
            source, limit, reset_after_epoch=reset_after_epoch, worker_config=self.worker_config
        )
        return source, limited

    def test_does_not_consume_beyond_limit(self):
        for limit in (0, 1, 3, 10, 15):
            for reset in (False, True):
                with self.subTest(limit=limit, reset=reset):
                    source, limited = self.make_dataset(limit, reset)
                    self.assertEqual(list(limited), list(range(min(limit, 10))))
                    self.assertEqual(source.reads, min(limit, 10))
                    self.assertEqual(source.resets, int(reset))

    def test_continues_from_next_unconsumed_sample(self):
        source, limited = self.make_dataset(3)
        self.assertEqual(list(limited), [0, 1, 2])
        self.assertEqual(list(limited), [3, 4, 5])
        self.assertEqual(list(limited), [6, 7, 8])
        self.assertEqual(list(limited), [9])
        self.assertEqual(source.reads, 10)
        self.assertEqual(source.resets, 0)

    def test_explicit_reset_repeats_window(self):
        source, limited = self.make_dataset(3, True)
        self.assertEqual(list(limited), [0, 1, 2])
        self.assertEqual(list(limited), [0, 1, 2])
        self.assertEqual(source.reads, 6)
        self.assertEqual(source.resets, 2)

    def test_per_worker_limits_preserve_unconsumed_samples(self):
        for limit in (1, 5):
            for worker_id in range(3):
                with (
                    self.subTest(limit=limit, worker_id=worker_id),
                    mock.patch(
                        "torch.utils.data.get_worker_info",
                        return_value=SimpleNamespace(id=worker_id, num_workers=3),
                    ),
                ):
                    config = WorkerConfig(rank=0, world_size=1, num_workers=3)
                    source = CountingDataset(config)
                    limited = LimitDataset(source, limit, worker_config=config)
                    local_limit = limit // 3 + int(worker_id < limit % 3)
                    self.assertEqual(list(limited), list(range(local_limit)))
                    self.assertEqual(source.reads, local_limit)
                    self.assertEqual(source.resets, 0)

    def test_checkpoint_resumes_remainder_and_next_window(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                _, limited = self.make_dataset(3, reset)
                iterator = iter(limited)
                self.assertEqual(next(iterator), 0)
                checkpoint = limited.save_state()
                expected_tail = list(iterator)
                expected_next = list(limited)
                source, restored = self.make_dataset(3, reset)
                restored.restore_state(checkpoint)
                self.assertEqual(list(restored), expected_tail)
                self.assertEqual(list(restored), expected_next)
                self.assertEqual(expected_tail, [1, 2])
                self.assertEqual(expected_next, [0, 1, 2] if reset else [3, 4, 5])
                self.assertEqual(source.reads, 5)


if __name__ == "__main__":
    unittest.main()
