# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

from megatron.energon.flavors.base_dataset import SavableDataset
from megatron.energon.worker import WorkerConfig
from megatron.energon.wrappers.packing_dataset import PackingDataset


class Numbers(SavableDataset):
    _savable_fields = ("position",)

    def __init__(self, worker_config, length=53):
        super().__init__(worker_config)
        self.length = length
        self.position = 0

    def __iter__(self):
        while self.position < self.length:
            position = self.position
            self.position += 1
            yield self.restore_sample(("Numbers", position))

    def restore_sample(self, key):
        return {"value": key[-1], "__restore_key__": key}

    def can_restore_sample(self):
        return True

    def len_worker(self, worker_idx=None):
        return self.length

    def worker_has_samples(self):
        return self.length > 0

    def reset_state_own(self):
        self.position = 0

    def config(self):
        return {"type": "Numbers", "length": self.length}


def select(samples):
    # Non-uniform packs plus a finite partial window.
    return [samples[i : i + 3] for i in range(0, len(samples), 3)]


def pack(samples):
    return {"values": [sample["value"] for sample in samples]}


class PackingTest(unittest.TestCase):
    def create(self, deferred, length=53):
        workers = WorkerConfig(rank=0, world_size=1, num_workers=0)
        source = Numbers(workers, length)
        dataset = PackingDataset(
            source,
            10,
            select,
            pack,
            final_packer_stateless=True,
            defer_refill=deferred,
            worker_config=workers,
        )
        return workers, source, dataset

    def consume(self, dataset, count=None):
        dataset.worker_config.worker_activate(0)
        try:
            iterator = iter(dataset)
            result = []
            for sample in iterator:
                result.append(sample)
                if count is not None and len(result) == count:
                    break
            return result
        finally:
            dataset.worker_config.worker_deactivate()

    def test_full_stream_and_read_ahead(self):
        for length in (0, 1, 10, 11, 53):
            _, _, legacy = self.create(False, length)
            _, _, deferred = self.create(True, length)
            self.assertEqual(self.consume(legacy), self.consume(deferred))
        _, source, deferred = self.create(True)
        self.consume(deferred, 4)
        self.assertEqual(source.position, 10)
        _, source, legacy = self.create(False)
        self.consume(legacy, 4)
        self.assertEqual(source.position, 19)

    def test_checkpoint_resume_each_window_position(self):
        _, _, reference = self.create(True)
        expected = self.consume(reference)
        for count in range(1, len(expected)):
            _, _, dataset = self.create(True)
            prefix = self.consume(dataset, count)
            state = dataset.save_state()
            _, _, restored = self.create(True)
            restored.restore_state(state)
            self.assertEqual(prefix + self.consume(restored), expected, count)

    def test_reject_stateful_final_packer(self):
        workers = WorkerConfig(rank=0, world_size=1, num_workers=0)
        with self.assertRaisesRegex(ValueError, "stateless"):
            PackingDataset(
                Numbers(workers),
                10,
                select,
                pack,
                defer_refill=True,
                worker_config=workers,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
