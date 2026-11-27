# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import pickle
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

from megatron.energon import (
    DefaultTaskEncoder,
    WorkerConfig,
    get_savable_loader,
    get_train_dataset,
    skip_safe,
    stateless,
)
from tests.test_dataloader import create_text_test_dataset


class SeededEncoder(DefaultTaskEncoder):
    @stateless(restore_task_encoder_seeds=True, skip_safe=True)
    def encode_sample(self, sample):
        return replace(sample, text=f"{sample.text}:{self.rng.random.random()}")


class TestLogicalWorkers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        create_text_test_dataset(self.path, range(55), range(55))

    def loader(
        self,
        workers=0,
        *,
        rank=0,
        world=1,
        logical=1,
        backend="fork",
        prefetch=2,
        epochize=None,
        limit=None,
        repeat=True,
        encoder=None,
        batch=1,
        packing=None,
    ):
        config = WorkerConfig(
            rank=rank,
            world_size=world,
            num_workers=workers,
            logical_workers=logical,
            seed_offset=42,
        )
        return get_savable_loader(
            get_train_dataset(
                self.path,
                worker_config=config,
                batch_size=batch,
                packing_buffer_size=packing,
                shuffle_buffer_size=5,
                max_samples_per_sequence=2,
                repeat=repeat,
                task_encoder=encoder or SeededEncoder(),
            ),
            worker_type=backend,
            prefetch_factor=prefetch,
            epochize=epochize,
            limit=limit,
            pin_memory=False,
        )

    @staticmethod
    def take(loader, count):
        return [sample.text for sample in itertools.islice(loader, count)]

    def test_fanout_preserves_seeded_stream(self):
        with self.loader() as reference:
            expected = self.take(reference, 24)
        for backend in ("fork", "thread"):
            for workers in (1, 2, 4):
                with (
                    self.subTest(backend=backend, workers=workers),
                    self.loader(workers, backend=backend) as loader,
                ):
                    self.assertEqual(self.take(loader, 24), expected)

    def test_logical_restore_at_partial_rounds_with_prefetch(self):
        with self.loader() as reference:
            expected = self.take(reference, 28)
        for backend in ("fork", "thread"):
            for prefix in (0, 1, 3, 5):
                with self.subTest(backend=backend, prefix=prefix):
                    with self.loader(4, backend=backend, prefetch=3) as source:
                        self.assertEqual(self.take(source, prefix), expected[:prefix])
                        state = pickle.loads(pickle.dumps(dict(source.save_state_logical_rank())))
                    for target in (0, 2, 4):
                        loader = self.loader(target, backend=backend)
                        loader.restore_state_logical_rank(state)
                        with loader:
                            self.assertEqual(self.take(loader, 12), expected[prefix : prefix + 12])

    def test_physical_restore_and_resave_logical(self):
        for backend in ("fork", "thread"):
            with self.loader(4, backend=backend) as source:
                self.take(source, 3)
                physical = source.save_state_rank()
                expected = self.take(source, 16)
            restored = self.loader(4, backend=backend)
            restored.restore_state_rank(physical)
            with restored:
                self.assertEqual(self.take(restored, 1), expected[:1])
                logical = dict(restored.save_state_logical_rank())
            target = self.loader(2, backend=backend)
            target.restore_state_logical_rank(logical)
            with target:
                self.assertEqual(self.take(target, 12), expected[1:13])

    def test_boundaries_continue_or_reset_and_restore(self):
        for workers in (0, 4):
            for backend in ("fork", "thread"):
                for option in ("epochize", "limit"):
                    with self.subTest(workers=workers, backend=backend, option=option):
                        kwargs = {option: 7}
                        with self.loader(workers, backend=backend, **kwargs) as source:
                            first = self.take(source, 3)
                            state = source.save_state_rank()
                            remainder = [sample.text for sample in source]
                            second = [sample.text for sample in source]
                            self.assertEqual(len(source), 7)
                            self.assertEqual(len(first) + len(remainder), 7)
                            self.assertEqual(len(second), 7)
                            if option == "limit":
                                self.assertEqual(first + remainder, second)
                            else:
                                self.assertNotEqual(first + remainder, second)
                        restored = self.loader(workers, backend=backend, **kwargs)
                        restored.restore_state_rank(state)
                        with restored:
                            self.assertEqual([sample.text for sample in restored], remainder)
                            self.assertEqual([sample.text for sample in restored], second)

    def test_logical_restore_boundary_with_changed_fanout(self):
        for option in ("epochize", "limit"):
            for prefix in (0, 1, 3, 7):
                with self.subTest(option=option, prefix=prefix):
                    kwargs = {option: 7}
                    with self.loader(4, **kwargs) as source:
                        self.take(source, prefix)
                        state = dict(source.save_state_logical_rank())
                        expected = [sample.text for sample in source]
                        following = [sample.text for sample in source]
                    for workers in (0, 2):
                        target = self.loader(workers, **kwargs)
                        target.restore_state_logical_rank(state)
                        with target:
                            self.assertEqual([sample.text for sample in target], expected)
                            self.assertEqual([sample.text for sample in target], following)

    def test_rank_changes_preserve_each_logical_stream(self):
        with ExitStack() as stack:
            sources = [
                stack.enter_context(self.loader(2, rank=rank, world=3, logical=2))
                for rank in range(3)
            ]
            streams = [iter(loader) for loader in sources]
            for _ in range(3):
                for iterator in streams:
                    next(iterator)
            states = dict(pair for loader in sources for pair in loader.save_state_logical_rank())
            expected = {0: [], 1: []}
            for step in range(12):
                for rank, iterator in enumerate(streams):
                    sample = next(iterator)
                    group = sources[rank]._worker_config.logical_global_worker_id((3 + step) % 2)
                    expected[group].append(sample.text)
        with ExitStack() as stack:
            targets = [self.loader(2, rank=rank, world=2, logical=2) for rank in range(2)]
            for target in targets:
                target.restore_state_logical_rank(states)
                stack.enter_context(target)
            for group, target in enumerate(targets):
                self.assertEqual(self.take(target, 12), expected[group][:12])

    def test_sample_restore_after_topology_change(self):
        with self.loader(4) as source:
            samples = list(itertools.islice(source, 5))
            state = dict(source.save_state_logical_rank())
        target = self.loader(2)
        target.restore_state_logical_rank(state)
        with target:
            for sample in samples:
                self.assertEqual(target.restore_sample(sample.__restore_key__).text, sample.text)

    def test_skip_safe_only_omits_pure_work(self):
        class CountingEncoder(DefaultTaskEncoder):
            def __init__(self):
                super().__init__()
                self.calls = []

            @stateless
            def encode_sample(self, sample):
                self.calls.append(sample.__key__)
                return sample

        class SafeCountingEncoder(CountingEncoder):
            @skip_safe
            @stateless
            def encode_sample(self, sample):
                return super().encode_sample(sample)

        for safe in (False, True):
            encoder = SafeCountingEncoder() if safe else CountingEncoder()
            with self.loader(2, backend="thread", encoder=encoder, prefetch=0) as loader:
                self.take(loader, 8)
            self.assertEqual(len(encoder.calls), 8 if safe else 15)

    def test_invalid_logical_checkpoints_and_boundary_lengths(self):
        with self.loader(2) as source:
            state = dict(source.save_state_logical_rank())
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.loader(epochize=invalid)
        for bad in ({}, list(state.items()) * 2):
            target = self.loader(2)
            with self.assertRaises(ValueError):
                target.restore_state_logical_rank(bad)
            target.close()

    def test_finite_stream_checkpoints_at_final_yield_and_eof(self):
        for prefix in (0, 1, 54, 55, 56):
            with self.subTest(prefix=prefix):
                with self.loader(4, repeat=False) as source:
                    self.take(source, prefix)
                    state = dict(source.save_state_logical_rank())
                    expected = [sample.text for sample in source]
                    following = [sample.text for sample in source]
                target = self.loader(2, repeat=False)
                target.restore_state_logical_rank(state)
                with target:
                    self.assertEqual([sample.text for sample in target], expected)
                    self.assertEqual([sample.text for sample in target], following)

    def test_packing_fanout_and_logical_restore(self):
        from megatron.energon import TextSample

        class PackingEncoder(DefaultTaskEncoder):
            @stateless
            def preencode_sample(self, sample):
                return sample

            @stateless
            def select_samples_to_pack(self, samples):
                return [samples[i : i + 3] for i in range(0, len(samples), 3)]

            @stateless
            def select_next_pack(self, samples):
                pack = list(itertools.islice(samples, 3))
                return [pack] if pack else []

            @stateless(restore_task_encoder_seeds=True, skip_safe=True)
            def postencode_sample(self, sample):
                return replace(sample, text=f"{sample.text}:{self.rng.random.random()}")

            @skip_safe
            @stateless
            def pack_selected_samples(self, samples):
                return TextSample.derive_from(
                    samples[0],
                    __key__="|".join(x.__key__ for x in samples),
                    __restore_key__=None,
                    text="|".join(x.text for x in samples),
                )

        for packing in (8, "stream"):
            for backend in ("fork", "thread"):
                with self.subTest(packing=packing, backend=backend):
                    with self.loader(packing=packing, encoder=PackingEncoder()) as baseline:
                        expected = self.take(baseline, 20)
                    with self.loader(
                        4, backend=backend, packing=packing, encoder=PackingEncoder()
                    ) as source:
                        self.assertEqual(self.take(source, 3), expected[:3])
                        state = dict(source.save_state_logical_rank())
                        self.assertEqual(self.take(source, 12), expected[3:15])
                    target = self.loader(
                        2, backend=backend, packing=packing, encoder=PackingEncoder()
                    )
                    target.restore_state_logical_rank(state)
                    with target:
                        self.assertEqual(self.take(target, 12), expected[3:15])

    def test_smaller_micro_batches_and_physical_save_before_start(self):
        with self.loader(batch=1) as reference:
            expected = [text for batch in self.take(reference, 28) for text in batch]
        with self.loader(4, batch=4) as source:
            self.take(source, 3)
            logical = dict(source.save_state_logical_rank())
        target = self.loader(2, batch=1)
        target.restore_state_logical_rank(logical)
        physical = target.save_state_rank()
        target.close()
        restored = self.loader(2, batch=1)
        restored.restore_state_rank(physical)
        with restored:
            self.assertEqual(
                [text for batch in self.take(restored, 12) for text in batch], expected[12:24]
            )

    def test_recipe_migration_cannot_discard_prefetched_logical_outputs(self):
        from megatron.energon.checkpoint.rank_state import without_prefetched_batches

        with self.loader(2, prefetch=2) as loader:
            self.take(loader, 1)
            state = loader.save_state_rank()
        self.assertTrue(any(state.prefetched_restore_keys))
        with self.assertRaisesRegex(
            ValueError, "Cannot discard prefetched batches with logical workers"
        ):
            without_prefetched_batches(state, rank=0)

    def test_distributed_framework_can_gather_logical_checkpoints(self):
        script = dedent("""
            import itertools
            import multiprocessing as mp
            import sys
            from pathlib import Path
            import torch.distributed as dist
            from megatron.energon import WorkerConfig, get_savable_loader, get_train_dataset
            from tests.test_logical_workers import SeededEncoder

            def loader(rank=0, world=1, workers=0):
                return get_savable_loader(get_train_dataset(Path(sys.argv[1]),
                    worker_config=WorkerConfig(rank=rank, world_size=world, num_workers=workers,
                        logical_workers=1, seed_offset=42), batch_size=1,
                    shuffle_buffer_size=5, max_samples_per_sequence=2, task_encoder=SeededEncoder()),
                    worker_type='thread', pin_memory=False)

            def run(rank):
                dist.init_process_group('gloo', init_method='file://'+sys.argv[2], rank=rank, world_size=2)
                try:
                    with loader(rank, 2, 2) as source:
                        list(itertools.islice(source, 3))
                        contributions = [None] * 2
                        dist.all_gather_object(contributions, source.save_state_logical_rank())
                        state = dict(pair for entries in contributions for pair in entries)
                    target = loader(rank, 2, 1)
                    target.restore_state_logical_rank(state)
                    with target:
                        outputs = [x.text for x in itertools.islice(target, 6)]
                    with loader() as reference:
                        expected = [x.text for x in itertools.islice(reference, 18)]
                    assert outputs == expected[6+rank:18:2], (outputs, expected)
                finally:
                    dist.destroy_process_group()

            if __name__ == '__main__':
                ctx = mp.get_context('fork')
                processes = [ctx.Process(target=run, args=(rank,)) for rank in range(2)]
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(45)
                    if process.is_alive():
                        process.terminate()
                        process.join()
                    assert process.exitcode == 0, process.exitcode
        """)
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.path), str(self.path / "rendezvous")],
            capture_output=True,
            text=True,
            timeout=100,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_filters_and_generators_still_define_the_stream(self):
        from megatron.energon import FilterDataset
        from tests.test_dataloader import GeneratorTaskEncoder

        class FilterEncoder(SeededEncoder):
            def build_train_datasets(self, *, worker_config, **kwargs):
                dataset = super().build_train_datasets(worker_config=worker_config, **kwargs)
                return FilterDataset(
                    dataset,
                    worker_config=worker_config,
                    filter_fn=lambda sample: int(sample.text[0].split(":")[0]) % 2 == 0,
                )

        for encoder in (FilterEncoder, GeneratorTaskEncoder):
            with self.subTest(encoder=encoder):
                with self.loader(encoder=encoder()) as reference:
                    expected = self.take(reference, 20)
                with self.loader(4, encoder=encoder()) as source:
                    self.assertEqual(self.take(source, 1), expected[:1])
                    state = dict(source.save_state_logical_rank())
                restored = self.loader(2, encoder=encoder())
                restored.restore_state_logical_rank(state)
                with restored:
                    self.assertEqual(self.take(restored, 12), expected[1:13])

    def test_add_remove_and_change_loader_budgets_on_logical_restore(self):
        with self.loader() as reference:
            expected = self.take(reference, 25)
        for old_length, new_length in ((None, 7), (7, None), (7, 10), (10, 7)):
            with self.subTest(old=old_length, new=new_length):
                with self.loader(4, epochize=old_length) as source:
                    self.take(source, 3)
                    state = dict(source.save_state_logical_rank())
                restored = self.loader(2, epochize=new_length)
                restored.restore_state_logical_rank(state)
                with restored:
                    count = 12 if new_length is None else new_length - 3
                    self.assertEqual(self.take(restored, count), expected[3 : 3 + count])
                    if new_length is not None:
                        self.assertEqual(list(restored), [])
                        self.assertEqual(
                            self.take(restored, new_length), expected[new_length : 2 * new_length]
                        )

    def test_limit_caps_source_length_and_combined_budgets_reset(self):
        for workers in (0, 2, 4):
            with self.loader(workers, repeat=False) as reference:
                source_length = len(reference)
                expected = [sample.text for sample in reference]
            for options, length in (({"epochize": 60}, 60), ({"limit": 60}, source_length)):
                with (
                    self.subTest(workers=workers, options=options),
                    self.loader(workers, repeat=False, **options) as loader,
                ):
                    self.assertEqual(len(loader), length)
                    self.assertEqual([sample.text for sample in loader], expected)
            for epochize, limit in ((7, 10), (10, 7)):
                with self.loader(workers, epochize=epochize, limit=limit) as loader:
                    self.assertEqual(len(loader), 7)
                    first = [sample.text for sample in loader]
                    self.assertEqual(len(first), 7)
                    self.assertEqual([sample.text for sample in loader], first)
