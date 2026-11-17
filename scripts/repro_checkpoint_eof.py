# Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: BSD-3-Clause

"""Check exact checkpoint continuation near EOF on both Energon loader implementations."""

import argparse
import inspect
import pickle
import tempfile
import traceback
from itertools import islice
from pathlib import Path

import webdataset as wds

import megatron.energon
from megatron.energon import WorkerConfig, get_savable_loader, get_val_dataset
from megatron.energon.flavors import BaseWebdatasetFactory


def create_dataset(path: Path) -> None:
    """Make 50 unique text samples, using the repository's text-fixture pattern."""
    with wds.ShardWriter(str(path / "data-%d.tar"), maxcount=10) as writer:
        for index in range(50):
            writer.write({"__key__": f"{index:06d}", "txt": str(index).encode()})
    BaseWebdatasetFactory.prepare_dataset(
        path,
        ["data-{0..4}.tar"],
        split_parts_ratio=[("train", 1.0)],
        shuffle_seed=None,
    )
    (path / ".nv-meta" / "dataset.yaml").write_text(
        "sample_type:\n  __module__: megatron.energon\n"
        "  __class__: TextSample\nfield_map:\n  text: txt\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--positions", type=int, nargs="+", default=[20, 44, 45, 46, 47, 48, 49, 50]
    )
    parser.add_argument("--all-positions", action="store_true", help="Check every position 0..50")
    args = parser.parse_args()
    positions = list(range(51)) if args.all_positions else args.positions
    if args.workers < 0 or any(position < 0 or position > 50 for position in positions):
        parser.error("workers must be nonnegative and positions must be within 0..50")

    # Old checkpoints need enough history to rewind past prefetch. The custom
    # loader ignores those compatibility options, so do not pass them there.
    custom = "worker_type" in inspect.signature(get_savable_loader).parameters
    options = (
        {}
        if custom
        else {
            "checkpoint_every_sec": 0,
            "checkpoint_every_min_n_samples": 1,
            "n_checkpoints": 100,
        }
    )
    print(f"Energon: {megatron.energon.__file__}", flush=True)
    print(f"Loader: {'custom' if custom else 'torch-based'}; workers={args.workers}", flush=True)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="energon-eof-") as directory:
        path = Path(directory)
        create_dataset(path)
        config = WorkerConfig(rank=0, world_size=1, num_workers=args.workers)

        def build():
            return get_savable_loader(
                get_val_dataset(path, split_part="train", worker_config=config, batch_size=None),
                **options,
            )

        with build() as loader:
            baseline = [sample.text for sample in islice(iter(loader), 51)]
        assert len(baseline) == 50 and set(baseline) == {str(index) for index in range(50)}, (
            f"Invalid baseline: {baseline}"
        )

        for count in positions:
            stage = "consume prefix"
            try:
                with build() as loader:
                    iterator = iter(loader)
                    prefix = [next(iterator).text for _ in range(count)]
                    assert prefix == baseline[:count], "Prefix changed between fresh loaders"
                    # Do not ask for another sample before saving: even at count=50,
                    # the consumer has not yet observed StopIteration.
                    stage = "save_state_rank"
                    state = loader.save_state_rank()
                    stage = "serialize checkpoint"
                    state = pickle.loads(pickle.dumps(state))
                    stage = "uninterrupted continuation"
                    expected = [sample.text for sample in islice(iterator, 51)]
                    assert expected == baseline[count:], "Uninterrupted continuation changed"

                stage = "restore_state_rank"
                restored = build()
                try:
                    # The custom loader starts workers in __enter__, so restore first.
                    restored.restore_state_rank(state)
                    stage = "restored continuation"
                    with restored:
                        actual = [sample.text for sample in islice(iter(restored), 51)]
                finally:
                    # Also clean up if restore fails before entering the context.
                    if custom:
                        restored.shutdown()
                    else:
                        restored.close()
                if actual != expected:
                    failures += 1
                    print(
                        f"FAIL after={count}: expected={len(expected)} restored={len(actual)}"
                        " (capped at 51)",
                        flush=True,
                    )
                    print(f"  expected: {expected}\n  restored: {actual}", flush=True)
                else:
                    print(f"PASS after={count}: remaining={len(actual)}", flush=True)
            except Exception as error:  # noqa: BLE001 - report every case, then exit nonzero
                failures += 1
                print(
                    f"FAIL after={count} during {stage}: {type(error).__name__}: {error}",
                    flush=True,
                )
                traceback.print_exc()
    print(f"{len(positions) - failures}/{len(positions)} cases passed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
