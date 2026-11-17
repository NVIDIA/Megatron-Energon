<!--- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Reproduce checkpoint failures near end of input

These investigation notes accompany `repro_checkpoint_eof.py`. They cover review
finding 2 (ordinary two-worker EOF checkpoint continuation), independently of
logical-worker fanout, processing hooks, packing, or batching.

## Expected behavior

A checkpoint saved after consuming N samples must restore exactly the suffix
that the original iterator would produce: same samples, same order, no duplicates.
With 50 samples, saving after 49 must restore one sample; saving immediately after
50 must restore zero samples.

The latter checkpoint is taken **before the consumer asks for the next item and
observes StopIteration**. It is not a request to start a new epoch. Workers may
already have reached EOF while prefetching; their progress is ahead of the consumer.

## Run the same script on either branch

Use Linux and the checkout's normal development dependencies (`uv sync --all-extras`
if the environment is not already prepared). No GPU, network dataset, or distributed
process group is needed. The script creates and cleans up its own temporary data.

From the checkout you want to test:

```bash
git rev-parse HEAD
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run python scripts/repro_checkpoint_eof.py
```

The script is portable between these branches; it does not import their test
modules. If it is not present in the target branch, copy it to a location outside
the checkout and use its absolute path instead:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" uv run python /absolute/path/repro_checkpoint_eof.py
```

Check the printed `Energon:` path to confirm that the intended checkout is imported.
Use separate worktrees if switching would disturb ongoing changes. Each branch
creates its own checkpoints; this is not a checkpoint-format migration test.

Optional commands for checking a candidate fix:

```bash
# Smallest pair of failing boundary cases:
PYTHONPATH="$PWD/src" uv run python scripts/repro_checkpoint_eof.py --positions 49 50

# Every checkpoint position, including before consuming the first sample:
PYTHONPATH="$PWD/src" uv run python scripts/repro_checkpoint_eof.py --all-positions

# Additional worker-count coverage:
PYTHONPATH="$PWD/src" uv run python scripts/repro_checkpoint_eof.py --workers 0
PYTHONPATH="$PWD/src" uv run python scripts/repro_checkpoint_eof.py --workers 1
```

Exit status **0** means all selected cases passed. Status **1** means an exception
or incorrect continuation was detected; the script prints the failing stage and
traceback or expected/actual samples. Output collection is capped at 51 samples
so an accidental repeating stream cannot run indefinitely. Run without `python -O`
so the baseline assertions remain enabled.

## What the script does

1. Creates five tar shards with ten unique text samples each, values 0 through 49.
2. Loads them with `get_val_dataset(..., batch_size=None)` and two workers by default.
   There is no shuffle or repetition. The expected two-worker order is
   `0, 25, 1, 26, ..., 24, 49`.
3. For each checkpoint position, constructs a fresh loader, consumes exactly N
   samples using `next()`, then calls `save_state_rank()` immediately.
4. Serializes and deserializes the state with pickle, then drains the original
   iterator to establish the expected continuation.
5. Constructs another fresh loader, calls `restore_state_rank()` **before entering
   its context manager**, and compares the restored sequence with that continuation.
   This ordering matters because the custom loader starts workers in `__enter__`.

The old loader uses `checkpoint_every_sec=0`,
`checkpoint_every_min_n_samples=1`, and `n_checkpoints=100`. This retains enough
checkpoint history for this tiny fixture and avoids an unrelated missing-history
failure. The custom loader uses its defaults, including two fork workers and
prefetch factor 2; the legacy checkpoint options are omitted there.

## Verified results, November 17, 2026

The same script was run against isolated source snapshots:

- Existing torch-based loader: `d93f32f2ab0b49f8016faa1d3c7a1e3fdfc2c463`,
  the inspected #346 fixes on top of #353.
- Custom loader: `90c87fdb43db2e7aab652b765b1f66149fee44e8`,
  [PR #162](https://github.com/NVIDIA/Megatron-Energon/pull/162).

These are pinned observations, not claims about future PR revisions.

| Samples consumed before save | Expected remaining | Existing loader | Custom loader |
| --- | --- | --- | --- |
| 20 | 30 | PASS | PASS |
| 44 | 6 | PASS | PASS |
| 45 | 5 | PASS | PASS |
| 46 | 4 | PASS | PASS |
| 47 | 3 | PASS | Save raises `StopIteration` |
| 48 | 2 | PASS | Save raises `StopIteration` |
| 49 | 1 | Restores 26 samples | Save raises `StopIteration` |
| 50 | 0 | Restores 50 samples | Save raises `StopIteration` |

The existing loader reports `6/8 cases passed` and exits 1. After 49 samples,
the expected continuation is `[49]`, but the restored continuation is
`[49, 0, 1, ..., 24]`: the finished worker's 25 samples are replayed.
After 50 samples, the expected continuation is empty, but all 50 samples reappear.

The custom loader reports `4/8 cases passed` and exits 1. The exception originates
in `src/megatron/energon/dataloader/dataloader.py:481` at the inspected commit:

```python
[get_sample_restore_key(sample_fut.get()) for sample_fut in prefetching_sample]
```

A prefetched result can represent EOF instead of a sample, so `.get()` raises
`StopIteration` before a checkpoint can be returned. Consequently these failing
cases never reach restore; they do not establish whether restore itself is correct.

## What a fix must preserve

All default cases should pass, including `after=49: remaining=1` and
`after=50: remaining=0`. Run the optional full-position sweep on a candidate fix.
Fixing only the exception is insufficient: samples already prefetched must still
be delivered exactly once, in order, and finished input must not restart while
continuing the saved iteration. Keep explicit next-epoch iteration working as well;
that separate behavior is not tested by this script.

The custom loader needs to distinguish worker-side exhaustion from consumer-side
completion when buffered samples or EOF signals remain. Merely dropping EOF
results, or marking a worker finished and skipping its buffered samples, can hide
the exception while losing data or restarting an epoch.

## Validation of this artifact

Both pinned runs used the existing Python 3.12 environment in isolated Linux
sandboxes, with each snapshot's `src` on `PYTHONPATH`, via
`uv run --no-project --no-sync --python <existing-venv>/bin/python scripts/repro_checkpoint_eof.py`.
Both produced the expected nonzero exits above. The optional commands are provided
for follow-up validation; they were not part of these eight-case runs.

Scoped `uv run ruff check scripts/repro_checkpoint_eof.py` and
`uv run ruff format --check scripts/repro_checkpoint_eof.py` passed using that
environment. The repository's `just fix` and `just check` could not be run because
`just` is not installed on this host; the scoped Ruff operations were used instead.
No production loader code was changed.
