---
name: writing-tests
description: Use when adding or changing tests in Megatron-Energon, or when a code change needs test coverage. Covers the end-to-end test style, dataset fixtures, save/restore and multi-worker checks, and how to run focused tests.
---

<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Writing tests

Tests use `unittest`, not `pytest`. Prefer end-to-end tests over many small unit tests.

## Approach

1. Create a small but real dataset in a temporary directory (`tempfile.TemporaryDirectory`,
   removed in `tearDown`/`tearDownClass`). Reuse the dataset-creation helpers of the nearest
   existing test file instead of writing new ones.
2. Load it through the public API: `get_train_dataset`, `get_val_dataset`, `get_loader`,
   `get_savable_loader`, ... Use `WorkerConfig(rank=0, world_size=1, num_workers=0)` for simple
   cases.
3. Assert on the samples that come out (keys, content, order or determinism), not on internals.
4. Where the change can affect it, also cover:
   - save/restore of the loader state (`save_state_rank` / `restore_state_rank`): the samples
     after restoring must equal the samples the uninterrupted loader produces.
   - multiple workers (`num_workers > 0`) and multiple ranks (`world_size > 1`).
5. A bug fix needs a test that fails without the fix.

## Rules

- Only cover behavior that matters to the user. Not every small piece of functionality needs its
  own test.
- Do not add a micro-test per function or test private helpers in isolation.
- Extend an existing test file before creating a new one. New files go in `tests/` as
  `test_<topic>.py` and need the NVIDIA license header (see an existing test).
- Keep datasets tiny and tests fast; no network, GPU or external data.
- Reference pattern: `tests/test_dataset.py`.

## Running

```bash
just test                                                    # full suite
uv run python -m unittest tests.test_dataset                 # one module
uv run python -m unittest tests.test_dataset.TestDataset.test_loader_close  # one test
```

Run `just fix` and `just check` after editing tests. When reporting verification, include the exact
command and its result.
