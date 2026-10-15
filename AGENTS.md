<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Repository Guidelines

Instructions for AI coding agents working in this repository. Humans should read
[CONTRIBUTING.md](CONTRIBUTING.md); everything there applies to agents as well.

## What this project is

Megatron Energon is a general-purpose PyTorch data loader for large-scale, multi-modal training.
It was built for [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), but it is a standalone
library and must stay usable with any PyTorch training loop. Do not introduce dependencies on
Megatron-Core.

Its core promises are: loading large datasets efficiently, blending many datasets, distributing
work across nodes, ranks and workers, and exact reproducibility and resumability (save/restore of
the loader state). Any change must preserve these.

Layout of `src/megatron/energon/`:

- `flavors/`: dataset formats (webdataset, jsonl, crude datasets) and sample types.
- `recipe/`: recipe (YAML) parsing and dataset blending configuration.
- `task_encoder/`: user hook for decoding, cooking, encoding and batching samples.
- `wrappers/`: composable iterable datasets (batch, blend, shuffle, packing, ...).
- `loader.py`, `savable_loader.py`, `worker.py`: data loaders, worker setup and state saving.
- `cache/`, `epathlib/`, `av/`, `media/`, `decoders/`: file caching, local/remote paths,
  audio/video decoding, media metadata, sample decoding.
- `cli/`, `tools/`: the `energon` command line tool.

User documentation lives in `docs/source` (Sphinx, MyST markdown).

## Environment and commands

- The project is managed with `uv` (`pyproject.toml`, `uv.lock`). Never hand-edit `uv.lock`.
  Use `uv run ...` instead of activating `.venv`.
- The `justfile` is the canonical entry point:
  - `just dev-sync`: sync all extras into `.venv`.
  - `just fix`: apply ruff fixes, formatting and license headers. Run it after every change.
  - `just check`: ruff lint and format check. Must pass.
  - `just test`: full test suite.
  - `just check-commits`: check sign-off and commit message format of your branch.
- Tests use `unittest`, not `pytest`. Run focused tests with dotted names:

  ```bash
  uv run python -m unittest tests.test_dataset.TestDataset.test_loader_close
  ```

- When reporting verification, include the exact command that was run and its result.

## Code

- Clean, maintainable, readable code is the bar. Follow the conventions of the surrounding
  module; match its naming, structure and comment density.
- Keep diffs minimal and on-topic: no drive-by refactors, reformatting or renames outside the
  task, no commented-out code.
- Every new `.py`, `.sh` and `.md` file needs the NVIDIA license header (`just fix` adds it in
  `src/` and `tests/`; copy it from an existing file elsewhere).
- User-facing changes need matching documentation in `docs/source`.

## Tests

Prefer end-to-end tests over many small unit tests:

- Create a small but real dataset in a temporary directory, load it through the public API
  (`get_train_dataset`, `get_val_dataset`, `get_savable_loader`, ...), and assert on the samples
  that come out. See `tests/test_dataset.py` for the pattern.
- Cover save/restore of the loader state and multiple workers where the change can affect them.
- Do not add a micro-test per function or test private helpers in isolation.
- Extend an existing test file before creating a new one.

## Commits and pull requests

- Follow the commit message conventions in [CONTRIBUTING.md](CONTRIBUTING.md#commit-messages)
  (Conventional Commits).
- Always commit with `git commit -s`. The human you work for signs off and is responsible for
  the change. Never add `Co-Authored-By` (or similar) trailers for AI assistants.
- Never push, open pull requests or issues, or comment on GitHub unless the human explicitly asks
  you to. Create pull requests as drafts and fill in the PR template, but leave the human review
  checkbox unticked: only the human may tick it.
