<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(contribution-guidelines)=

# Contribution Guidelines

This workflow keeps formatting, tests, documentation, and compatibility review consistent across
contributions.

## Development environment

Energon supports Python 3.10 and newer. Install
[uv](https://docs.astral.sh/uv/) and [just](https://just.systems/), then create the complete development
environment:

```shell
just dev-sync
```

This runs `uv sync --all-extras` with the repository cache directory, installing test, lint, and
documentation dependencies in the project environment.

## Standard commands

```shell
just fix    # Apply Ruff formatting/fixes and repair license headers in src and tests
just check  # Check formatting, lint, copyright headers, and static types
just test   # Run the full unittest suite
just docs   # Build the Sphinx documentation
just build  # Build the Python distribution
```

`just fix` modifies files; review the diff afterward. It is the canonical formatting command, replacing
the old Black/isort workflow.

Run a focused unittest while developing, for example:

```shell
uv run python -m unittest tests.test_checkpoint_resume
uv run python -m unittest tests.test_logical_workers
```

Then run the full checks appropriate to the change before requesting review.

## Implementation conventions

- Prefer typed dataclasses and named tuples over untyped dictionaries when data has a defined schema.
- Keep configuration objects serializable and separate from mutable checkpoint state.
- Add type annotations to public and internal extension contracts.
- Write Google-style docstrings for supported classes and methods so Sphinx can publish them.
- Preserve `SourceInfo` and restore keys through transformations and wrappers.
- Use `WorkerRng` and the savable helpers instead of unsaved random generators or counters.
- Keep storage resources worker-local and close readers that own handles.
- Add or update user documentation for every supported feature or configuration field.

The design contracts behind these conventions are described in {ref}`code-structure`,
{ref}`dataset-formats`, and {ref}`savability`.

## Before opening a pull request

1. Inspect the complete diff from the target branch, including generated or copied files.
2. Search source, documentation, tests, configuration, and commit history for private project names,
   credentials, internal paths, and unrelated code.
3. Run `just fix`, then inspect any mechanical changes it made.
4. Run focused tests for the affected compatibility surface.
5. Run `just check`, `just test`, and `just docs`.
6. Update public documentation, docstrings, exports, compatibility aliases, and release notes as needed.
7. State any breaking surface explicitly in the pull request description.

Do not rely only on the final tree when checking for sensitive or unrelated content: review the commits
being submitted as well.

## Compatibility labels

Use the following exact labels in a pull request description when applicable:

- `CHECKPOINT BREAKING CHANGE`: exact save/restore structure changed incompatibly.
- `ITERATION ORDER BREAKING CHANGE`: a fixed configuration and seed may produce a different order.
- `API BREAKING CHANGE`: the supported Python API changed incompatibly.
- `DATASET CONFIG BREAKING CHANGE`: prepared `.nv-meta` data or indexes changed incompatibly.
- `RECIPE CONFIG BREAKING CHANGE`: recipe syntax or meaning changed incompatibly.

A change may require multiple labels. Describe the affected configurations and any migration. See
{ref}`compatibility` for the detailed review and test matrix.

## Tests for common changes

Useful regression tests include:

- `test_recipe.TestDataset.test_save_restore_state_train` for exact recipe save/restore;
- the current-batch-index tests in `test_dataset.py` for stable iteration;
- `test_checkpoint_resume.py` for recipe topology migration;
- `test_logical_workers.py` for physical/logical worker fanout;
- `test_dataset_factory_resolver.py` and format-specific suites for path detection;
- `test_filter_index.py` for filtered ordinal translation.

New behavior should normally have a focused test next to the closest existing suite. When fixing a bug,
write a test that fails for the original cause rather than only for one observed symptom.

## Documentation

User-facing behavior belongs in Basic Usage or Advanced Usage. Supported classes and functions belong in
the API reference. Contributor-only contracts, invariants, and extension checklists belong in Internals.

`just docs` is incremental. Before final review of a large documentation change, also run a forced Sphinx
build so deleted pages, stale references, and cached targets cannot hide a problem:

```shell
uv run sphinx-build -E -a -b html docs/source docs/build
```

Warnings should be fixed or documented as pre-existing with their exact source.
