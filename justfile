# https://github.com/casey/just

# List justfile recipes
help:
    just --list

# Update the environment with the latest version of the dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

# Update the environment but not with the development dependencies
prod-sync:
    uv sync --all-extras --no-dev --cache-dir .uv_cache

# Fix the code style and format
fix: dev-sync
    uv run ruff check --fix
    uv run ruff format
    uv run scripts/license_headers.py src --fix
    uv run scripts/license_headers.py tests --fix

# Execute the ruff code linter and format checker
check: dev-sync
    uv run ruff check

# Static type check: gate on NEW errors only, via the committed `.mypy-baseline.txt`.
# Plain pipe (no pipefail): mypy always exits non-zero while baseline errors remain, so
# the gate's pass/fail must come from `mypy-baseline filter`. A mypy crash is still caught
# (empty input -> filter reports all baseline entries resolved -> non-zero). Never add
# `--allow-unsynced` here: it is the one flag that defeats the ratchet.
typecheck: dev-sync
    uv run mypy | uv run mypy-baseline filter

# Regenerate the baseline after fixing (or deliberately accepting) type errors.
# This is the ONLY sanctioned way to write `.mypy-baseline.txt`; never hand-edit it.
typecheck-baseline: dev-sync
    uv run mypy | uv run mypy-baseline sync

# Show remaining type debt (files with the most errors) from the committed baseline.
typecheck-progress: dev-sync
    uv run mypy-baseline top-files

# Execute all unit tests
test: dev-sync
    uv run -m unittest discover -v -s tests

# Build the docs
docs: dev-sync
    uv run sphinx-build -b html docs/source docs/build

# Build the release package
build:
    rm -rf dist
    uv build --wheel
    uv build --sdist
