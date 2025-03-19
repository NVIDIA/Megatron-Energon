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

# Execute the ruff code linter and format checker
check: dev-sync
    uv run ruff check

# Execute all unit tests
test: dev-sync
    uv run -m unittest discover -s tests

# Build the docs
docs: dev-sync
    uv run sphinx-build -b html docs/source docs/build

# Build the release package
build: dev-sync
    rm -rf dist
    uv build --wheel
    uv build --sdist
