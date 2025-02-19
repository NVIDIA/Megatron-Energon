# https://github.com/casey/just

dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

prod-sync:
	uv sync --all-extras --no-dev --cache-dir .uv_cache

format:
	uv run ruff format

check:
	uv run ruff check

test:
	uv run -m unittest discover -s tests

docs:
	uv run sphinx-build -b html docs/source docs/build

build:
    rm -rf dist
    uv build
