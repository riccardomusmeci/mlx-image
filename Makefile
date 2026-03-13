SHELL = /bin/bash


.PHONY: all install clean check test full coverage format

all:
	make clean
	make install

install:
	uv sync --all-extras
	make full

clean:
	-rm -rf .mypy_cache
	-rm -rf .pytest_cache
	-rm -rf .eggs
	-rm -rf dist
	-rm -rf *.egg-info
	-find . -not -path "./.git/*" -name logs -exec rm -rf {} \;
	-rm -rf logs
	-rm -rf tests/logs
	-rm -rf .ruff_cache
	-find tests -depth -type d -empty -delete

check:
	uv run ruff check --diff .
	uv run ruff format --check --diff .
	uv run ty check src/

format:
	uv run ruff check --show-fixes .
	uv run ruff format .

test:
	uv run pytest tests/

full:
	make check
	make test

coverage:
	uv run pytest tests/ --cov=src/mlxim --cov-report=term-missing
