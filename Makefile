SHELL = /bin/bash

pkg_name = mlx-im
src_pkg = src/$(pkg_name)

.PHONY: all install clean check test full coverage
all:
	make clean
	make install

install:
    # Uncomment the following line if you want to run a prebuild script (must exist)
	pip install -e .[dev,test,docs]
	make full

clean:
	-rm -rf htmlcov
	-rm -rf .benchmarks
	-rm -rf .mypy_cache
	-rm -rf .pytest_cache
	-rm -rf docs/_build
	-rm -rf docs/source/_autosummary
	-rm -rf prof
	-rm -rf build
	-rm -rf .eggs
	-rm -rf dist
	-rm -rf *.egg-info
	-find . -not -path "./.git/*" -name logs -exec rm -rf {} \;
	-rm -rf logs
	-rm -rf tests/logs
	-rm .coverage
	-rm -rf .ruff_cache
	-find . -not -path "./.git/*" -name '.benchmarks' -exec rm -rf {} \;
	-find tests -depth -type d -empty -delete
check:
	ruff check --diff .
	black --check --diff .
	docformatter --check --diff src tests
	mypy .

format:
	ruff check --show-fixes .
	black .
	docformatter --in-place src tests
	mypy .

test:
	python -m pytest --cov=$(src_pkg) --cov-branch --cov-report=term-missing --cov-fail-under=$(coverage_percentage) tests

full:
	make check
	make test

coverage:
	make test
	coverage report -m
	coverage html
	@echo "results are in ./htmlcov/index.html"
