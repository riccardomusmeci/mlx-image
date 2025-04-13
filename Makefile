SHELL = /bin/bash


.PHONY: all install clean check test full coverage
all:
	make clean
	make install

install:
    # Uncomment the following line if you want to run a prebuild script (must exist)
	pip install -e .[dev,test,docs]
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
	ruff check --diff .
	black --check --diff .
	mypy .

format:
	ruff check --show-fixes .
	black .
	mypy .

test:
	pytest tests/

full:
	make check
	make test

