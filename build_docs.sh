#!/usr/bin/env bash
set -e

# Clean
rm -rf docs/_build docs/_generated

uv run sphinx-build -b html docs docs/_build/html
