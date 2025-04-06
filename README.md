# Fibonacci Lattice

[![pypi](https://img.shields.io/pypi/v/fiblat)](https://pypi.org/project/fiblat/)
[![build](https://github.com/erikbrinkman/fibonacci_lattice/actions/workflows/build.yml/badge.svg)](https://github.com/erikbrinkman/fibonacci_lattice/actions/workflows/build.yml)
[![docs](https://img.shields.io/badge/api-docs-blue)](https://erikbrinkman.github.io/fibonacci_lattice/)

A simple small python package for generating uniform points on the sphere.
This module provides to functions `fiblat.cube_lattice` and `fiblat.sphere_lattice`.
Both functions take a dimension and a number of points and return numpy arrays that are roughly evenly spaced in either the `[0, 1]` hypercube or the unit hypersphere.

## Installation

```bash
pip install fiblat
```

## Usage

```python
from fiblat import sphere_lattice, cube_lattice

cube = cube_lattice(3, 100)
sphere = sphere_lattice(3, 100)
```

## Development

```sh
uv run ruff format --check
uv run ruff check
uv run pyright
uv run pytest
```

## Publishing

```sh
rm -rf dist
uv build
uv publish --username __token__
```
