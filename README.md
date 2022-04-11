Fibonacci Lattice
=================
[![pypi](https://img.shields.io/pypi/v/fiblat)](https://pypi.org/project/fiblat/)
[![build](https://github.com/erikbrinkman/fibonacci_lattice/actions/workflows/build.yml/badge.svg)](https://github.com/erikbrinkman/fibonacci_lattice/actions/workflows/build.yml)

A simple small python package for generating uniform points on the sphere.
This module provides to functions `fiblat.cube_lattice` and `fiblat.sphere_lattice`.
Both functions take a dimension and a number of points and return numpy arrays that are roughly evenly spaced in either the `[0, 1]` hypercube or the unit hypersphere.

Installation
------------

```bash
pip install fiblat
```

Usage
-----

```python
from fiblat import sphere_lattice, cube_lattice

cube = cube_lattice(3, 100)
sphere = sphere_lattice(3, 100)
```

Development
-----------

- setup: `poetry install`
- tests: `poetry run pytest && poetry run pyre`
