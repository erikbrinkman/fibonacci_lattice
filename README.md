Fibonacci Lattice
=================

A simple small python package for generating uniform points on the sphere.
This module provides to functions `fiblat.cube_lattice` and `fiblat.sphere_lattice`.
Both functions take a dimension and a number of points and return a list of vectors that are roughly evenly spaced in either the `[0, 1]` hypercube or the unit hypersphere.

Development
-----------

- setup: `poetry install`
- tests: `poetry run pytest && poetry run pyre`
