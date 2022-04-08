from statistics import stdev, mean, median
from math import sqrt
from itertools import chain
from typing import Tuple

import pytest
from fiblat import cube_lattice


def dist(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    val = 0.0
    for lv, rv in zip(left, right):
        val += (lv - rv) ** 2
    return sqrt(val)


def test_one_dim() -> None:
    one = cube_lattice(1, 4)
    for ind, (actual,) in enumerate(one):
        assert ind / 4 == actual


def test_in_unit_cube() -> None:
    lattice = cube_lattice(27, 1000)
    for point in lattice:
        for val in point:
            assert 0 <= val <= 1


def test_evenly_distributed() -> None:
    lattice = cube_lattice(27, 100)
    min_dists = [
        min(dist(point, other) for other in chain(lattice[:i], lattice[i + 1 :]))
        for i, point in enumerate(lattice)
    ]
    errors = sum(d > 1.8 for d in min_dists)
    assert errors < 2


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        cube_lattice(0, 2)
    with pytest.raises(ValueError):
        cube_lattice(2, 0)
