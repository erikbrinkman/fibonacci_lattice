from itertools import chain
from typing import Tuple

import pytest
from fiblat import sphere_lattice


def dist(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    val = 1.0
    for lv, rv in zip(left, right):
        val -= lv * rv
    return max(val, 0)


def test_on_unit_cube() -> None:
    lattice = sphere_lattice(17, 1000)
    for point in lattice:
        diff_norm = 1.0
        for val in point:
            diff_norm -= val**2
        assert abs(diff_norm) < 1e-6


def test_on_unit_cube_large() -> None:
    lattice = sphere_lattice(600, 3)
    for point in lattice:
        diff_norm = 1.0
        for val in point:
            diff_norm -= val**2
        assert abs(diff_norm) < 1e-6


def test_evenly_distributed() -> None:
    lattice = sphere_lattice(27, 100)
    min_dists = [
        min(dist(point, other) for other in chain(lattice[:i], lattice[i + 1 :]))
        for i, point in enumerate(lattice)
    ]
    errors = sum(d > 0.7 for d in min_dists)
    assert errors < 1


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        sphere_lattice(1, 2)
    with pytest.raises(ValueError):
        sphere_lattice(2, 0)
