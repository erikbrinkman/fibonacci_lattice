import numpy as np

import pytest
from fiblat import cube_lattice


def test_one_dim() -> None:
    one = cube_lattice(1, 4)
    assert np.allclose(one, np.arange(4)[:, None] / 4)


def test_in_unit_cube() -> None:
    lattice = cube_lattice(27, 1000)
    assert np.all((0 <= lattice) & (lattice <= 1))


def test_evenly_distributed() -> None:
    lattice = cube_lattice(27, 100)
    dists = np.linalg.norm(lattice[:, None] - lattice, 2, -1) + 27 * np.eye(100)
    min_dists = dists.min(-1)
    errors = np.sum(min_dists > 1.8)
    assert errors < 2


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        cube_lattice(0, 2)
    with pytest.raises(ValueError):
        cube_lattice(2, 0)
