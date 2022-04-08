import numpy as np

import pytest
from fiblat import sphere_lattice


def test_on_unit_sphere() -> None:
    lattice = sphere_lattice(17, 1000)
    norms = np.sum(lattice**2, -1)
    assert np.allclose(norms, 1)


def test_on_unit_sphere_large() -> None:
    lattice = sphere_lattice(600, 3)
    norms = np.sum(lattice**2, -1)
    assert np.allclose(norms, 1)


def test_evenly_distributed() -> None:
    lattice = sphere_lattice(27, 100)
    dists = 1 - (lattice @ lattice.T) + 2 * np.eye(100)
    min_dists = dists.min(-1)
    assert np.all(min_dists < 0.7)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        sphere_lattice(1, 2)
    with pytest.raises(ValueError):
        sphere_lattice(2, 0)
