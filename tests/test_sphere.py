import numpy as np

import pytest
from fiblat import sphere_lattice
from pytest_benchmark.fixture import BenchmarkFixture


def test_on_unit_sphere(benchmark: BenchmarkFixture) -> None:
    lattice = benchmark(sphere_lattice, 27, 1000)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


def test_on_unit_sphere_large(benchmark: BenchmarkFixture) -> None:
    lattice = benchmark(sphere_lattice, 600, 3)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


@pytest.mark.long
def test_on_large_case(benchmark: BenchmarkFixture) -> None:
    lattice = benchmark(sphere_lattice, 300, 30_000)
    norms = np.linalg.norm(lattice, 2, -1)
    assert np.allclose(norms, 1)


def test_evenly_distributed() -> None:
    lattice = sphere_lattice(27, 100)
    dists = 1 - (lattice @ lattice.T) + 2 * np.eye(100)
    min_dists = dists.min(-1)
    errors = np.sum(min_dists < 0.3)
    assert errors < 9


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        sphere_lattice(1, 2)
    with pytest.raises(ValueError):
        sphere_lattice(2, 0)
