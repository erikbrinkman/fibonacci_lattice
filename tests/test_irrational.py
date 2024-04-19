import numpy as np
from fiblat._irrational import n_primes


def test_n_primes() -> None:
    assert np.all(n_primes(0) == np.array([]))
    assert np.all(n_primes(1) == np.array([2]))
    assert np.all(n_primes(2) == np.array([2, 3]))
    assert np.all(n_primes(3) == np.array([2, 3, 5]))
    assert np.all(n_primes(4) == np.array([2, 3, 5, 7]))
    assert np.all(n_primes(5) == np.array([2, 3, 5, 7, 11]))
    assert np.all(n_primes(6) == np.array([2, 3, 5, 7, 11, 13]))
