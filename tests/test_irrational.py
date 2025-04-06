"""Test irational module."""

import numpy as np

from fiblat._irrational import n_primes


def test_n_primes() -> None:
    """Test that the n_primes function returns the first n primes."""
    primes = np.array([2, 3, 5, 7, 11, 13], "i8")
    for i in range(7):
        assert np.all(n_primes(i) == primes[:i])
