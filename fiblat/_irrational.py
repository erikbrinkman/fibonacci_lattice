import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.jit(nb.int64[:](nb.int64), cache=True, nogil=True)
def n_primes(n: int) -> NDArray[np.float64]:  # pragma: nocover
    """Create an array of the first n primes."""
    res = np.empty(n, "i8")
    if n <= 0:
        return res
    res[0] = 2
    if n == 1:
        return res
    # this is an upper bound on the highest number the nth prime could be
    max_num = max(int(n * np.exp(np.sqrt(2 * np.log(n) - 2)) + 1), 10)
    seive = np.ones((max_num - 4) // 2, "?")
    j = 1
    for i in range(3, max_num + 1, 2):
        if seive[(i - 3) // 2]:
            res[j] = i
            j += 1
            if j == n:
                return res
            else:
                seive[(i**2 - 3) // 2 :: i] = False
    raise ValueError("inaccurate bound")
