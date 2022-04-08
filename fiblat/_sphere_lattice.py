from math import cos, pi, sin, exp, lgamma, log
from numba import njit
import numpy as np

from ._cube_lattice import cube_lattice


@njit
def int_sin_m(x: float, m: int) -> float:  # pragma: no cover
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    cosx = cos(x)
    sinx = sin(x)
    start = m % 2
    res = x if start == 0 else 1 - cosx
    for p in range(start + 1, m, 2):
        res = p / (p + 1) * res - cosx * sinx**p / (p + 1)
    return res


@njit
def inv_mult_int_sin_m(
    mult: float,
    m: int,
    target: float,
    lower: float = 0,
    upper: float = pi,
    atol: float = 1e-10,
) -> float:  # pragma: no cover
    """Returns func inverse of mult * integral of sin(x) ** m

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = mult * int_sin_m(mid, m)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = mult * int_sin_m(mid, m)
    return mid


_HLP = log(pi) / 2


@njit
def _sphere_projection(cube: np.ndarray) -> np.ndarray:  # pragma: no cover
    npoints, odim = cube.shape
    dim = odim + 1
    mults = np.empty((dim - 2,), np.float64)
    for d in range(2, dim):
        mults[d - 2] = exp(lgamma((d + 1) / 2) - lgamma(d / 2) - _HLP)

    output = np.ones((npoints, dim), np.float64)
    for i in range(npoints):
        points = output[i]
        base = cube[i]

        points[0] *= sin(2 * pi * base[0])
        points[1] *= cos(2 * pi * base[0])
        for d in range(2, dim):
            deg = inv_mult_int_sin_m(mults[d - 2], d - 1, base[d - 1])
            for j in range(d):
                points[j] *= sin(deg)
            points[d] *= cos(deg)
    return output


def sphere_lattice(dim: int, num_points: int) -> np.ndarray:
    """Generate num_points points over the dim - 1 dimensional hypersphere

    Generate a `num_points` length list of `dim`-dimensional tuples such the
    each element has an l2 norm of 1, and their nearest neighbor is roughly
    identical for each point.

    Parameters
    ----------
    dim : the dimension of points to sample, i.e. the length of tuples in the
        returned list
    num_points : the number of points to generate
    """
    if dim < 2:
        raise ValueError("dimension must be greater than one")
    if num_points < 1:
        raise ValueError("must request at least one point")
    return _sphere_projection(cube_lattice(dim - 1, num_points))
