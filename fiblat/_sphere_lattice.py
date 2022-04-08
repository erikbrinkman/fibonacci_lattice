from itertools import count, islice
from math import cos, gamma, pi, sin, sqrt, exp, lgamma, log
from typing import Callable, Iterator, List, Optional, Sequence, Tuple
from numba import jit

from ._cube_lattice import cube_lattice


@jit(nopython=True)
def int_sin_m(x: float, m: int) -> float:  # pragma: no cover
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    cosx = cos(x)
    sinx = sin(x)
    start = m % 2
    res = x if start == 0 else 1 - cosx
    for p in range(start + 1, m, 2):
        res = p / (p + 1) * res - cosx * sinx**p / (p + 1)
    return res


@jit(nopython=True)
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


def cube_to_sphere(cube: Sequence[Sequence[float]]) -> Iterator[Tuple[float, ...]]:
    """Map points from [0, 1]^dim to the sphere

    Maps points in [0, 1]^dim to the surface of the sphere; dim + 1 dimensional
    points with unit l2 norms. This mapping preserves relative distance between
    points.

    Parameters
    ----------
    cube : a sequence points in the [0, 1]^dim hyper cube
    """
    dims = {len(p) + 1 for p in cube}
    assert len(dims) == 1, "not all points had the same dimension"
    (dim,) = dims

    output = [[1.0 for _ in range(dim)] for _ in cube]
    mults = [exp(lgamma(d / 2 + 0.5) - lgamma(d / 2) - _HLP) for d in range(2, dim)]
    for base in cube:
        points = [1.0 for _ in range(dim)]
        points[0] *= sin(2 * pi * base[0])
        points[1] *= cos(2 * pi * base[0])

        for d, (mult, lat) in enumerate(zip(mults, base[1:]), 2):
            deg = inv_mult_int_sin_m(mult, d - 1, lat)
            for j in range(d):
                points[j] *= sin(deg)
            points[d] *= cos(deg)
        yield tuple(points)


def sphere_lattice(dim: int, num_points: int) -> List[Tuple[float, ...]]:
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
    return list(cube_to_sphere(cube_lattice(dim - 1, num_points)))
