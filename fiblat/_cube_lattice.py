from itertools import chain, islice, count
from typing import Iterator, List, Optional, Tuple

from ._irrational import root_primes


def cube_lattice(
    dim: int, num_points: int, *, irrationals: Optional[Iterator[float]] = None
) -> List[Tuple[float, ...]]:
    """Generate num_points points over the dim dimensional cube

    Generates `num_points` roughly evenly from the `[0, 1]^dim`.

    Parameters
    ----------
    dim : dimension of cube to generate points in
    num_points : the number of points to generate
    irrationals : an iterator of at least `dim-1` irrational numbers, if
        omitted the square roots of the primes are used
    """
    if dim < 1:
        raise ValueError("dimension must be greater than zero")
    if num_points < 1:
        raise ValueError("must request at least one point")
    mult_iter = root_primes() if irrationals is None else irrationals
    mults = tuple(islice(chain([1 / num_points], mult_iter), dim))
    return [tuple(i * m % 1.0 for m in mults) for i in range(num_points)]
