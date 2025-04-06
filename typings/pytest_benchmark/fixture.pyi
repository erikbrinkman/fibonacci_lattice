from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

class BenchmarkFixture:
    def __call__(
        self, func: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
    ) -> R: ...
