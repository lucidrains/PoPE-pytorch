from functools import wraps
from typing import Callable, ParamSpec, TypeVar


T = TypeVar('T')
P = ParamSpec('P')


def default(v: T, d: T) -> T:
    return v if exists(v) else d

def divisible_by(num: int | float, den: int | float) -> bool:
    return den != 0 and ((num % den) == 0)

def exists(v: object | None) -> bool:
    return v is not None

def once(fn: Callable[P, T]) -> Callable[P, T | None]:
    called: bool = False

    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> T | None:
        nonlocal called
        if called:
            return
        called = True
        return fn(*args, **kwargs)

    return inner

print_once= once(print)

