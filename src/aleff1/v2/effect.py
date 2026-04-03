from functools import wraps, partial
import threading
from logging import getLogger
from typing import Callable, Concatenate, NoReturn

##
# type notations:
#   P: effect parameters
#   R: effect return type
#   V: handler return type
##


class Effect[**P, R]:
    def __init__(self, name: str):
        self.name = name

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R:
        handler, fn = _get_item(self)

        try:
            v = fn(*args, **kwargs)
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            return e.value  # :: R


class Resume[R]:
    def __call__(self, value: R) -> NoReturn:
        raise _ResumeSignal(value)


class handler[V]:
    def __init__(self):
        self._token = self  # reset marker

    def on[**P, R](self, effect: Effect[P, R]) -> Callable[["_EffectHandler[P, V, R]"], Callable[P, V]]:
        resume: Resume[R] = Resume()

        def decorator(fn: _EffectHandler[P, V, R]) -> Callable[P, V]:
            f = wraps(fn)(partial(fn, resume))
            _put_item(self._token, effect, f)
            return f

        return decorator

    def __call__(self, fn: Callable[[], V]) -> V:
        try:
            return fn()
        except _AbortSignal as ex:
            # ex :: _AbortSignal[V]
            return ex.value
        except _ResumeSignal as ex:
            raise RuntimeError("cannot resume outside of handler") from ex
        finally:
            _remove_item(self._token)


class EffectNotHandledError(RuntimeError):
    pass


type _EffectHandler[**P, V, R] = Callable[Concatenate[Resume[R], P], V]


class _AbortSignal[V](BaseException):
    def __init__(self, handler: handler[V], value: V):
        self.token = handler
        self.value = value


class _ResumeSignal[R](BaseException):
    def __init__(self, value: R):
        self.value = value


_logger = getLogger(__name__)


_local = threading.local()

# (runner, effect, handler)
type _StackItem[**P, R, V] = tuple[handler[V], Effect[P, R], Callable[P, V]]


def _get_stack[**P, R, V]() -> list[_StackItem[P, R, V]]:
    if not hasattr(_local, "s"):
        _local.s = []
    return _local.s


def _get_item[**P, R, V](effect: Effect[P, R]) -> tuple[handler[V], Callable[P, V]]:
    s: list[_StackItem[P, R, V]] = _get_stack()
    for h, e, f in reversed(s):
        if e is effect:
            return h, f
    raise EffectNotHandledError(f"no handler for {effect.name}")


def _put_item[**P, R, V](handler: handler[V], effect: Effect[P, R], fn: Callable[P, V]) -> None:
    _logger.debug(f"> {effect.name} | {id(handler)}")
    _get_stack().append((handler, effect, fn))


def _remove_item[**P, R, V](handler: handler[V]) -> None:
    _logger.debug(f"< | {id(handler)}")
    s = [x for x in _get_stack() if x[0] is not handler]
    _local.s = s
