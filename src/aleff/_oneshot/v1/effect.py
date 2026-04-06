import threading
from logging import getLogger
from typing import Callable, NoReturn, cast


class EffectNotHandledError(RuntimeError):
    pass


class Effect[**P, R]:
    def __init__(self, name: str):
        self.name = name

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R:
        handler, token = _get_item(self)

        try:
            v = handler(*args, **kwargs)
            raise _AbortSignal(token, v)
        except _ResumeSignal as e:
            return e.value


def resume[R](value: R) -> NoReturn:
    # shift
    raise _ResumeSignal(value)


class handler[V]:
    result: None | V

    def __init__(self):
        self._token = None

    def __enter__(self) -> "handler[V]":
        # reset
        self._token = object()  # reset marker
        return self

    def __call__[**P, R](self, effect: Effect[P, R]) -> Callable[[Callable[P, V]], Callable[P, V]]:
        if self._token is None:
            raise RuntimeError(f"{self.__class__.__qualname__} must be used as a context manager")

        def decorator(fn: Callable[P, V]) -> Callable[P, V]:
            _put_item(effect, fn, self._token)
            return fn

        return decorator

    def __exit__(self, ty: type[BaseException] | None, ex: BaseException | None, tb):
        if self._token is None:
            raise RuntimeError(f"{self.__class__.__qualname__} must be used as a context manager")

        _remove_item(self._token)
        if isinstance(ex, _AbortSignal) and ex.token is self._token:
            self.result = cast(V, ex.value)
            return True


class _AbortSignal[V](BaseException):
    def __init__(self, token: object, value: V):
        self.token = token
        self.value = value


class _ResumeSignal[R](BaseException):
    def __init__(self, value: R):
        self.value = value


_logger = getLogger(__name__)


_local = threading.local()

type _StackItem[**P, R, V] = tuple[Effect[P, R], Callable[P, V], object]


def _get_stack[**P, R, V]() -> list[_StackItem[P, R, V]]:
    if not hasattr(_local, "s"):
        _local.s = []
    return _local.s


def _get_item[**P, R, V](effect: Effect[P, R]) -> tuple[Callable[P, V], object]:
    s = _get_stack()
    for tag, handler, token in reversed(s):
        if tag is effect:
            return handler, token
    raise EffectNotHandledError(f"no handler for {effect.name}")


def _put_item[**P, R, V](effect: Effect[P, R], handler: Callable[P, V], token: object) -> None:
    _logger.debug(f"> {effect.name} with token {token}")
    _get_stack().append((effect, handler, token))


def _remove_item[**P, R, V](token: object) -> None:
    _logger.debug(f"< token {token}")
    s = [x for x in _get_stack() if x[2] is not token]
    _local.s = s
