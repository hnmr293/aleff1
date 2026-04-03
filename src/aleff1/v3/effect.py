from functools import wraps, partial
import threading
from logging import getLogger
from typing import Callable, cast, Concatenate, NoReturn, overload, Protocol, runtime_checkable

##
# type notations:
#   P: effect parameters
#   R: effect return type
#   V: handler return type
##

type Caller[V] = Callable[[], V]

type EffectHandler[**P, V, R] = Callable[Concatenate[Resume[R], P], V]


@runtime_checkable
class Effect[**P, R](Protocol):
    @property
    def name(self) -> str: ...

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R: ...


@overload
def effect[**P, R](name: str) -> Effect[P, R]: ...


@overload
def effect[**P, R, V](effect: Effect[P, R]) -> Callable[[Caller[V]], Caller[V]]: ...


@overload
def effect[V](*effects: Effect) -> Callable[[Caller[V]], Caller[V]]: ...


def effect[**P, R, V](*args: str | Effect[P, R]) -> Effect[P, R] | Callable[[Caller[V]], Caller[V]]:
    if len(args) == 0:
        raise TypeError("at least one argument is required")

    if isinstance(args[0], str):
        return _Effect(args[0])

    if not all(isinstance(arg, Effect) for arg in args):
        raise TypeError("all arguments must be Effect")

    args = cast(tuple[Effect], args)
    decorator = _create_effect_annotator(args)
    return decorator


def _create_effect_annotator[V](effects: tuple[Effect]) -> Callable[[Caller[V]], Caller[V]]:
    def decorator(fn: Caller[V]) -> Caller[V]:
        return wraps(fn)(fn)

    return decorator


class Resume[R]:
    def __call__(self, value: R) -> NoReturn:
        raise _ResumeSignal(value)


class handler[V]:
    def __init__(self, *effects: Effect):
        self._token = self  # reset marker
        self._effects = tuple(effects)
        self._unbound_effects = set(effects)

    def on[**P, R](self, effect: Effect[P, R]) -> Callable[[EffectHandler[P, V, R]], Callable[P, V]]:
        resume: Resume[R] = Resume()

        def decorator(fn: EffectHandler[P, V, R]) -> Callable[P, V]:
            f = wraps(fn)(partial(fn, resume))
            _put_item(self._token, effect, f)
            return f

        # raises an error if the same effect is handled multiple times
        if effect not in self._effects:
            raise ValueError(f"{effect} is not declared in the handler")
        if effect not in self._unbound_effects:
            raise ValueError(f"{effect} is already handled")
        self._unbound_effects.remove(effect)

        return decorator

    def check(self, fn: Caller[V]) -> None:
        if len(self._unbound_effects) > 0:
            unboud_effects = ", ".join(str(e) for e in self._unbound_effects)
            raise ValueError(f"not all effects are handled: {unboud_effects}")

    def __call__(self, fn: Caller[V], *, check=True) -> V:
        if check:
            self.check(fn)

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
    def __init__(self, effect: Effect):
        super().__init__(f"no handler for the effect: {effect}")
        self.effect = effect


class _Effect[**P, R](Effect):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R:
        found = _get_item(self)
        if found is None:
            raise EffectNotHandledError(self)

        handler, fn = found

        try:
            v = fn(*args, **kwargs)
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            return e.value  # :: R

    def __str__(self) -> str:
        return f"<effect {self.name}>"


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


def _get_item[**P, R, V](effect: Effect[P, R]) -> tuple[handler[V], Callable[P, V]] | None:
    s: list[_StackItem[P, R, V]] = _get_stack()
    for h, e, f in reversed(s):
        if e is effect:
            return h, f
    return None


def _put_item[**P, R, V](handler: handler[V], effect: Effect[P, R], fn: Callable[P, V]) -> None:
    _logger.debug(f"> {effect.name} | {id(handler)}")
    _get_stack().append((handler, effect, fn))


def _remove_item[**P, R, V](handler: handler[V]) -> None:
    _logger.debug(f"< | {id(handler)}")
    s = [x for x in _get_stack() if x[0] is not handler]
    _local.s = s
