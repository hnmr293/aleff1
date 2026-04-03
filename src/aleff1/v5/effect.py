from asyncio import Lock
from contextvars import ContextVar
from functools import wraps, partial
from logging import getLogger
from typing import (
    Callable,
    cast,
    Concatenate,
    Coroutine,
    NoReturn,
    overload,
    Protocol,
    runtime_checkable,
    Any,
)

_logger = getLogger(__name__)


def loglevel(level: int):
    _logger.setLevel(level)


def _debug(msg: str):
    _logger.debug(msg)


##
# type notations:
#   P: effect parameters
#   R: effect return type
#   V: handler return type
##

type Caller[V] = Callable[[], V]
type AsyncCaller[V] = Caller[Coroutine[Any, Any, V]]
type EffectHandler[**P, V, R] = Callable[Concatenate[Resume[R], P], V]
type AsyncEffectHandler[**P, V, R] = EffectHandler[P, Coroutine[Any, Any, V], R]
type PartialHandler[**P, V] = Callable[P, V]
type AsyncPartialHandler[**P, V] = PartialHandler[P, Coroutine[Any, Any, V]]


@runtime_checkable
class Effect[**P, R](Protocol):
    @property
    def name(self) -> str: ...

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R: ...


@runtime_checkable
class AsyncEffect[**P, R](Protocol):
    @property
    def name(self) -> str: ...

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> Coroutine[Any, Any, R]: ...


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


def async_effect[**P, R](name: str) -> AsyncEffect[P, R]:
    return _AsyncEffect(name)


def _create_effect_annotator[V](effects: tuple[Effect]) -> Callable[[Caller[V]], Caller[V]]:
    def decorator(fn: Caller[V]) -> Caller[V]:
        return wraps(fn)(fn)

    return decorator


class Resume[R]:
    def __call__(self, value: R) -> NoReturn:
        raise _ResumeSignal(value)

    @overload
    def apply[**P, V](self, fn: EffectHandler[P, V, R]) -> PartialHandler[P, V]: ...

    @overload
    def apply[**P, V](self, fn: AsyncEffectHandler[P, V, R]) -> AsyncPartialHandler[P, V]: ...

    def apply[**P, V](
        self,
        fn: EffectHandler[P, V, R] | AsyncEffectHandler[P, V, R],
    ) -> PartialHandler[P, V] | AsyncPartialHandler[P, V]:
        reduced = partial(fn, self)
        return wraps(fn)(reduced)


_num = ContextVar[int]("handler_count", default=0)


class _handler_base[Caller: Callable, Eff: Effect | AsyncEffect, EffectHandler: Callable, PartialHandler: Callable]:
    def __init__(self, *effects: Eff):
        self._token = self  # reset marker
        self._effects = tuple(effects)
        self._unbound_effects = set(effects)
        self._reserved_effects: list[tuple[Eff, PartialHandler]] = []
        self._n = _num.get()
        _num.set(self._n + 1)
        _debug(f"@ {self}")

    # actually on[**P, R, V](self, effect: Effect[P, R]) -> Callable[[EffectHandler[P, V, R]], PartialHandler[P, V]]
    # unfortunately python does not support higher-kinded types, so we cannot express it directly
    def on[R](self, effect: Eff) -> Callable[[EffectHandler], PartialHandler]:
        _debug(f"|> {effect} | {self}")

        resume: Resume[R] = Resume()

        def decorator(fn: EffectHandler) -> PartialHandler:
            f: PartialHandler = resume.apply(fn)  # type: ignore
            self._reserved_effects.append((effect, f))
            return f

        # raises an error if the same effect is handled multiple times
        if effect not in self._effects:
            raise ValueError(f"{effect} is not declared in the handler")
        if effect not in self._unbound_effects:
            raise ValueError(f"{effect} is already handled")
        self._unbound_effects.remove(effect)

        return decorator

    def check(self, fn: Caller) -> None:
        if len(self._unbound_effects) > 0:
            unboud_effects = ", ".join(str(e) for e in self._unbound_effects)
            raise ValueError(f"not all effects are handled: {unboud_effects}")

    def __str__(self) -> str:
        return f"<handler #{self._n} | {', '.join(e.name for e in self._effects)}>"


# python does not support higher-kinded types...
class handler[V](
    _handler_base[
        Caller[V],
        Effect[Any, Any],
        EffectHandler[Any, Any, Any],
        PartialHandler[Any, Any],
    ]
):
    def __call__(self, fn: Caller[V], *, check=True) -> V:
        _debug(f"|@ {self}")

        if check:
            self.check(fn)

        # setup
        _put_handlers(self._token, self._reserved_effects)

        try:
            v = fn()
            _debug(f"|< {v!r} | {self}")
            return v
        except _AbortSignal as ex:
            # ex :: _AbortSignal[V]
            _debug(f"|<! **abort** {ex.value!r} | {self}")
            return ex.value
        except _ResumeSignal as ex:
            raise RuntimeError("cannot resume outside of handler") from ex
        finally:
            _remove_all_handlers(self._token)


class async_handler[V](
    _handler_base[
        AsyncCaller[V],
        AsyncEffect[Any, Any],
        AsyncEffectHandler[Any, Any, Any],
        AsyncPartialHandler[Any, Any],
    ]
):
    async def __call__(self, fn: AsyncCaller[V], *, check=True) -> V:
        _debug(f"|@ {self}")

        if check:
            self.check(fn)

        # setup
        await _put_handlers_async(self._token, self._reserved_effects)

        try:
            v = await fn()
            _debug(f"|< {v!r} | {self}")
            return v
        except _AbortSignal as ex:
            # ex :: _AbortSignal[V]
            _debug(f"|<! **abort** {ex.value!r} | {self}")
            return ex.value
        except _ResumeSignal as ex:
            raise RuntimeError("cannot resume outside of handler") from ex
        finally:
            await _remove_all_handlers_async(self._token)


class EffectNotHandledError(RuntimeError):
    def __init__(self, effect: Effect):
        super().__init__(f"no handler for the effect: {effect}")
        self.effect = effect


class _EffectBase[**P, R](Effect):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _find[V](self) -> tuple[object, Callable[P, V]]:
        found = _get_item(self)
        if found is None:
            raise EffectNotHandledError(self)

        return found


class _Effect[**P, R](_EffectBase):
    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R:
        _debug(f"||> perform {self}")

        handler, fn = self._find()

        _debug(f"||>     found handler {handler}")

        try:
            v = fn(*args, **kwargs)
            _debug(f"||< **abort** perform {self} = {v!r}")
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            _debug(f"||< perform {self} = {e.value!r}")
            return e.value  # :: R

    def __str__(self) -> str:
        return f"<effect {self.name}>"


class _AsyncEffect[**P, R](_Effect):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        _debug(f"||> perform {self}")

        handler, fn = self._find()

        _debug(f"||>     found handler {handler}")

        try:
            v = await fn(*args, **kwargs)
            _debug(f"||< **abort** perform {self} = {v!r}")
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            _debug(f"||< perform {self} = {e.value!r}")
            return e.value  # :: R


class _AbortSignal[V](BaseException):
    def __init__(self, handler: object, value: V):
        self.token = handler
        self.value = value


class _ResumeSignal[R](BaseException):
    def __init__(self, value: R):
        self.value = value


# (runner, effect, handler)
type _StackItem[V] = tuple[_handler_base, Effect | AsyncEffect, Callable[..., V]]

_stack = ContextVar[list[_StackItem]]("handler_stask", default=[])
_lock = Lock()


def _get_stack[V]() -> list[_StackItem[V]]:
    return _stack.get()


def _set_stack[V](stack: list[_StackItem[V]]) -> None:
    _stack.set(stack)


def _get_item[**P, R, V](effect: Effect[P, R]) -> tuple[_handler_base, Callable[..., V]] | None:
    s = _get_stack()
    for h, e, f in reversed(s):
        if e is effect:
            return h, f
    return None


def _put_handlers(
    handler: _handler_base,
    effects: list[tuple[Effect | AsyncEffect, Callable]],
) -> None:
    stask = _get_stack()
    for effect, fn in effects:
        stask.append((handler, effect, fn))
    _set_stack(stask)


async def _put_handlers_async(
    handler: _handler_base,
    effects: list[tuple[Effect | AsyncEffect, Callable]],
) -> None:
    async with _lock:
        stask = _get_stack()
        for effect, fn in effects:
            stask.append((handler, effect, fn))
        _set_stack(stask)


def _remove_all_handlers[**P, R, V](handler: _handler_base) -> None:
    stack = [x for x in _get_stack() if x[0] is not handler]
    _set_stack(stack)


async def _remove_all_handlers_async[**P, R, V](handler: _handler_base) -> None:
    async with _lock:
        stack = _get_stack()
        stack = [x for x in stack if x[0] is not handler]
        _set_stack(stack)
