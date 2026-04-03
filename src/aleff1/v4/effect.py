from asyncio import Lock
from contextvars import ContextVar
from functools import wraps, partial
from logging import getLogger
from typing import (
    Awaitable,
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


##
# type notations:
#   P: effect parameters
#   R: effect return type
#   V: handler return type
##

type Caller[V] = Callable[[], V]
type AsyncCaller[V] = Coroutine[Any, Any, V]
type EffectHandler[**P, V, R] = Callable[Concatenate[Resume[R], P], V]
type AsyncEffectHandler[**P, V, R] = Callable[Concatenate[Resume[R], P], AsyncCaller[V]]


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


class handler[V]:
    def __init__(self, *effects: Effect):
        self._token = self  # reset marker
        self._effects = tuple(effects)
        self._unbound_effects = set(effects)
        self._reserved_effects: list[tuple[Effect, Callable]] = []

    def on[**P, R](self, effect: Effect[P, R]) -> Callable[[EffectHandler[P, V, R]], Callable[P, V]]:
        resume: Resume[R] = Resume()

        def decorator(fn: EffectHandler[P, V, R]) -> Callable[P, V]:
            f = wraps(fn)(partial(fn, resume))
            self._reserved_effects.append((effect, f))
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

        # setup
        _put_handlers(self._token, self._reserved_effects)

        _logger.debug(f"handle {id(self)}")

        try:
            return fn()
        except _AbortSignal as ex:
            # ex :: _AbortSignal[V]
            return ex.value
        except _ResumeSignal as ex:
            raise RuntimeError("cannot resume outside of handler") from ex
        finally:
            _remove_all_handlers(self._token)


class async_handler[V]:
    def __init__(self, *effects: AsyncEffect):
        self._token = self  # reset marker
        self._effects = tuple(effects)
        self._unbound_effects = set(effects)
        self._reserved_effects: list[tuple[Effect, Callable]] = []

    def on[**P, R](
        self,
        effect: AsyncEffect[P, R],
    ) -> Callable[[AsyncEffectHandler[P, V, R]], Callable[P, Coroutine[Any, Any, V]]]:
        resume: Resume[R] = Resume()

        def decorator(fn: AsyncEffectHandler[P, V, R]) -> Callable[P, Coroutine[Any, Any, V]]:
            @wraps(fn)
            async def wrapper(*args: P.args, **kwargs: P.kwargs) -> V:
                return await fn(resume, *args, **kwargs)

            self._reserved_effects.append((effect, wrapper))
            return wrapper

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

    async def __call__(self, fn: Caller[Awaitable[V]], *, check=True) -> V:
        if check:
            self.check(fn)

        # setup
        await _put_handlers_async(self._token, self._reserved_effects)

        _logger.debug(f"handle {id(self)}")

        try:
            return await fn()
        except _AbortSignal as ex:
            # ex :: _AbortSignal[V]
            return ex.value
        except _ResumeSignal as ex:
            raise RuntimeError("cannot resume outside of handler") from ex
        finally:
            await _remove_all_handlers_async(self._token)


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
        _logger.debug(f"perform {self.name}")

        found = _get_item(self)
        if found is None:
            raise EffectNotHandledError(self)

        handler, fn = found

        try:
            v = fn(*args, **kwargs)
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            _logger.debug(f"perform {self.name} --> {e.value!r}")
            return e.value  # :: R

    def __str__(self) -> str:
        return f"<effect {self.name}>"


class _AsyncEffect[**P, R](_Effect):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        _logger.debug(f"perform {self.name}")

        found = _get_item(self)
        if found is None:
            raise EffectNotHandledError(self)

        handler, fn = found

        try:
            v = await fn(*args, **kwargs)
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            _logger.debug(f"perform {self.name} --> {e.value!r}")
            return e.value  # :: R


class _AbortSignal[V](BaseException):
    def __init__(self, handler: object, value: V):
        self.token = handler
        self.value = value


class _ResumeSignal[R](BaseException):
    def __init__(self, value: R):
        self.value = value


# (runner, effect, handler)
type _StackItem[**P, R, V] = tuple[object, Effect[P, R], Callable[P, V]]

_stack = ContextVar[list[_StackItem]]("handler_stask", default=[])
_lock = Lock()


def _get_stack[**P, R, V]() -> list[_StackItem[P, R, V]]:
    return _stack.get()


def _set_stack[**P, R, V](stack: list[_StackItem[P, R, V]]) -> None:
    _stack.set(stack)


def _get_item[**P, R, V](effect: Effect[P, R]) -> tuple[object, Callable[P, V]] | None:
    s: list[_StackItem] = _get_stack()
    for h, e, f in reversed(s):
        if e is effect:
            return h, f
    return None


def _put_handlers[**P, R, V](handler: handler[V], effects: list[tuple[Effect[P, R], Callable[P, V]]]) -> None:
    stask = _get_stack()
    for effect, fn in effects:
        _logger.debug(f"> {effect.name} | {id(handler)}")
        stask.append((handler, effect, fn))
    _set_stack(stask)


async def _put_handlers_async[**P, R, V](handler: object, effects: list[tuple[Effect[P, R], Callable[P, V]]]) -> None:
    _logger.debug(f"> {', '.join(e.name for e, _ in effects)} | {id(handler)}")
    async with _lock:
        stask = _get_stack()
        for effect, fn in effects:
            stask.append((handler, effect, fn))
        _set_stack(stask)


def _remove_all_handlers[**P, R, V](handler: object) -> None:
    _logger.debug(f"< | {id(handler)}")
    stack = [x for x in _get_stack() if x[0] is not handler]
    _set_stack(stack)


async def _remove_all_handlers_async[**P, R, V](handler: object) -> None:
    _logger.debug(f"< | {id(handler)}")
    async with _lock:
        stack = _get_stack()
        stack = [x for x in stack if x[0] is not handler]
        _set_stack(stack)
