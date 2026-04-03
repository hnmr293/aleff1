from dataclasses import dataclass, field
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
    overload,
    Protocol,
    runtime_checkable,
    Any,
)
import greenlet as gl

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
type AsyncCaller[V] = Coroutine[Any, Any, V]
type EffectHandler[**P, V, R] = Callable[Concatenate[Resume[R, V], P], V]
type AsyncEffectHandler[**P, V, R] = Callable[Concatenate[Resume[R, V], P], AsyncCaller[V]]


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
def effect(name: str, /) -> Effect[..., Any]: ...


@overload
def effect[V](*effects: Effect[..., Any]) -> Callable[[Caller[V]], Caller[V]]: ...


def effect[**P, R, V](*args: str | Effect[P, R]) -> Effect[..., Any] | Callable[[Caller[V]], Caller[V]]:
    if len(args) == 0:
        raise TypeError("at least one argument is required")

    if isinstance(args[0], str):
        return _Effect[..., Any](args[0])

    if not all(isinstance(arg, Effect) for arg in args):
        raise TypeError("all arguments must be Effect")

    args = cast(tuple[Effect[..., Any]], args)
    decorator = _create_effect_annotator(args)
    return decorator


def async_effect(name: str, /) -> AsyncEffect[..., Any]:
    return _AsyncEffect(name)


def _create_effect_annotator[V](effects: tuple[Effect[..., Any]]) -> Callable[[Caller[V]], Caller[V]]:
    def decorator(fn: Caller[V]) -> Caller[V]:
        return wraps(fn)(fn)

    return decorator


##
# greenlet-based implementation
#
# handler_gl --|caller_gl.switch()|-> caller_gl
#   effect --|handler_gl.switch(eff, args)|-> handler_gl
#     resume v --|caller_gl.switch(eff, v)|-> caller_gl
#     return v --> *finish* v
##


@dataclass(frozen=True)
class EffectContext[**P, R]:
    effect: Effect[P, R]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict[str, Any])

    def __repr__(self) -> str:
        return f"({self.effect} | args={self.args!r}, kwargs={self.kwargs!r})"


class Resume[R, V]:
    def __call__(self, value: R) -> V:
        caller_gl = _get_caller()

        _debug("||> @caller")

        v = caller_gl.switch(value)

        _debug(f"||< @caller = {v!r}")

        return _drive(caller_gl, v)


def _drive[V](caller_gl: Any, value: V | EffectContext[..., Any]) -> V:
    _debug("||> @main")

    if caller_gl.dead:
        # computation finished
        _debug(f"||= return = {value!r}")
        _debug("||< @main")
        assert not isinstance(value, EffectContext)
        return value

    # effect performed
    # switch to the handler greenlet

    if not isinstance(value, EffectContext):
        raise RuntimeError(f"invalid value passed to caller: {value!r}")

    effect: Effect[..., Any]
    effect, args, kwargs = value.effect, value.args, value.kwargs  # type: ignore

    found = _get_item(effect)
    if found is None:
        raise EffectNotHandledError(effect)

    handler, fn = found

    _debug(f"||> ... found handler {handler} | {fn.__name__}")

    _set_caller(caller_gl)
    v = fn(*args, **kwargs)

    if not caller_gl.dead:
        # resume not called in the handler
        # discard the result
        _debug(f"||< **abort** perform {effect} = {v!r}")
        caller_gl.throw(gl.GreenletExit)

    _debug(f"||< perform {effect} = {v!r}")

    _debug("||< @main")
    return v


_num = 0


class handler[V]:
    def __init__(self, *effects: Effect[..., Any]):
        global _num

        self._effects = tuple(effects)
        self._unbound_effects = set(effects)
        self._reserved_effects: list[tuple[Effect[..., Any], Callable[..., Any]]] = []
        self._n = _num
        _num += 1
        _debug(f"@ {self}")

    def on[**P, R](self, effect: Effect[P, R]) -> Callable[[EffectHandler[P, V, R]], Callable[P, V]]:
        _debug(f"|+ {effect} | {self}")

        resume: Resume[R, V] = Resume()

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

    def __call__(self, fn: Caller[V], *, check: bool = True) -> V:
        _debug(f"|> {self}")

        if check:
            self.check(fn)

        token = object()

        # setup
        _put_handlers(token, self, self._reserved_effects)

        try:
            caller_gl = gl.greenlet(fn)

            _debug(f"|> @caller | {self}")

            v = _drive(caller_gl, caller_gl.switch())

            _debug(f"|< @caller = {v!r}")

            return v
        finally:
            _remove_all_handlers(token)

    def __str__(self) -> str:
        return f"#handler({self._n}) | {', '.join(e.name for e in self._effects)}"


class async_handler[V]:
    def __init__(self, *effects: AsyncEffect):
        global _num

        self._effects = tuple(effects)
        self._unbound_effects = set(effects)
        self._reserved_effects: list[tuple[Effect, Callable]] = []
        self._n = _num
        _num += 1
        _debug(f"@ {self}")

    def on[**P, R](
        self,
        effect: AsyncEffect[P, R],
    ) -> Callable[[AsyncEffectHandler[P, V, R]], Callable[P, Coroutine[Any, Any, V]]]:
        _debug(f"|+ {effect} | {self}")

        resume: Resume[R, V] = Resume()

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
        _debug(f"|> {self}")

        if check:
            self.check(fn)

        token = object()

        # setup
        await _put_handlers_async(token, self, self._reserved_effects)

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
            await _remove_all_handlers_async(token)

    def __str__(self) -> str:
        return f"#handler({self._n}) | {', '.join(e.name for e in self._effects)}"


class EffectNotHandledError[**P, R](RuntimeError):
    def __init__(self, effect: Effect[P, R]):
        super().__init__(f"no handler for the effect: {effect}")
        self.effect = effect


class _Effect[**P, R](Effect[P, R]):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R:
        _debug(f"||> perform {self}")

        # pass to the main greenlet
        handler_gl = gl.getcurrent().parent
        if handler_gl is None:
            raise EffectNotHandledError(self)
        return handler_gl.switch(EffectContext(self, args, kwargs))

    def __str__(self) -> str:
        return f"<effect {self.name}>"


class _AsyncEffect[**P, R](_Effect[P, R]):
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        _debug(f"||> perform {self}")

        found = _get_item(self)
        if found is None:
            raise EffectNotHandledError(self)

        handler, fn = found

        _debug(f"||>     found handler {handler}")

        try:
            v = await fn(*args, **kwargs)
            _debug(f"||< **abort** perform {self} = {v!r}")
            raise _AbortSignal(handler, v)
        except _ResumeSignal as e:  # :: _ResumeSignal[R]
            _debug(f"||< perform {self} = {e.value!r}")
            return e.value  # :: R


# ---

_caller_gl = ContextVar("caller_gl", default=None)


def _get_caller() -> Any:
    g = _caller_gl.get()
    if g is None:
        raise RuntimeError("no caller found")
    return g


def _set_caller(g: Any) -> None:
    _caller_gl.set(g)


# ---

# handler stask
# NB. two greenlets cannot share the ContextVar
#     so only handler_gl should access this stack

# (runner, effect, handler)
type _StackItem[**P, R, V] = tuple[object, handler[V], Effect[P, R], Callable[P, V]]

_stack = ContextVar[list[_StackItem[..., Any, Any]]]("handler_stask", default=[])
_lock = Lock()


def _get_stack() -> list[_StackItem[..., Any, Any]]:
    return _stack.get()


def _set_stack(stack: list[_StackItem[..., Any, Any]]) -> None:
    _stack.set(stack)


def _get_item[**P, R](effect: Effect[P, R]) -> tuple[handler[Any], Callable[P, Any]] | None:
    s: list[_StackItem[..., Any, Any]] = _get_stack()
    for _, h, e, f in reversed(s):
        if e is effect:
            return h, f
    return None


def _put_handlers[**P, R, V](
    token: object,
    handler: handler[V],
    effects: list[tuple[Effect[P, R], Callable[P, V]]],
) -> None:
    stask = _get_stack()
    for effect, fn in effects:
        stask.append((token, handler, effect, fn))
    _set_stack(stask)


async def _put_handlers_async[**P, R, V](
    token: object,
    handler: handler[V],
    effects: list[tuple[Effect[P, R], Callable[P, V]]],
) -> None:
    async with _lock:
        stask = _get_stack()
        for effect, fn in effects:
            stask.append((token, handler, effect, fn))
        _set_stack(stask)


def _remove_all_handlers[**P, R, V](token: object) -> None:
    stack = [x for x in _get_stack() if x[0] is not token]
    _set_stack(stack)


async def _remove_all_handlers_async[**P, R, V](token: object) -> None:
    async with _lock:
        stack = _get_stack()
        stack = [x for x in stack if x[0] is not token]
        _set_stack(stack)
