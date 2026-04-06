from typing import Any, Callable, Concatenate, Coroutine, Protocol, runtime_checkable


##
# type notations:
#   P: effect parameters
#   R: effect return type
#   V: handler return type
##


@runtime_checkable
class Effect[**P, R](Protocol):
    @property
    def name(self) -> str: ...

    def __call__[V](self, *args: P.args, **kwargs: P.kwargs) -> R: ...


@runtime_checkable
class Resume[R, V](Protocol):
    def __call__(self, value: R) -> V: ...


@runtime_checkable
class ResumeAsync[R, V](Protocol):
    async def __call__(self, value: R) -> V: ...


type Caller[V] = Callable[[], V]
type AsyncCaller[V] = Coroutine[Any, Any, V]
type EffectHandler[**P, V, R] = Callable[Concatenate[Resume[R, V], P], V]
type AsyncEffectHandler[**P, V, R] = Callable[Concatenate[ResumeAsync[R, V], P], AsyncCaller[V]]


class handler[V](Protocol):
    def on[**P, R](self, effect: Effect[P, R]) -> Callable[[EffectHandler[P, V, R]], Callable[P, V]]: ...

    def check(self, caller: Caller[V]) -> None: ...

    def __call__(self, caller: Caller[V], *, check: bool = True) -> V: ...


class async_handler[V](Protocol):
    def on[**P, R](
        self,
        effect: Effect[P, R],
    ) -> Callable[[AsyncEffectHandler[P, V, R]], Callable[P, Coroutine[Any, Any, V]]]: ...

    def check(self, caller: Caller[V | Coroutine[Any, Any, V]]) -> None: ...

    async def __call__(self, caller: Caller[V | Coroutine[Any, Any, V]], *, check: bool = True) -> V: ...


class EffectNotHandledError[**P, R](RuntimeError):
    def __init__(self, effect: Effect[P, R]):
        super().__init__(f"no handler for the effect: {effect}")
        self.effect = effect
