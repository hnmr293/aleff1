from typing import Callable, Any
from .intf import Effect, Handler, AsyncHandler


def effects(fn: Callable[..., Any]) -> frozenset[Effect[..., Any]]:
    """Return the set of effects used by the given function.

    The function must have been decorated with ``@effect(e1, e2, ...)``.

    ```python
    read: Effect[[], str] = effect("read")
    write: Effect[[str], int] = effect("write")

    @effect(read, write)
    def process():
        s = read()
        return write(s)

    effects(process)  # frozenset({read, write})
    ```
    """

    return getattr(fn, "__effects__", frozenset())


def unhandled_effects[V](
    fn: Callable[..., Any],
    *handlers: Handler[V] | AsyncHandler[V],
) -> frozenset[Effect[..., Any]]:
    """Return the set of effects used by *fn* that are not handled by *handlers*.

    ```python
    read: Effect[[], str] = effect("read")
    write: Effect[[str], int] = effect("write")

    @effect(read, write)
    def process():
        ...

    h = create_handler(read)

    @h.on(read)
    def _read(k):
        return k("")

    unhandled_effects(process, h)  # frozenset({write})
    ```
    """

    fn_effects = effects(fn)

    handled_effects: set[Effect[..., Any]] = set()
    for h in handlers:
        handled_effects.update(h.effects)

    return fn_effects - handled_effects
