"""
Demo: State Machine with Shallow Handlers

Implements mutable state (get/put) using shallow handlers.
Each effect invocation is handled once, then the handler re-installs
itself with the (potentially updated) state.

With deep handlers, the handler stays installed across all effect
invocations and must use mutable variables to track state.
With shallow handlers, state is encoded in the handler itself:
each invocation explicitly re-installs the handler with new state.

This is a classic pattern from algebraic effects research.
"""

from typing import Callable

from aleff import (
    effect,
    Effect,
    Resume,
    Handler,
    create_handler,
)


# ---------------------------------------------------------------------------
# Effects
# ---------------------------------------------------------------------------

get: Effect[[], int] = effect("get")
put: Effect[[int], None] = effect("put")


# ---------------------------------------------------------------------------
# Computation (pure — no knowledge of how state is managed)
# ---------------------------------------------------------------------------


def counter() -> list[int]:
    """Increment state three times, collecting intermediate values."""
    results: list[int] = []
    for _ in range(3):
        x = get()
        results.append(x)
        put(x + 1)
    return results


def factorial(n: int) -> int:
    """Compute n! using get/put as an accumulator."""
    put(1)
    for i in range(1, n + 1):
        acc = get()
        put(acc * i)
    return get()


# ---------------------------------------------------------------------------
# Shallow handler: state encoded in handler re-installation
# ---------------------------------------------------------------------------


def run_state[V](init: int, comp: Callable[[], V]) -> tuple[int, V]:
    """Run a computation with get/put state, starting from *init*.

    Returns ``(final_state, result)``.

    Each get/put is handled by a shallow handler that re-installs
    itself with the current (or updated) state value.
    """

    # We use a mutable container to capture the final state,
    # since the outermost handler sees the last put() value.
    state_box: list[int] = [init]

    def _handler(s: int, computation: Callable[[], V]) -> V:
        h: Handler[V] = create_handler(get, put, shallow=True)

        @h.on(get)
        def _get(k: Resume[int, V]) -> V:
            # Return current state to the caller, re-install with same state
            return _handler(s, lambda: k(s))

        @h.on(put)
        def _put(k: Resume[None, V], new_s: int) -> V:
            # Re-install with updated state
            state_box[0] = new_s
            return _handler(new_s, lambda: k(None))

        return h(computation, check=False)

    result = _handler(init, comp)
    return state_box[0], result


# ---------------------------------------------------------------------------
# Deep handler (for comparison): state tracked via mutable variable
# ---------------------------------------------------------------------------


def run_state_deep[V](init: int, comp: Callable[[], V]) -> tuple[int, V]:
    """Same semantics, but with a deep handler and mutable state."""
    state = [init]

    h: Handler[V] = create_handler(get, put)

    @h.on(get)
    def _get(k: Resume[int, V]) -> V:
        return k(state[0])

    @h.on(put)
    def _put(k: Resume[None, V], new_s: int) -> V:
        state[0] = new_s
        return k(None)

    result = h(comp, check=False)
    return state[0], result


# ---------------------------------------------------------------------------
# State machine: traffic light controller
# ---------------------------------------------------------------------------

next_signal: Effect[[], str] = effect("next_signal")


def traffic_light_cycle() -> list[str]:
    """Run 8 signal transitions and collect the colors."""
    colors: list[str] = []
    for _ in range(8):
        color = next_signal()
        colors.append(color)
    return colors


def run_traffic_light() -> list[str]:
    """Traffic light as a state machine using shallow handlers.

    States: RED -> GREEN -> YELLOW -> RED -> ...

    Each shallow handler handles one next_signal() invocation,
    returns the current color, and re-installs the next state's handler.
    """
    transitions: dict[str, str] = {
        "RED": "GREEN",
        "GREEN": "YELLOW",
        "YELLOW": "RED",
    }

    def _state(color: str, computation: Callable[[], list[str]]) -> list[str]:
        h: Handler[list[str]] = create_handler(next_signal, shallow=True)

        @h.on(next_signal)
        def _next(k: Resume[str, list[str]]) -> list[str]:
            next_color = transitions[color]
            # Return current color, transition to next state
            return _state(next_color, lambda: k(color))

        return h(computation, check=False)

    return _state("RED", traffic_light_cycle)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # --- State effect ---
    print("=== Shallow handler: counter ===")
    final_state, result = run_state(0, counter)
    print(f"  counter()  = {result}")
    print(f"  final state = {final_state}")
    assert result == [0, 1, 2]
    assert final_state == 3

    print()
    print("=== Deep handler: counter (for comparison) ===")
    final_state_d, result_d = run_state_deep(0, counter)
    print(f"  counter()  = {result_d}")
    print(f"  final state = {final_state_d}")
    assert result_d == result
    assert final_state_d == final_state

    print()
    print("=== Shallow handler: factorial ===")
    final_state, result_f = run_state(0, lambda: factorial(5))
    print(f"  factorial(5) = {result_f}")
    print(f"  final state  = {final_state}")
    assert result_f == 120
    assert final_state == 120

    print()
    print("=== Shallow handler: traffic light ===")
    colors = run_traffic_light()
    print(f"  signals = {colors}")
    assert colors == [
        "RED",
        "GREEN",
        "YELLOW",
        "RED",
        "GREEN",
        "YELLOW",
        "RED",
        "GREEN",
    ]

    print()
    print("All demos passed.")
