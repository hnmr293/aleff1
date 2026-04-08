"""Performance benchmarks for aleff.

Run with: uv run pytest benchmarks/ --benchmark-only
"""

from __future__ import annotations

from typing import Callable

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # pyright: ignore[reportMissingTypeStubs]

from aleff import effect, Effect, Resume, Handler, create_handler
from aleff.oneshot import (
    effect as oneshot_effect,
    Effect as OneshotEffect,
    Resume as OneshotResume,
    Handler as OneshotHandler,
    create_handler as oneshot_create_handler,
)


# ---------------------------------------------------------------------------
# Baseline: plain function call
# ---------------------------------------------------------------------------


def plain_function(x: int) -> int:
    return x + 1


def test_baseline_plain_call(benchmark: BenchmarkFixture) -> None:
    benchmark(plain_function, 42)


# ---------------------------------------------------------------------------
# One-shot: single effect invoke & resume
# ---------------------------------------------------------------------------


get_value: Effect[[], int] = effect("get_value")
h_single: Handler[int] = create_handler(get_value)


@h_single.on(get_value)
def _get_value(k: Resume[int, int]) -> int:
    return k(42)


def test_oneshot_single_effect(benchmark: BenchmarkFixture) -> None:
    def run() -> int:
        return get_value()

    benchmark(h_single, run)


# ---------------------------------------------------------------------------
# One-shot: multiple sequential effects
# ---------------------------------------------------------------------------


eff_a: Effect[[], int] = effect("eff_a")
eff_b: Effect[[], int] = effect("eff_b")
eff_c: Effect[[], int] = effect("eff_c")
h_multi: Handler[int] = create_handler(eff_a, eff_b, eff_c)


@h_multi.on(eff_a)
def _a(k: Resume[int, int]) -> int:
    return k(1)


@h_multi.on(eff_b)
def _b(k: Resume[int, int]) -> int:
    return k(2)


@h_multi.on(eff_c)
def _c(k: Resume[int, int]) -> int:
    return k(3)


def test_oneshot_multiple_effects(benchmark: BenchmarkFixture) -> None:
    def run() -> int:
        return eff_a() + eff_b() + eff_c()

    benchmark(h_multi, run)


# ---------------------------------------------------------------------------
# Multi-shot: resume N times
# ---------------------------------------------------------------------------


choose: Effect[[], int] = effect("choose")


@pytest.mark.parametrize("n_shots", [2, 5, 10, 20])
def test_multishot_resume_n(benchmark: BenchmarkFixture, n_shots: int) -> None:
    h: Handler[list[int]] = create_handler(choose)

    @h.on(choose)
    def _choose(k: Resume[int, list[int]]) -> list[int]:
        results: list[int] = []
        for i in range(n_shots):
            results.extend(k(i))
        return results

    def run() -> list[int]:
        return [choose()]

    benchmark(h, run)


# ---------------------------------------------------------------------------
# Scaling: nested handler depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [1, 5, 10, 20])
def test_nested_handler_depth(benchmark: BenchmarkFixture, depth: int) -> None:
    effects: list[Effect[[], int]] = [effect(f"e_{i}") for i in range(depth)]
    handlers: list[Handler[int]] = []
    for eff in effects:
        h: Handler[int] = create_handler(eff)

        @h.on(eff)
        def _impl(k: Resume[int, int], _eff: Effect[[], int] = eff) -> int:
            return k(1)

        handlers.append(h)

    def run() -> int:
        total = 0
        for eff in effects:
            total += eff()
        return total

    def nested() -> int:
        def _wrap(prev: Callable[[], int], handler: Handler[int]) -> Callable[[], int]:
            def wrapped() -> int:
                return handler(prev)
            return wrapped

        result: Callable[[], int] = run
        for handler in reversed(handlers):
            result = _wrap(result, handler)
        return result()

    benchmark(nested)


# ---------------------------------------------------------------------------
# Scaling: call stack depth before effect invocation
# ---------------------------------------------------------------------------


depth_eff: Effect[[], int] = effect("depth_eff")
h_depth: Handler[int] = create_handler(depth_eff)


@h_depth.on(depth_eff)
def _depth(k: Resume[int, int]) -> int:
    return k(1)


def _make_deep_caller(depth: int) -> Callable[[], int]:
    def leaf() -> int:
        return depth_eff()

    def _wrap(prev: Callable[[], int]) -> Callable[[], int]:
        def wrapped() -> int:
            return prev()
        return wrapped

    fn: Callable[[], int] = leaf
    for _ in range(depth):
        fn = _wrap(fn)
    return fn


@pytest.mark.parametrize("depth", [1, 5, 10, 20, 50])
def test_call_stack_depth(benchmark: BenchmarkFixture, depth: int) -> None:
    caller = _make_deep_caller(depth)
    benchmark(h_depth, caller)


# ---------------------------------------------------------------------------
# Comparison: oneshot module vs multishot module
# ---------------------------------------------------------------------------


os_eff: OneshotEffect[[], int] = oneshot_effect("os_eff")
os_h: OneshotHandler[int] = oneshot_create_handler(os_eff)


@os_h.on(os_eff)
def _os_impl(k: OneshotResume[int, int]) -> int:
    return k(1)


ms_eff: Effect[[], int] = effect("ms_eff")
ms_h: Handler[int] = create_handler(ms_eff)


@ms_h.on(ms_eff)
def _ms_impl(k: Resume[int, int]) -> int:
    return k(1)


def test_comparison_oneshot_module(benchmark: BenchmarkFixture) -> None:
    def run() -> int:
        return os_eff()

    benchmark(os_h, run)


def test_comparison_multishot_module(benchmark: BenchmarkFixture) -> None:
    def run() -> int:
        return ms_eff()

    benchmark(ms_h, run)
