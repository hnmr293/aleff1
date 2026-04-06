#!/usr/bin/env bash
set -euo pipefail

targets=(
    "3.12"
    "3.13"
    "3.14"
)

for ver in "${targets[@]}"; do
    PYTHON="python$ver" make debug
done

for ver in "${targets[@]}"; do
    echo "===== Python $ver ====="
    uv run --quiet --python "$ver" pytest tests/ -q
    echo
done
