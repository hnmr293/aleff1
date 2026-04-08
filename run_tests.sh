#!/usr/bin/env bash
set -euo pipefail

targets=(
    "3.12.13"
    "3.13.12"
    "3.14.3"
    "3.14.3t"
)

for ver in "${targets[@]}"; do
    PYTHON="uv run --python $ver python" make debug
done

for ver in "${targets[@]}"; do
    echo "===== Python $ver ====="
    uv run --quiet --python "$ver" pytest tests/ -q
    echo
done
