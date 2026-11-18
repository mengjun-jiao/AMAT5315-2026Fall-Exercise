#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bin=.install/bin/tnsample
for n in 4 5 6 7 8 9 10; do
  "$bin" --mode count --n "$n" --seed 2026 --out "runs/count_nq$n"
done
for n in 6 8; do
  "$bin" --mode sample --n "$n" --samples 200 --seed 2026 --out "runs/sample_nq$n"
done
