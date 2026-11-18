#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bin=.install/bin/tnsample
"$bin" --mode count --l 10 --p-af 0 --beta 2.0 --seed 2026 --out runs/count_p0
"$bin" --mode count --input runs/pred/factor_15.bundle.json --beta 2.0 --seed 2026 --out runs/count_f15
"$bin" --mode sample --input runs/pred/factor_15.bundle.json --beta 2.0 --samples 2000 --seed 2026 --out runs/sample_f15
for k in 2 3 4 5 6 7 8; do
  for ord in greedy treesa; do
    "$bin" --mode plan --order "$ord" --input "runs/pred/scale_$k.bundle.json" --seed 2026 --out "runs/plan_k${k}_$ord"
  done
done
