#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.install/bin:$PATH"
mkdir -p runs/pred
pred path Factoring SpinGlass > runs/pred/path.txt 2>&1
pred create Factoring --target 15 --m 2 --n 3 -o runs/pred/factor_15.json > runs/pred/factor_15-create.txt 2>&1
pred reduce runs/pred/factor_15.json --to SpinGlass -o runs/pred/factor_15.bundle.json > runs/pred/factor_15-reduce.txt 2>&1
pred solve runs/pred/factor_15.bundle.json > runs/pred/factor_15-solve.txt 2>&1
pred evaluate runs/pred/factor_15.json --config 1,1,1,0,1 > runs/pred/factor_15-evaluate.txt 2>&1
for n in 6 7; do
  pred create Factoring --target "$n" --m 2 --n 2 -o "runs/pred/factor_$n.json" > "runs/pred/factor_$n-create.txt" 2>&1
  pred reduce "runs/pred/factor_$n.json" --to SpinGlass -o "runs/pred/factor_$n.bundle.json" > "runs/pred/factor_$n-reduce.txt" 2>&1
  pred solve "runs/pred/factor_$n.bundle.json" > "runs/pred/factor_$n-solve.txt" 2>&1
done
for k in 2 3 4 5 6 7 8; do
  t=$(( ((1 << k) - 1) * ((1 << k) - 1) ))
  pred create Factoring --target "$t" --m "$k" --n "$k" -o "runs/pred/scale_$k.json" > "runs/pred/scale_$k-create.txt" 2>&1
  pred reduce "runs/pred/scale_$k.json" --to SpinGlass -o "runs/pred/scale_$k.bundle.json" > "runs/pred/scale_$k-reduce.txt" 2>&1
done
echo 'Reduction bundles and CLI output saved in runs/pred.'
