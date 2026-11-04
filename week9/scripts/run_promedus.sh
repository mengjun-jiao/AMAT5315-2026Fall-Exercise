#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p artifacts
export RAYON_NUM_THREADS=1
model=promedus/Promedus_14.uai
evidence=promedus/Promedus_14.uai.evid
infer=./.install/bin/infer
"$infer" --model "$model" --evidence "$evidence" --order greedy --query 59,102,139 --out artifacts/promedus-greedy >artifacts/promedus-greedy.log
"$infer" --model "$model" --evidence "$evidence" --order treesa --seed 42 --query 59,102,139 --out artifacts/promedus-treesa >artifacts/promedus-treesa.log
"$infer" --model "$model" --evidence "$evidence" --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out artifacts/promedus-bp >artifacts/promedus-bp.log
for order in greedy treesa; do
  "$infer" --model "$model" --evidence "$evidence" --analyze "artifacts/promedus-$order/order.json" --out "artifacts/promedus-$order-analysis" >"artifacts/promedus-$order-analysis.log"
done
printf 'Promedus inference and saved-plan analysis completed.\n'
