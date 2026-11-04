#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
INFER=./.install/bin/infer
mkdir -p artifacts runs
for case in prior AD ADX; do
 "$INFER" --model asia.uai --evidence "$case" --out "artifacts/$case" >"artifacts/exact-$case.log"
 "$INFER" --model asia.uai --evidence "$case" --method bp --damping 0.5 --tolerance 1e-10 --max-sweeps 1000 --out "artifacts/bp-$case" >"artifacts/bp-$case.log"
done
mkdir -p runs
"$INFER" --model asia.uai --evidence AD --order random --seed 42 --out runs/random >runs/random.log
"$INFER" --model asia.uai --evidence AD --order greedy --out runs/greedy >runs/greedy.log
"$INFER" --model asia.uai --evidence AD --analyze runs/random/order.json --out runs/analysis >runs/analysis.log
"$INFER" --model asia.uai --evidence AD --plan runs/greedy/order.json --out runs/replay >runs/replay.log
printf 'ASIA exact, BP and order runs completed.\n'
