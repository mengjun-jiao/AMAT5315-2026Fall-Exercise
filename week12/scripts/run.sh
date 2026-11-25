#!/usr/bin/env bash
set -euo pipefail
# Run from week12 with branchopt, pred and uv on PATH.
mkdir -p artifacts
bench() {
    local out=$1
    shift
    mkdir -p "$out"
    branchopt "$@" --out "$out" | tee "$out/stdout.tsv"
    if [[ "$out" == artifacts/eval_* ]]; then
        sha256sum artifacts/model.json > "$out/model.sha256"
    fi
}
case "${1:-}" in
  pred)
    mkdir -p artifacts/pred
    cd artifacts/pred
    pred --version
    pred create SpinGlass --graph 0-1,1-2,2-3,3-0 --couplings=1,1,1,-1 -o square.json > square.create.log 2>&1
    { time pred solve square.json; } > square.solve.log 2>&1
    pred create SpinGlass --random --num-vertices 100 --edge-prob 0.04 --seed 2026 -o r100.json > r100.create.log 2>&1
    { time pred solve r100.json; } > r100.solve.log 2>&1
    ;;
  gap) bench artifacts/gap --mode gap --spins 22 --edges 52 --count 16 --seed 2026 ;;
  bnb)
    bench artifacts/bnb_most --mode bnb --spins 22 --edges 52 --count 48 --policy most-frac --seed 2026
    bench artifacts/bnb_strong --mode bnb --spins 22 --edges 52 --count 48 --policy strong --seed 2026
    cp artifacts/bnb_strong/stdout.tsv artifacts/bnb_strong/stdout_no_labels.tsv
    ;;
  labels) bench artifacts/bnb_strong --mode bnb --spins 22 --edges 52 --count 48 --policy strong --export-labels --seed 2026 ;;
  train) uv run --with jax==0.12.1 python train_branch.py --dataset artifacts/bnb_strong/dataset.json --out artifacts/model.json ;;
  test) bench artifacts/eval_test --mode eval --spins 22 --edges 52 --count 16 --seed 3026 --model artifacts/model.json ;;
  transfer) bench artifacts/eval_transfer --mode eval --spins 30 --edges 72 --count 8 --seed 4026 --model artifacts/model.json ;;
  figures)
    shift
    uv run --with matplotlib==3.11.2 python scripts/plot_results.py "$@"
    ;;
  check)
    uv run --with numpy==2.5.5 --with scipy==1.18.1 python scripts/check_results.py
    uv run --with matplotlib==3.11.2 python scripts/check_figures.py
    ;;
  *) echo "Choose pred, gap, bnb, labels, train, test, transfer, figures, or check." >&2; exit 2 ;;
esac
