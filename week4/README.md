# Week 4: Continuum Fluid Dynamics

## Overview

This week implements reusable Forward Euler, explicit midpoint, and classical RK4 time integrators; a one-dimensional periodic advection-diffusion solver with Fourier and centered finite-difference derivatives; RK stability and temporal-order validation; and a two-dimensional incompressible vorticity-streamfunction solver.

The two-dimensional solver uses Fourier pseudospectral differentiation, streamfunction-based velocity recovery, and two-thirds dealiasing. The validation studies cover Taylor-Green flow, a seeded random flow, diffusive and advective stability limits, sensitivity to initial conditions, RK4 temporal convergence, and Richardson time-step selection.

The optional Week 4 Challenge is not included.

## Requirements

- Rust and Cargo
- Python 3
- Pillow (`PIL`), used by the plotting scripts

The numerical solvers are Rust binaries. The Python scripts use only the standard library and Pillow for orchestration, analysis, and rendering.

## Build and Install

From the repository root:

```text
cd week4
cargo build --release
cargo install --path . --quiet
mkdir -p artifacts evidence
```

`artifacts/` is intentionally ignored by Git. The committed figures and JSON evidence remain in `evidence/`.

## Command-Line Tools

`field` writes one JSON field object to standard output. It supports the exact Taylor-Green field and a deterministic seeded random field.

`fluid` reads a field object from standard input, integrates vorticity with Euler, midpoint, or RK4, prints snapshot diagnostics, and writes `run.json` and `fields.jsonl` under the requested output directory.

Taylor-Green example:

```text
field taylor-green --n 64 |
  fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 \
  --every 0.1 --out artifacts/taylor-green \
  > artifacts/taylor-green.tsv
```

Random-flow example:

```text
field random --n 128 --seed 2026 --k-min 2 --k-max 6 |
  fluid --method rk4 --nu 0.004 --dt 0.01 --t-end 10 \
  --every 0.1 --out artifacts/random \
  > artifacts/random.tsv
```

## Reproducing the Evidence

Run these commands from `week4/` in the listed order. They do not assume any pre-existing local `artifacts/` directory.

```text
mkdir -p artifacts evidence

field taylor-green --n 64 |
  fluid --method rk4 --nu 0.1 --dt 0.01 --t-end 1 \
  --every 0.1 --out artifacts/taylor-green \
  > artifacts/taylor-green.tsv
field taylor-green --n 64 --nu 0.1 --t 1 \
  > artifacts/taylor-green/exact-t1.json

python3 scripts/compare_derivatives.py
python3 scripts/plot_line_stability.py
python3 scripts/plot_line_accuracy.py
python3 scripts/plot_taylor_green.py

python3 scripts/run_part3.py
python3 scripts/plot_part3.py

python3 scripts/run_part4.py
python3 scripts/plot_part4.py
```

The committed evidence files are regenerated as follows:

| File | Producing command or prerequisite |
| --- | --- |
| `evidence/line-stability.png` | `python3 scripts/plot_line_stability.py` |
| `evidence/line-accuracy.png` | `python3 scripts/plot_line_accuracy.py` |
| `evidence/taylor-green.png` | Taylor-Green commands above, then `python3 scripts/plot_taylor_green.py` |
| `evidence/random.png` | `python3 scripts/run_part3.py`, then `python3 scripts/plot_part3.py` |
| `evidence/blowup.png` | `python3 scripts/run_part3.py`, then `python3 scripts/plot_part3.py` |
| `evidence/sensitivity.png` | `python3 scripts/run_part3.py`, then `python3 scripts/plot_part3.py` |
| `evidence/order.png` | `python3 scripts/run_part4.py`, then `python3 scripts/plot_part4.py` |
| `evidence/convergence.png` | `python3 scripts/run_part4.py`, then `python3 scripts/plot_part4.py` |
| `evidence/convergence.json` | `python3 scripts/run_part4.py` |
| `evidence/order.json` | `python3 scripts/run_part4.py` |

`run_part3.py` generates the reference random run, Taylor-Green stability scans, random-flow stability scans, and both sensitivity pairs. `run_part4.py` generates the Taylor-Green order study, random-flow self-convergence study, `order.json`, and `convergence.json`. The plotting scripts read those Rust-generated artifacts; they do not replace the Rust solver.

## Validation Results

Part 1:

```text
RK4 line stability limit: 0.049386063870

Euler                 1.032742
midpoint              2.005486
RK4                   4.003965
equal-weight 4-stage 2.002533
```

Part 2:

```text
Fourier derivative errors: below 1e-10
Centered finite-difference N=32/N=64 error ratios: approximately 4

Taylor-Green E(0) = 0.250000
Taylor-Green Z(0) = 0.500000
Taylor-Green E(1) = 0.167580
Taylor-Green Z(1) = 0.335160
Velocity relative error = 7.03858653e-07
```

Part 3:

```text
Random reference E(0)  = 0.50000000
Random reference Z(0)  = 6.63468483
Random reference E(10) = 0.28419699
Random reference Z(10) = 1.02502471

Taylor-Green predicted diffusive limit = 0.031575963719
Taylor-Green measured bracket: 0.032 stable, 0.033 unstable
Random initial Umax = 2.306233859809
Predicted random advective bound = 0.020659452271
Random RK4 measured bracket: 0.035 stable, 0.038 unstable
Random sensitivity growth = 68.4486584
```

Part 4:

```text
Taylor-Green RK4 order = 4.10440618
Random temporal convergence order = 4.03858956
Richardson selected dt = 0.0125
Predicted error = 3.426060666100e-06
Measured error = 3.344346869443e-06
Acceptance threshold = 5e-6
```

The random-flow values use this implementation's deterministic seeded phase generator.

## Generated Artifacts

`week4/artifacts/` contains regenerable simulation output and is intentionally ignored by Git. It is not required to be copied into a clean clone.

Committed evidence is stored in `week4/evidence/`.

## Tests

From `week4/`, run:

```text
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```

The Rust test suite covers the integrators, one-dimensional derivatives, periodic wrapping, Nyquist handling, two-dimensional derivatives, streamfunction recovery, dealiasing, Taylor-Green identities, and random-field resolution consistency.

## Repository Layout

```text
Cargo.toml              Rust crate and binary definitions
Cargo.lock              Locked Rust dependency state
field.design.toml       field command contract
fluid.design.toml       fluid command contract
src/                    reusable solver library and binaries
scripts/                reproducible run, comparison, and plotting scripts
evidence/               committed figures and numerical evidence JSON
artifacts/              ignored regenerable simulation output
```
