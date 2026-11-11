# Week 10: Approximate tensor contraction for the square Ising model

This project implements the Week 10 hands-on exercises with CPU Rust: a Gaussian-matrix SVD, open-boundary contraction with a boundary MPS, and periodic-boundary tensor renormalization group (TRG). It estimates the two-dimensional zero-field Ising critical temperature from finite-size specific-heat peaks. Spins are ±1, coupling and Boltzmann's constant are one, and the Hamiltonian is `H = -sum_edges s_i s_j`.

## Layout and environment

- `src/svd_evidence.rs`: seed-2026 Gaussian matrix and saved SVD reconstructions.
- `src/boundary.rs`: open-square row absorption, exact right-canonical sweep and truncated left-canonical sweep.
- `src/trg.rs`: the two diagonal splits, four-factor regrouping, scale tracking and independent opposite-leg closure.
- `src/linalg.rs`: f64 matrix multiplication and thin SVD through OpenBLAS; `src/main.rs` implements `thermo`.
- `thermo.design.toml`: command and evidence specification. `tests/small_lattices.rs` and `scripts/` provide independent enumeration, scans, plots and audits.
- `artifacts/`: the three requested figures are included in Git. Saved SVD arrays, g(K) tables, peak/verification tables and audit results remain local. `runs/` contains individual run records and complete peak-stencil tensors and scales. Raw data, run records, the supplied PDF, local reference copy, virtual environment, toolchain and build products are ignored by Git.

The recorded environment is WSL2 Linux x86-64, Rust/Cargo 1.96.0, Python 3.14.4, NumPy 2.5.4 and Matplotlib 3.11.2. The Rust implementation dynamically loads the NumPy wheel's CPU OpenBLAS ILP64 library and uses its `scipy_LAPACKE_dgesdd64_` and `scipy_cblas_dgemm64_` symbols. This backend is deliberately pinned to that NumPy wheel interface; builds for platforms or wheels with different library names or symbols require a corresponding backend change. No GPU is used. The observed scans used four independent worker processes and one OpenBLAS thread per process.

The Ising calculations use generated tensors, not an external dataset. The TRG split/regrouping is adapted from the [fixed BackwardsLinalg.jl example](https://github.com/GiggleLiu/BackwardsLinalg.jl/blob/b16273a2d695fef31334085e40f9e0341a1ec0f0/examples/trg.jl); as required by the learning sheet, the final closure here is `sum_ij A[i,j,i,j]`, not the source example's four-equal-index closure. The original learning sheet is retained locally.

## Reproduce and verify

Use Bash in a Linux x86-64 environment with Python, pip, a C toolchain, `rustup`, and network access. From `week10/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source scripts/env.sh
rustup toolchain install 1.96.0 --profile minimal --no-self-update
cargo +1.96.0 install --locked --path . --root "$PWD/.install" --force
cargo +1.96.0 test --release --locked

svd-evidence --out artifacts/svd-data
python scripts/plot_svd.py
python scripts/check_small.py

thermo --method boundary --l 64 --k 0.45 --chi 16 --out runs/boundary64
thermo --method trg --l 1024 --k 0.44 --chi 16 --out runs/trg1024
python scripts/sweep.py --workers 4 --method all
python scripts/audit_results.py
python scripts/verify_saved.py
```

The environment script sets project-local `RUSTUP_HOME` and `CARGO_HOME`, locates the NumPy OpenBLAS shared library, and defaults to one BLAS thread. Set `WEEK10_RUSTUP_HOME` only if intentionally using another toolchain directory, or `WEEK10_BLAS_THREADS` to change BLAS threads. The CLI requires method, L, K, χ, and output path; its default `--record full` stores the full evidence. `--record summary` saves metadata and scales for broad scans. Every nine-point peak stencil in the recorded scans uses full mode.

The sweep covers K = 0.40 through 0.50 in steps of 0.0025 at χ=16. For each size it then uses nine points at spacing 0.00125 around the nearest coarse peak, for χ=16 and χ=32. The wider-step χ=32 comparison uses every other refined point. Three-point parabolas are fitted in K; their vertices are converted to `T_peak=1/K_peak`, then the two finite-size fits are performed independently for each geometry.

Cached sweep runs are bound to the SHA-256 hash of the installed `thermo` binary in `runs/sweep-config.json`; the script refuses to mix results from another binary. If the executable changes, preserve `runs/sweep/` and `runs/sweep-config.json` elsewhere before starting a fresh scan. Re-running the scan with the same binary reuses complete runs. Solver times in the tables are sums of per-run elapsed times; workers run concurrently, and coarse and refined grids overlap, so these sums must not be added to infer wall-clock runtime.

## SVD results

The 512×512 matrix uses `ChaCha20Rng::seed_from_u64(2026)` and `rand_distr::StandardNormal`. The saved reconstructions at ranks 16, 64, 128, 256 and 512 have independently measured relative Frobenius errors 0.942178, 0.790032, 0.614586, 0.326611 and 2.92e-15. Each agrees with the discarded singular-spectrum prediction within 1e-8.

The first rank with error at most 10% is 395. Two factors at that rank use 1.542969 times the original matrix's storage; at rank 256 they already use equal storage, so this random matrix does not achieve a storage-saving 10% approximation. `artifacts/svd.png` plots the spectrum, all-rank tail prediction, saved reconstruction errors, equal-storage boundary and 10% threshold.

## Open and periodic critical peaks

For χ=32 and refined `h=0.00125`:

| Geometry | L | Peak temperature | χ16→χ32 shift | h=.0025→.00125 shift | Sum of nine χ32 solver times |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open | 16 | 2.093105 | 0.000000014% | 0.001445% | 0.61 s |
| Open | 32 | 2.179676 | 0.000001702% | 0.000969% | 5.94 s |
| Open | 64 | 2.224220 | 0.000042373% | 0.001175% | 37.21 s |
| Periodic | 64 | 2.281602 | 0.098265% | 0.000155% | 127.54 s |
| Periodic | 256 | 2.272260 | 0.012180% | 0.015856% | 172.13 s |
| Periodic | 1024 | 2.270012 | 0.035255% | 0.030671% | 213.76 s |

Open-boundary fits give **2.26750609** using all three sizes and **2.26876460** using the two largest. Periodic fits give **2.26920472** and **2.26926314** respectively. The Onsager value is **2.26918531**; all four estimates are within 1%, and all recorded rank and grid-step peak shifts are below 1%. Open peaks approach from below, periodic peaks from above. The periodic plot zooms to T=2.225–2.325 to match the reference figure; the full scanned g(K) values remain in the CSV.

`artifacts/boundary-critical.png` and `artifacts/trg-critical.png` show the heat-capacity curves, χ=32 refined points, both peak extrapolations and summary tables. Their adjacent `*-g.csv`, `*-peak-table.csv`, `*-small-checks.csv` and `*-summary.json` files retain values, paths, timings, fits and small-system checks at full precision.

## Independent checks

- Open L=4, χ=8: g at K=0, 0.44 and 0.7 agrees with independent enumeration within 2.23e-16. At K=0.44, the finite-difference heat capacity is 0.43939307 versus the energy-variance result 0.43939691.
- Periodic L=2, χ=16: g differs from enumeration by 2.23e-16, and heat capacity is 0.40428194 versus 0.40427823 from the energy variance. L=4, χ=8 differs in g by 0.00010564, below the requested 0.002 tolerance. The periodic enumerator counts each positive-direction bond per site, including both distinct parallel bonds at L=2.
- `scripts/audit_results.py` independently recomputes the SVD errors, all finite differences and peak fits from the stored g tables, checks all 108 refined runs have full evidence, and checks each plotted series against the numerical source.
- `scripts/verify_saved.py` independently closes all 55 full open runs and all 55 full periodic runs (54 peak-stencils plus one specified example per method). It checks left-isometry of boundary cores and separately reconstructs every TRG split and regrouping for the L=1024 examples at χ=16 and χ=32. The largest recovered log-Z absolute discrepancies are 1.37e-12 for boundary and 3.50e-10 for TRG; the independent full-round reconstructions match the final tensors within 2.55e-16 relative maximum error.

The three requested figures are tracked. Raw arrays, tensors, scales and other run outputs remain local and are excluded from Git.
