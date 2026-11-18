# Week 11: Reductions and tensor-network sampling

This project follows the Week 11 learning sheet: reduce factoring to a spin glass using `problemreductions-cli`, enumerate Ising logic gadgets in Rust, contract and sample magnets, and count and sample N-queens boards through sparse carry tensors. All requested figures are drawn by Python scripts from the saved Rust run records.

## Environment and reproducibility

The recorded runs used WSL2 Ubuntu 26.04 (x86-64), Rust/Cargo 1.96.0, Python 3.14.4, `problemreductions-cli` (`pred`) 0.6.0, `omeco` 0.2.6, NumPy 2.5.5 and Matplotlib 3.11.2. `requirements.txt` also provides CMake and libclang needed to build the ILP backend. A C/C++ toolchain, Python venv support, rustup and network access are required for initial setup.

From `week11/`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
source scripts/env.sh
rustup toolchain install 1.96.0 --profile minimal --no-self-update
cargo install problemreductions-cli@0.6.0 --locked --root "$PWD/.install"
cargo install --locked --path . --root "$PWD/.install" --force

bash scripts/run_reductions.sh
tnsample --mode gadgets --seed 2026 --out runs/gadgets
python scripts/plot_gates.py
bash scripts/run_magnets.sh
python scripts/extract_sample.py
python scripts/plot_factoring.py
bash scripts/run_queens.sh
python scripts/plot_nqueens.py
python scripts/audit.py
python scripts/strict_audit.py
```

`scripts/env.sh` confines the Rust toolchain, Cargo cache, binaries and Python dependencies to this week, sets `LIBCLANG_PATH` for the bundled wheel, and sets eight planner threads by default; use `WEEK11_THREADS` to change the thread count before sourcing. All requested seeds are 2026. Random spin bonds and sample draws use `ChaCha20Rng::seed_from_u64`. `pred` 0.6.0 prints a recovered panic from one attempted reduction while solving; the script checks for successful exit, and the final JSON, including the `ilp` solution, is retained in `runs/pred/`.

`omeco` 0.2.6 initializes TreeSA trials with its own internal trial seeds, and tie ordering may depend on hash iteration. Magnet planning uses its default schedule with eight trials; the N-queens planner uses twelve trials with a space-optimized score. The recorded seed controls our spin/sample generator, not every planner decision. Consequently, repeating a run can produce another valid contraction tree and a different sample histogram; the actual tree, run parameters, all samples and energies are saved in `run.json`. Reference plot numbers are not substituted for measured data.

## What was computed

1. **Reduction round trip.** `pred` follows Factoring → CircuitSAT → SpinGlass. For 15 with factor bit lengths 2 and 3, the 104-spin ground state decodes to `[1,1,1,0,1]`, meaning 3 × 5, with `Min(0)`. The 2-bit case for 6 also yields `Min(0)`; the 7 case has a nonzero residual. Full JSON bundles, solver output and intermediate configurations are retained locally.
2. **Gate enumeration.** `src/gadgets.rs` saves all truth-table rows of AND, OR, NOT, COPY, SET0, SET1 and XOR (minimizing over both XOR helper states), and all 16 circuit states. The circuit reaches −5 on exactly three states. It also enumerates all 4096 states of the 12-spin multiplier cell and checks the 16 ground states at −20. `artifacts/gates.png` displays every gate and circuit row.
3. **Magnets.** `src/magnet.rs` constructs tensors with the bundle convention `H = +sum J_ij s_i s_j + sum h_i s_i`; open lattices use coefficients `−J` for the sheet's `H = −sum J_ij s_i s_j`. `omeco` plans a binary tree; the upward contraction combines rescaled Boltzmann weights with min-plus energy/degeneracy pairs. The downward pass draws the indices removed at each node using its stored children. Full scale factors are accumulated in log space.
4. **N-queens.** `src/queens_network.rs` explicitly constructs the prescribed 0/1 three-leg carry tensors for each row, column and diagonal, folds the end conditions into their boundary factors, and has `omeco` plan the binary contraction tree. Contractions join and accumulate **only nonzero entries**. To keep N=10 intermediate tables manageable, count mode sums ten disjoint slices fixing the first-row queen position; their individual counts and the tree are saved. N=6 and N=8 samples are drawn by a downward pass through the stored sparse contraction tree. `src/queens.rs` provides a separate row-frontier check, enabled only by `WEEK11_ROW_FRONTIER=1`; it is not used for the reported plots.

`tnsample.design.toml` is the assignment specification. `scripts/run_reductions.sh`, `scripts/run_magnets.sh` and `scripts/run_queens.sh` list the individual calculations. Raw bundle, tensor run and sample records remain in ignored `runs/`. The three result figures in `artifacts/` are tracked at the project owner’s request; generated data and audit output remain local and ignored. Build products, local reference files and environment directories are also excluded from Git.

## Recorded results

- The 10×10 ferromagnet gives `E_min = −180` and `g₀ = 2`.
- The factor-15 spin glass gives `E_min = −299`, `g₀ = 1`, `log Z = 598.317945217` at β=2 and `sc = 5`. The ground-state expectation is about 1455.3 out of 2000 samples; the recorded run observed 1481. `pred extract` decodes a sampled ground state to 3 × 5.
- For equal k-bit factors k=2..8, recorded TreeSA space exponents are 4, 7, 9, 11, 13, 15, 17. The corresponding greedy values and all contraction times are saved in `runs/plan_k*/run.json` and plotted in `artifacts/factoring.png`. Heuristic greedy values need not match the example run.
- N=4..10 board counts are 2, 10, 4, 40, 92, 352, 724. All 400 sampled boards pass row, column and diagonal checks. For N=8, the recorded 200 draws contain 79 distinct boards; for N=6, the four observed frequencies are 40, 56, 52 and 52. These are random draws, not fixed target values. `artifacts/nqueens.png` shows counts, frequencies and the four boards.

## Independent verification

`scripts/audit.py` checks all gate rows, independently enumerates N-queens solutions for N=4..10, checks every sampled board, recomputes all 2000 spin-sample energies directly from the saved bundle, checks `pred extract`, and compares an independent 2×2 exact ferromagnet partition sum with a fresh `tnsample` run. It also regenerates each of the three plots in memory and compares every pixel with the saved PNG. The resulting local report is `artifacts/audit.json`.

`scripts/strict_audit.py` independently verifies all 14 saved factoring contraction trees and their time/space complexity, recomputes the gate and multiplier-cell energies and the N=10 slice counts, and checks plot artists against the saved data. It also tests sampling on an exactly enumerated five-spin system and performs an additional N=6 uniformity check. Its local report is `artifacts/strict-audit.json`.
