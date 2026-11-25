# Week 12 — Integer programming and learned branching

This exercise finds the minimum energy of a spin glass and asks how much search is saved by learning which spin to branch on. Rust solves each LP with HiGHS. A CPU JAX model learns from Strong Branching decisions. The model chooses a spin and HiGHS supplies the bounds used for pruning, so all four branching rules retain the same exact optimum.

## Reproduce the experiments

Run the commands below from `week12/`. The recorded environment was Ubuntu 26.04 LTS under WSL2 on an AMD Ryzen 7 H 260, Rust/Cargo 1.96.0, `highs` 2.2.0 (`highs-sys` 1.14.2), `pred` 0.6.0, uv 0.13.10, Python 3.14.4, JAX/jaxlib 0.12.1, NumPy 2.5.5, SciPy 1.18.1 and Matplotlib 3.11.2. HiGHS runs on one CPU thread with presolve disabled. Both MIP gap tolerances are zero.

On a fresh machine install Rust/Cargo, a C/C++ compiler, CMake, libclang, Python and uv. Use the following setup to build the two binaries with the locked crate versions.

```bash
export CARGO_INSTALL_ROOT="$PWD/.install"
export PATH="$PWD/.install/bin:$PWD/.tools/bin:$PATH"
export UV_CACHE_DIR="$PWD/.uv-cache"
export JAX_PLATFORMS=cpu
cargo install problemreductions-cli@0.6.0 --locked
cargo install --path . --locked --force
```

The prepared WSL directory already contains both binaries in `.install/bin/` and uv in `.tools/bin/`. To use them, run the four `export` lines above. The script below lists each phase in experimental order. The exact `branchopt` and `uv` invocations, including every seed and output path, are in [the run script](scripts/run.sh).

```bash
bash scripts/run.sh pred
bash scripts/run.sh gap
bash scripts/run.sh bnb
bash scripts/run.sh labels
bash scripts/run.sh train
bash scripts/run.sh test
bash scripts/run.sh transfer
bash scripts/run.sh figures
bash scripts/run.sh check
```

To review one part as soon as its run finishes, use `bash scripts/run.sh figures --figure gap` after `gap`, `--figure bnb` after `bnb`, `--figure neural_bnb` after `test`, or `--figure eval` after `transfer`. Each command prints the measurements for that part. Running `figures` without arguments builds all four figures and the combined text report. The solver preserves the original Strong Branching timing before the label-export run reuses its directory.

| Phase | Graphs | Seed | Measurements |
|---|---|---:|---|
| Opening examples | Four-spin square and a 100-spin graph with edge probability 0.04 | 2026 for the random graph | `pred` solutions and directly evaluated energies |
| Gap | 16 graphs, 22 spins and 52 bonds | 2026 | LP, MILP, fractional spins and nearest rounding |
| B&B and labels | 48 graphs, 22 spins and 52 bonds | 2026 | Search trees, LP calls, timings and Strong Branching supervision |
| Train | Graph IDs 0–35 train and 36–47 validate | 2026 | 120 epochs, checkpoint and six parity probes |
| Held-out test | 16 graphs, 22 spins and 52 bonds | 3026 | Four policies on new instances |
| Transfer | 8 graphs, 30 spins and 72 bonds | 4026 | The same weights applied at a larger size |

[The solver design](branchopt.design.toml) specifies the SplitMix64 graph generator, energy convention, tolerances, policy scores and feature definitions. [The trainer design](train_branch.design.toml) specifies the optimizer, split and checkpoint rule. The four result PNGs are included in `artifacts/`. The run scripts generate the datasets, model weights, timing records and audit files locally in the same directory.

## 1. Why rounding the relaxation loses the ground state

Write each spin as `s_i = 1 − 2x_i` and each edge disagreement as `z_ij = x_i XOR x_j`. The objective is linear in these variables. Four inequalities per edge enforce XOR for integral spin bits, so the edge variables can stay continuous. Relaxing the spin bits to the interval from 0 to 1 gives a lower bound on the minimum energy.

On the four-spin square the LP sets every spin bit to 0.5 and the disagreements to `(1,1,1,0)`. Its energy is −4. Rounding every bit up makes all spins equal and gives +2. The exact optimum is −2. The 100-spin `pred` instance gives −138. The independent diet LP gives quantities `(89/28, 8/7)` and cost `323/14 = 23.071429`.

On all 16 seeded graphs the LP bound is −52 with 22 fractional spins. The mean exact energy is −38.25, the mean gap is 13.75 and the mean relative gap is 36.45%. All 16 roundings miss the ground state. The mean excess is 34.25 and its maximum is 60 on instance 11. The left panel connects each bound to the corresponding exact energy. The right panel compares its gap and rounding excess. [Gap figure](artifacts/gap.png)

## 2. How Strong Branching changes the search

The depth-first search begins with an empty incumbent. It prunes infeasible LPs and nodes with a lower bound at least the incumbent minus `1e-6`. Most-frac chooses the free spin closest to 0.5 and breaks ties by spin index. Strong Branching probes both children of every candidate and maximizes the product of the two floored bound gains. Trial LPs contribute to the LP count and leave the incumbent untouched.

| Tetrahedron visit | Fixed bits along the branch | LP bound | Action |
|---:|---|---:|---|
| 1 | Root | −6 | Split on spin 0 |
| 2 | x0 = 1 | −4 | Split on spin 1 |
| 3 | x0 = 1, x1 = 1 | −3 | Accept the incumbent (1, 1, 0, 0) |
| 4 | x0 = 1, x1 = 0 | −3 | Prune at the incumbent |
| 5 | x0 = 0 | −2 | Prune while three spins are still fractional |

On 48 instances, most-frac averages 50.875 nodes, 50.875 LP solves and 19.615 ms. Strong Branching averages 16.000 nodes, 287.417 LP solves and 112.176 ms. Its tree is 3.18 times smaller, at a cost of 17.96 LP solves per explored node. The maximum absolute energy difference from the HiGHS MILP is 1.62 × 10⁻¹². Both panels independently sort each policy's observations. The unlabelled Strong Branching run supplies these timings. [B&B figure](artifacts/bnb.png)

## 3. What the network learns

Each candidate has 12 local features from the relaxed spins, fields, degrees, couplings, neighboring spins, bond gaps, LP duals and incumbent gap. A shared `12 → 48 tanh → 1` network scores each candidate. A masked softmax converts the real candidates' scores to probabilities. Ties in the teacher's best score share the target probability. Features use training-only means and standard deviations. Adam runs for 120 epochs in CPU float64 with learning rate 0.004, batch size 64 and a 0.0002 squared-weight penalty. Initialization uses seeded Glorot uniform weights and zero biases. The saved curves show cross-entropy without the penalty.

The 36 training graphs supplied 276 labelled nodes. The 12 validation graphs supplied 84, with zero shared graph IDs. Epoch 85 gave the lowest eligible validation cross-entropy, 1.753257. The maximum absolute Rust/JAX probability difference across six validation probes was 1.94 × 10⁻¹⁶, below the sheet's 10⁻⁴ tolerance.

On the 16 unseen graphs, neural branching cut mean explored nodes by 70.0%. Instance 4 saved the most. Most-frac split spins 0 and 1 first and used 157 nodes in 59.925 ms. Neural branching split spins 4 and 21 first and used 21 nodes in 8.957 ms. Both returned −38. [Neural B&B figure](artifacts/neural_bnb.png)

## 4. Does the speedup transfer?

The online neural policy scores candidates at every branching node. Root-neural scores all spins once, standardizes the root scores, and adds three times the current fractional distance when ranking candidates deeper in the tree. Both evaluation runs record the same model-file SHA-256 digest. The following means include feature extraction, network inference and search management in total wall time. The independent MILP oracle and file I/O run outside the timed search.

| Set | Policy | Nodes | LP solves | Total ms | LP ms | Feature + inference ms |
|---|---|---:|---:|---:|---:|---:|
| 22 spins | Most-frac | 50.875 | 50.875 | 19.414 | 19.359 | 0 |
| 22 spins | Neural | 15.250 | 15.250 | 6.494 | 6.307 | 0.1495 |
| 22 spins | Root-neural | 15.500 | 15.500 | 5.928 | 5.877 | 0.0229 |
| 22 spins | Strong | 15.625 | 287.750 | 113.288 | 113.179 | 0 |
| 30 spins | Most-frac | 177.750 | 177.750 | 97.121 | 96.947 | 0 |
| 30 spins | Neural | 39.750 | 39.750 | 22.698 | 22.149 | 0.4923 |
| 30 spins | Root-neural | 50.750 | 50.750 | 26.903 | 26.796 | 0.0363 |
| 30 spins | Strong | 38.000 | 899.250 | 506.178 | 505.855 | 0 |

All non-Strong policies used exactly one LP per visited node on every test graph. On 30 spins, online and root-neural search used 77.6% and 71.4% fewer nodes than most-frac. They ran 22.3 and 18.8 times faster than Strong Branching on this machine. The largest energy difference from HiGHS over both test sets was 2.02 × 10⁻¹². Training initialization influences learned decisions, and wall times depend on the CPU. The learning sheet reports validation cross-entropy 1.759 and mean online-neural trees of 15.4 and 39.5 nodes in its example run. [Transfer and timing figure](artifacts/eval.png)

The transfer timing panel prints the feature and inference components for the two neural rules above their total-time labels. Their measured segments are shorter than one pixel at the worksheet's common linear scale. Most-frac and Strong Branching spend zero time on neural inference.

## Verification and local evidence

Running the `figures` phase writes `artifacts/report.txt` with the five visits, test-instance comparison, time breakdowns and error measurements. [The numerical audit](scripts/check_results.py) independently regenerates every seeded graph and computes exact energies by integer min-sum variable elimination on all 72 distinct benchmark graphs. It also verifies the 12 features at all 360 training and validation nodes, re-solves the 13,028 Strong Branching trial LPs using a separate product-variable formulation in SciPy, and recomputes the checkpoint cross-entropy and six probes in NumPy. [The figure audit](scripts/check_figures.py) checks every numerical series, reproduces the PNG pixels and compares the printed report with the saved runs. The optional reading on semidefinite relaxation, Goemans–Williamson rounding and optimal branching has no build task.
