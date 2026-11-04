# Week 9 — Probabilistic inference by tensor contraction

This project implements the hands-on work in the Week 9 learning sheet. It uses CPU `f64` tensors and exact contraction to answer questions about the ASIA network, compares contraction orders, implements synchronous loopy belief propagation (BP), and runs the UAI 2014 Promedus_14 benchmark. The supplied medical network is a teaching example, not a diagnostic model.

## Files

- `src/`: binary UAI parser and enumeration (`model.rs`), contraction planning and tenferro execution (`plan.rs`), BP (`bp.rs`), and the `infer` command (`main.rs`). Only binary MARKOV models are supported.
- `infer.design.toml`: command and output design; `Cargo.toml` and `Cargo.lock` pin the Rust dependencies.
- `scripts/`: pinned data download, ASIA and Promedus runs, independent NumPy checks, plotting and environment setup.
- `evidence/`: diagrams and plots listed below; `evidence/plans/` holds the reviewed, versionable contraction plans, per-step scalar complexity and search settings.
- `asia.uai`, `promedus/`: downloaded inputs and published answers; `artifacts/`, `runs/`: detailed local results, traces, logs and saved plans. These and the build/install directories are ignored by Git. The original learning sheet is also kept locally.

## Environment and reproduction

The recorded runs used WSL2 Linux x86_64, Rust 1.96.0 and Cargo 1.96.0; `tenferro-tensor`, `tenferro-cpu` and `tenferro-einsum` 0.9.0 (CPU native, no GPU/autodiff); `omeco` 0.2.6; and Python 3.14.4 with NumPy 2.5.4 and Matplotlib 3.11.2. A working `rustup`, C compiler, Python, `curl`, and network access for downloads are prerequisites. Commands below are run from this directory in Bash. `scripts/env.sh` keeps the Rust toolchain and Cargo files local to this project. The `--no-self-update` option avoids changing an existing rustup installation.

```bash
bash scripts/download_data.sh
source scripts/env.sh
rustup toolchain install 1.96.0 --profile minimal --no-self-update
cargo +1.96.0 install --locked --path . --root "$PWD/.install"
cargo +1.96.0 test --release --locked

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/run_asia.sh
bash scripts/run_promedus.sh
.venv/bin/python scripts/verify_asia.py
.venv/bin/python scripts/verify_promedus.py
.venv/bin/python scripts/audit_bp_traces.py
.venv/bin/python scripts/audit_exact_paths.py
.venv/bin/python scripts/export_plan_evidence.py --refresh
.venv/bin/python scripts/export_plan_evidence.py --verify
.venv/bin/python scripts/plot_evidence.py
.venv/bin/python scripts/check_figure_data.py
```

The plotting and verification scripts require NumPy and Matplotlib, not the Rust inference itself. The original recorded plotting and checks used an existing separate Python environment with the versions above. Scripts overwrite their named local outputs on rerun. Rerunning greedy planning can produce a different valid tree, so regenerate the checks, exported plans and plots together after rerunning. The exported plans are snapshots: `--verify` checks their per-step costs and SHA-256 manifest without local run data; running `export_plan_evidence.py` with no options checks them against the current local runs. `--refresh` replaces them with the current local runs. See `evidence/plans/README.md` for their contents and replay limits. To analyze without evaluating tensors, for example:

```bash
infer --model asia.uai --evidence AD --analyze evidence/plans/asia-ad-random/order.json --out runs/analysis
infer --model asia.uai --evidence AD --plan evidence/plans/asia-ad-greedy/order.json --out runs/replay
```

`--order`, `--plan`, and `--analyze` are mutually exclusive. The saved plan includes its scopes, individual retained labels and costs; the analysis/replay checks that input scopes match the conditioned model. Query plans for Promedus are also saved. `run_promedus.sh` sets `RAYON_NUM_THREADS=1`. The time-optimized TreeSA search uses 150 inverse-temperature values beginning at 0.05 in increments of 0.1, 20 iterations per temperature, one trial, greedy initialization, and seed 42. In omeco 0.2.6, the first-trial seed is fixed to 42; this implementation therefore accepts only 42 for TreeSA. Hash-map tie-breaking in the greedy initializer can vary across processes. A seed does not guarantee identical plans on another process or version: use the saved plan for the recorded path.

The downloads use fixed commits, as in the learning sheet: [ASIA model](https://github.com/TensorBFS/TensorInference.jl/tree/935ed86f8e473f523cac9461c45a1f470e74f0f9/examples/asia-network) and [UAI 2014 Promedus mirror](https://github.com/mroavi/uai-2014-inference-competition/tree/22005844423f5bca62798548df808cc418fc0d7d/uai2014). `scripts/download_data.sh` gives the exact paths for model, observations, PR, and MAR.

## ASIA exact inference

The model's state 0 means yes/present/positive. `prior` has no evidence; `AD` fixes A=0 and D=0; `ADX` additionally fixes X=0. The parser keeps the UAI scope order and treats the last axis as fastest varying. In particular, scope `[T,A]` gives `[[0.05,0.01],[0.95,0.99]]`. Enumerating all 256 assignments independently reproduces the contracted probabilities.

| Case | Evidence mass | P(T=yes) | P(L=yes) | P(B=yes) |
| --- | ---: | ---: | ---: | ---: |
| prior | 1 | 0.010400 | 0.055000 | 0.450000 |
| AD | 0.004501375 | 0.087751 | 0.099525 | 0.811402 |
| ADX | 0.00098822675 | 0.391712 | 0.444271 | 0.628822 |

Across the three cases, the largest absolute marginal/joint discrepancy against independent enumeration is at most 2.23e-16; the largest relative evidence-mass discrepancy is about 2.12e-15. With ADX, P(T=yes,L=yes) is 0.022214 while the product of the two marginal probabilities is about 0.174026. The difference reflects dependence; it is not an inference error.

The tenferro matrix product is `[[4,4],[10,8]]`, its trace is 12, and the three-spin Ising partition function is approximately 12.60256209895193. `verify_asia.py` independently compares the matrix product and trace to NumPy and the Ising contraction to all eight spin assignments, printing values and absolute errors; the threshold is 1e-12.

## Plans and belief propagation

For the saved AD scalar plans, random seed 42 costs 200 terms and has a largest input or intermediate of 32 entries; the saved greedy path costs 38 and has a largest tensor of 8 entries. The random plan's largest generated intermediate has 32 entries; the greedy plan's largest generated intermediates have 4 entries, while its largest input has 8. The drawing distinguishes the largest inputs from the largest intermediates. These costs differ from the learning sheet's sample plans: the program slices observed indices and carries scalar evidence factors separately before planning, and pair selection/tie-breaking can differ. The two saved paths give the same evidence mass. `verify_asia.py` independently recounts every step, analyzes one saved plan, and checks replay. Work is the sum over pair contractions of the product of dimensions in their union; it is a model-independent count of terms, not measured FLOPs.

BP initializes all messages uniformly, applies evidence to the factor tables, uses synchronous proposals, and takes 50% of each proposed update. The tolerance is 1e-10, with at most 1,000 sweeps. The residual is the maximum **undamped** proposal difference at the same saved message state as the beliefs. NumPy checks reconstruct the final proposals and beliefs from saved directed messages, and independently repeat every recorded ASIA and Promedus sweep. A second NumPy implementation independently replays ASIA scalar plans and the Promedus TreeSA scalar and all three query plans. The nonuniform tree unit test compares BP with enumeration both with and without observations.

| Case | Sweeps | Final residual | Max absolute marginal error |
| --- | ---: | ---: | ---: |
| prior | 53 | 7.03e-11 | 0.003340 |
| AD | 92 | 9.94e-11 | 0.011407 |
| ADX | 93 | 8.23e-11 | 0.010612 |

All three ASIA runs satisfy the residual stopping tolerance; their nonzero errors show that a fixed point on a loopy graph need not be exact. BP update work counts message-update multiplication and addition operations, excluding normalization, belief computation and the additional final residual probe.

## Promedus_14

This MARKOV model has 414 binary variables and nine observations. Original potential values are preserved. The conditioned scalar factors and logarithmic intermediate scales are retained so that `log10 Z` is meaningful; `Z` itself is the evidence partition sum, not an evidence probability unless divided by the unconditioned partition sum. The saved greedy and TreeSA exact runs agree on `log10 Z` to 1.7e-14 and with published PR (`-7.80675`) within 1e-4. Both entries of each marginal agree with MAR within 1e-5.

| ID | Exact P(state 1) | Published MAR P(state 1) | BP P(state 1) |
| --- | ---: | ---: | ---: |
| 59 | 0.1975882582 | 0.197588 | 0.0000856180 |
| 102 | 0.1497534323 | 0.149753 | 0.0002507605 |
| 139 | 0.6175220184 | 0.617522 | 0.9998749358 |

For the scalar plans in this recorded run, greedy work is 5,492,329,020 with 16,777,216 entries in its largest generated tensor; TreeSA work is 10,538,082 with 131,072 entries. These are measured properties of the saved paths, not optimality claims. The three retained-query paths have their own independently recounted costs in `artifacts/promedus_checks.json`. Exact timings for the scalar plus three separate query contractions, including path searches, were about 6.41 s for greedy and 1.62 s for TreeSA on this run; hardware and paths change timings. BP computed beliefs for all variables in about 0.068 s and 22,538,000 counted update operations, so these timings represent different workloads.

BP reaches its 1,000-sweep limit with undamped residual 2.1163e-10, above 1e-10, and maximum absolute error 0.382353 over both states of the three selected queries. It is therefore **unconverged and inaccurate** for these queries. Independent checks reconstruct its final residual and beliefs from saved messages and verify both exact plans against the published answers.

## Evidence figures

- `evidence/graphs.svg` (`graphs.png` preview): AD tuberculosis query drawn as a tensor network with factor boxes, shared-index junctions, evidence/sum caps and open T index, following the learning sheet.
- `evidence/posteriors.png`: disease probabilities grouped by observation case; bars show enumeration and black marks show tensor contraction.
- `evidence/joint.png`: T,L joint, marginal product and their difference, with shared color scales.
- `evidence/orders.png`: saved AD contraction trees and cost bars, largest inputs (yellow), and largest generated intermediates (orange). Observed indices are sliced before these plans are constructed.
- `evidence/bp_asia.png`: per-sweep ASIA residuals and errors.
- `evidence/bp_adx_comparison.png`: ADX exact/BP probability comparison and convergence/error curves, matching the type of example shown in the learning sheet.
- `evidence/promedus.png`: scalar-plan costs and BP residual/error traces.

Detailed run records and the independently generated `artifacts/independent_checks.json` and `artifacts/promedus_checks.json` remain in the local directory for inspection.
