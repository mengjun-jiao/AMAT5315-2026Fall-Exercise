# Week 3: Two-Dimensional Ising Sampling

This week studies a two-dimensional ferromagnetic Ising model on an `L x L`
periodic square lattice. Spins are `+1` or `-1`, the coupling is `J = 1`, and
there is no external field. Temperatures are in units of `J`.

The command-line program `ising` implements two update rules:

- Metropolis: one step is `L^2` independently and uniformly selected
  single-spin proposals with replacement. Its time unit is `sweep`.
- Wolff: one step grows and flips one aligned cluster. Its time unit is
  `cluster_flip`, and every flip is accepted.

For both algorithms, the first temperature starts from an all-up lattice. The
final lattice at one temperature is inherited by the next temperature, and one
seeded random stream is carried through the complete temperature ramp.

In `series.jsonl`, `M` is the signed mean spin and `E` is the energy per site.
`sweep` starts at one after discard at each temperature. Wolff rows also have
`cluster_size`. In `spins.jsonl`, `m` is the signed mean spin and `sweep` is a
cumulative move count across the full ramp, including discard. The contract is
in [`ising.design.toml`](ising.design.toml).

## Clean environment setup

The Rust crate requires Rust/Cargo with edition 2021 support. Rust 1.85 or
newer is sufficient; the development environment used here was stable Rust
1.98.1. Python 3.10 or newer with `venv` and `pip` is required for the analysis
scripts. The Python dependencies are only NumPy and Matplotlib, listed in
[`requirements.txt`](requirements.txt).

From a clean clone, run:

```bash
cd week3
cargo test
cargo install --path . --locked --quiet
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The install command places `ising` in Cargo's user bin directory. Alternatively,
all simulation commands below can use `cargo run --` from `week3/`, which does
not require modifying `PATH`.

## Reproduction order

All commands in this section are run from `week3/`. Generated `runs/` and
`artifacts/` directories are local inputs/outputs and are ignored by Git.

### Part 1: contract and manual checks

The following short checks were used for the measured Part 1 evidence:

```bash
cargo run -- --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 --discard 2000 --measure 2000 --seed 2026 --out runs/T1.8
cargo run -- --update metropolis --l 64 --t-from 3.0 --t-to 3.0 --t-step 0.05 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.0
cargo run -- --update metropolis --l 64 --t-from 3.1 --t-to 3.1 --t-step 0.05 --discard 2000 --measure 2000 --seed 2026 --out runs/T3.1
```

The Boltzmann plot is generated from the T=3.0 and T=3.1 files:

```bash
.venv/bin/python plot_boltzmann.py
```

The two seed-2026 runs were compared byte-for-byte, then a seed-2027 run was
used as the different-seed check:

```bash
cargo run -- --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 --discard 2000 --measure 2000 --seed 2026 --out runs/a
cargo run -- --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 --discard 2000 --measure 2000 --seed 2026 --out runs/b
cargo run -- --update metropolis --l 64 --t-from 1.8 --t-to 1.8 --t-step 0.05 --discard 2000 --measure 2000 --seed 2027 --out runs/c
cmp runs/a/series.jsonl runs/b/series.jsonl
cmp --silent runs/a/series.jsonl runs/c/series.jsonl && echo "unexpected equality" || true
```

The recording used for the viewer frames was:

```bash
cargo run -- --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.05 --discard 2000 --measure 200 --every 20 --seed 2026 --out runs/ramp
```

This produces 41 temperatures and 10 frames per temperature. The committed
`spins.jsonl` is the resulting 410-frame recording.

### Part 2: Metropolis scans

```bash
cargo run -- --update metropolis --l 32 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 1042 --out artifacts/coarse-l32
cargo run -- --update metropolis --l 64 --t-from 1.5 --t-to 3.5 --t-step 0.1 --discard 2000 --measure 5000 --seed 42 --out artifacts/coarse-l64
cargo run -- --update metropolis --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 1042 --out artifacts/window-l32
cargo run -- --update metropolis --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 2000 --measure 100000 --seed 42 --out artifacts/window-l64
```

The Part 2 analysis uses window data inside 2.0--2.6 and coarse data outside
that interval. It does not concatenate chains.

```bash
.venv/bin/python scripts/plot_part2.py
.venv/bin/python scripts/peaks.py
.venv/bin/python scripts/errors.py
.venv/bin/python scripts/bootstrap.py
```

`bootstrap.py` uses 500 replicates at block lengths 2000, 4000, and 8000,
root seed 20260916, and circular moving blocks. It reads each source file once
per process. A replicate draws `ceil(N/B)` consecutive circular blocks and
truncates their concatenation to exactly 100000 rows.

### Part 4: Wolff scans

These are the formal Wolff window scans used for the committed comparison. They
are intentionally not rerun by the documentation step.

```bash
cargo run -- --update wolff --l 32 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 1042 --out artifacts/wolff-l32
cargo run -- --update wolff --l 64 --t-from 2.0 --t-to 2.6 --t-step 0.05 --discard 20000 --measure 100000 --seed 42 --out artifacts/wolff-l64
```

The default comparison command reproduces the sampler-agreement and Wolff
critical-temperature report and image. It performs the 500-replicate,
2000/4000/8000 circular moving-block bootstrap independently for each
algorithm and size:

```bash
.venv/bin/python scripts/compare.py
```

The efficiency-only command avoids that bootstrap and only computes the
work-normalized autocorrelation comparison:

```bash
.venv/bin/python scripts/compare.py --efficiency
```

The distinction matters: `tau_work` is the plotted quantity. The corresponding
effective-sample relation is `n_eff = n / (2*tau)`, so `2*tau_work` appears in
that relation. The ratio in `work-compare.txt` compares spin-update work, not
wall-clock runtime: Metropolis counts proposals including rejected proposals,
whereas Wolff counts flipped spins through
`tau_moves * mean(cluster_size) / L^2`.

## Analysis scripts and committed evidence

The scripts are portable: paths are based on the script/repository location,
and the Python executable is supplied by the caller. Their external imports are
only NumPy and Matplotlib.

| Committed file | Reproduction command or manual procedure |
| --- | --- |
| `evidence/boltzmann.png` | From `week3/`, run `.venv/bin/python plot_boltzmann.py` using `runs/T3.0/` and `runs/T3.1/`. |
| `evidence/magnetization.png` | `.venv/bin/python scripts/plot_part2.py` |
| `evidence/susceptibility.png` | `.venv/bin/python scripts/plot_part2.py` |
| `evidence/peaks.txt` | `.venv/bin/python scripts/peaks.py` |
| `evidence/errors.txt` | `.venv/bin/python scripts/errors.py` |
| `evidence/trace.png` | `.venv/bin/python scripts/errors.py` |
| `evidence/acf-binning.png` | `.venv/bin/python scripts/errors.py` |
| `evidence/tau.png` | `.venv/bin/python scripts/errors.py` |
| `evidence/chi-bootstrap.png` | `.venv/bin/python scripts/bootstrap.py` |
| `evidence/bootstrap.txt` | `.venv/bin/python scripts/bootstrap.py` |
| `evidence/magnetization-compare.png` | `.venv/bin/python scripts/compare.py` |
| `evidence/sampler-compare.txt` | `.venv/bin/python scripts/compare.py` |
| `evidence/tau-compare.png` | `.venv/bin/python scripts/compare.py --efficiency` |
| `evidence/work-compare.txt` | `.venv/bin/python scripts/compare.py --efficiency` |
| `evidence/viewer-T1.8.png` | Manual course-player export described below. |
| `evidence/viewer-T2.3.png` | Manual course-player export described below. |
| `evidence/viewer-T3.0.png` | Manual course-player export described below. |

The committed `spins.jsonl` is produced by the Part 1 recording command above.
The source code and tests are built with `cargo fmt`, `cargo test`, and
`cargo clippy --all-targets --all-features -- -D warnings` from `week3/`.

### Manual viewer screenshots

The course viewer is:

<https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html>

The recording URL loaded by the viewer is:

<https://raw.githubusercontent.com/mengjun-jiao/AMAT5315-2026Fall-Exercise/main/week3/spins.jsonl>

The manual procedure is:

1. Open the course player and load the recorded `spins.jsonl` data.
2. Move to the complete temperature-marked frames at `T=1.8`, `T=2.3`, and
   `T=3.0`.
3. Confirm the displayed magnetizations: approximately `m=0.9458`, `0.3442`,
   and `0.0571`, respectively.
4. Use the player's `Save PNG` control, then copy the exported files into
   `week3/evidence/` and rename them exactly `viewer-T1.8.png`,
   `viewer-T2.3.png`, and `viewer-T3.0.png`.

Python does not recreate these screenshots. They are browser exports from the
course player and remain separate from the reproducible analysis plots.

## Results and uncertainty limits

The latest reports give the following values:

- Metropolis two-size estimate: `Tc = 2.27201664`. Its circular block-bootstrap
  SEs are `0.00753755`, `0.00743502`, and `0.00761892` for block lengths 2000,
  4000, and 8000; their cross-block-length summary is `0.00753049`, and the
  three values satisfy the stated +/- 10% stability rule.
- Metropolis `L=64`, `T=2.3` mean-absolute-magnetization error remains
  `sampling error unresolved` under the Part 3 binning-platform check.
- Sampler agreement at `T=2.3` is `agreement provisional`, because the
  Metropolis errors are not stable across block lengths even though the three
  combined-error `d` values are below 3.
- Wolff two-size estimate: `Tc = 2.27449286`, relative error `0.233690%`.
  Wolff `Tc` bootstrap SEs are `0.00063394`, `0.00061452`, and `0.00058444`;
  their cross-block-length summary is `0.00061097` and is stable by the +/- 10%
  rule.
- At `L=64`, `T=2.3`, the work ratio is `R = 482.905469` (about `482.9`).
  The reported values are `tau_moves=4.704514`,
  `mean_cluster_size=938.893610`, `Metropolis tau_work=520.754893`, and
  `Wolff tau_work=1.078379`.

Bootstrap standard errors are sampling-error estimates, not 95% confidence
intervals. They exclude finite-size extrapolation bias and five-point
quadratic-fit bias. The work ratio is not a wall-clock speedup. The
work-normalized tau comparison has no additional error bars in this work.

## Acceptance status

### Student-run checks

- Part 1 temperature checks, seed checks, the temperature ramp recording, and
  the specified Part 1 verification values were run by the repository owner.
- The Part 4 Metropolis and Wolff formal window simulations were run by the
  repository owner and copied into the local `artifacts/` directories.

### Automated checks

- Rust unit and CLI tests pass, including Metropolis and Wolff behavior,
  periodic boundaries, L=2 handling, deterministic seeds, metadata, row
  counts, and cluster sizes.
- Python analysis scripts were actually run against the local artifacts.
- PNG files were opened and checked for valid image data and readable layouts.

### Manual image checks

The repository owner manually checked the Boltzmann, Part 2, Part 3, sampler
comparison, work-normalized efficiency, and viewer images reported in the
conversation. The three viewer PNGs are manual browser exports.

The incognito-window check is **pending**; it is not claimed as passed. The full
clean-clone reproduction from the top of this README was executed in an
isolated temporary clone on 2026-09-16. The simulation, analysis, Rust test,
and Clippy commands completed there; the generated data and build products
were not copied back or committed. The Extension and Neural Sampling
Challenge are not complete and are not claimed here.

## Repository and contract boundaries

The local `ising.design.toml` was copied from the WSL Desktop attachment and
checked against page 3 of the learning sheet. The official reference named on
page 17 is available at:

<https://giggleliu.github.io/AMAT5315-2026Fall/downloads/week3-ising.design.toml>

That file was downloaded to a temporary path and compared section by section
and field by field with the local contract. The semantic contract matches; the
official copy only differs in alignment whitespace and a trailing blank line.
The local contract was not overwritten.

The following are intentionally local and not committed:

- `week3/artifacts/` and `week3/runs/` raw simulation data;
- `week3/target/` Rust build output and Python/cache directories;
- `week2/week2-viewer.html`, an existing untracked user file.

All committed Week 3 files are below 5 MB. `week3/evidence/` and
`week3/spins.jsonl` are tracked. The raw artifacts and runs remain available in
the local clone for analysis but are excluded from the submitted tree.

## Independent clean-clone preparation

The following creates a new temporary clone from the current local repository
without overwriting any existing directory. It does not run the full scans:

```bash
cd /home/mengjun
clone_dir=$(mktemp -d /tmp/amat5315-week3-clean.XXXXXX)
git clone /home/mengjun/AMAT5315-2026Fall-Exercise "$clone_dir"
cd "$clone_dir/week3"
git status --short
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cargo test
```

After this preparation, follow the reproduction commands above in order. The
clean-clone run described above is a completed local verification; repeat it
in a new temporary directory when an independently fresh verification is
needed.
