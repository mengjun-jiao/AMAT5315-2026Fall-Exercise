#!/usr/bin/env bash
# Source from any working directory; keep build caches and tools under week10.
WEEK10_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUSTUP_HOME="${WEEK10_RUSTUP_HOME:-$WEEK10_ROOT/.rustup}"
export CARGO_HOME="$WEEK10_ROOT/.cargo"
export PATH="$HOME/.cargo/bin:$WEEK10_ROOT/.install/bin:$WEEK10_ROOT/.venv/bin:$PATH"
export OPENBLAS_NUM_THREADS="${WEEK10_BLAS_THREADS:-1}"
export WEEK10_OPENBLAS="$("$WEEK10_ROOT/.venv/bin/python" -c 'import pathlib,numpy; p=pathlib.Path(numpy.__file__).parent.parent/"numpy.libs"; print(next(p.glob("libscipy_openblas64_*.so")))')"
