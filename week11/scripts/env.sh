#!/usr/bin/env bash
# Source from any directory after creating .venv and installing Rust via rustup.
WEEK11_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUSTUP_HOME="$WEEK11_ROOT/.rustup"
export CARGO_HOME="$WEEK11_ROOT/.cargo"
export LIBCLANG_PATH="$("$WEEK11_ROOT/.venv/bin/python" -c 'import clang,pathlib; print(pathlib.Path(clang.__file__).parent / "native")')"
export PATH="$WEEK11_ROOT/.install/bin:$WEEK11_ROOT/.venv/bin:$HOME/.cargo/bin:$PATH"
export RAYON_NUM_THREADS="${WEEK11_THREADS:-8}"
