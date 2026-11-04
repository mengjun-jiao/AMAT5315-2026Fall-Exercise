#!/usr/bin/env bash
# Source this file from the week9 directory (or source by path).
WEEK9_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUSTUP_HOME="$WEEK9_ROOT/.rustup"
export CARGO_HOME="$WEEK9_ROOT/.cargo"
export PATH="$HOME/.cargo/bin:$WEEK9_ROOT/.install/bin:$PATH"
