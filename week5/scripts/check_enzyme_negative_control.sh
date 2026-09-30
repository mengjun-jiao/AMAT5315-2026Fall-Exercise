#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
temporary_dir=$(mktemp -d /tmp/amat5315-enzyme-negative.XXXXXX)
trap 'rm -rf "$temporary_dir"' EXIT

if rustc +nightly-2026-09-05 \
    --edition=2024 \
    --crate-name enzyme_kernel \
    --crate-type staticlib \
    -C opt-level=3 \
    -C lto=fat \
    -C panic=abort \
    "$repo_root/week5/seismic/src/kernel.rs" \
    -o "$temporary_dir/libenzyme_kernel.a" \
    >"$temporary_dir/stdout" 2>"$temporary_dir/stderr"; then
    echo "negative control unexpectedly succeeded" >&2
    exit 1
fi

if ! grep -Fq "using the autodiff feature requires -Z autodiff=Enable" "$temporary_dir/stderr"; then
    echo "negative control failed for an unexpected reason:" >&2
    sed -n '1,40p' "$temporary_dir/stderr" >&2
    exit 1
fi

echo "negative control passed: removing -Zautodiff=Enable fails as expected"
