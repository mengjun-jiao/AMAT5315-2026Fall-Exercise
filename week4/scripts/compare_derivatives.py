"""Compare Rust Fourier and periodic centered finite-difference derivatives."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMMAND = ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "week4" / "Cargo.toml"), "--bin", "derivative_data"]


def main():
    output = subprocess.check_output(COMMAND, cwd=ROOT, text=True)
    print("Derivative             FD error N=32    FD error N=64    Error ratio N32/N64    Fourier error N=32")
    for line in output.splitlines():
        name, fd32, fd64, fourier = line.split(",")
        fd32, fd64, fourier = map(float, (fd32, fd64, fourier))
        print(f"{name:20s} {fd32:16.8e} {fd64:16.8e} {fd32 / fd64:20.8f} {fourier:22.8e}")


if __name__ == "__main__":
    main()
