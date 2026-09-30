"""Run official Week 5 Born and whole-trajectory finite-difference validation."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_seismic(binary: Path, experiment: Path, mode: str, output: Path) -> None:
    subprocess.run(
        [str(binary), "--experiment", str(experiment), "--mode", mode, "--out", str(output)],
        check=True,
        text=True,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    binary = root / "seismic" / "target" / "release" / "seismic"
    official = root / "inputs" / "reflector.json"
    if not binary.exists():
        raise SystemExit(f"release binary is missing: {binary}")

    with official.open() as handle:
        base = json.load(handle)
    background = np.asarray(base["background"], dtype=np.float64)
    perturbation = np.asarray(base["perturbation"], dtype=np.float64)

    with tempfile.TemporaryDirectory(prefix="amat5315-born-") as temporary:
        temporary = Path(temporary)
        born_output = temporary / "born"
        run_seismic(binary, official, "born", born_output)
        born = np.load(born_output / "born_data.npy")
        if born.shape != (3, 240, 14) or born.dtype != np.float64:
            raise AssertionError(f"unexpected born_data: {born.shape}, {born.dtype}")
        if not np.isfinite(born).all() or np.all(born == 0.0):
            raise AssertionError("born_data must be finite and nonzero")

        print(f"born_data_shape={born.shape} born_data_dtype={born.dtype}")
        print(f"global_born_l2={np.linalg.norm(born):.15e}")
        born_sum_squared = float(np.sum(born * born))
        print(f"born_sum_squared={born_sum_squared:.15e}")
        if abs(born_sum_squared - 0.0348479) / 0.0348479 >= 1.0e-3:
            raise AssertionError(f"Born scale differs from the handout reference: {born_sum_squared}")
        for shot, norm in enumerate(np.linalg.norm(born, axis=(1, 2))):
            print(f"born_shot={shot} l2={norm:.15e}")
        index = np.unravel_index(np.argmax(np.abs(born)), born.shape)
        print(
            "born_max_abs="
            f"{abs(born[index]):.15e} shot={index[0]} step={index[1]} receiver={index[2]}"
        )

        background_output = temporary / "background-forward"
        run_seismic(binary, official, "forward", background_output)
        background_traces = np.load(background_output / "traces.npy")
        background_l2 = float(np.linalg.norm(background_traces))
        background_relative = abs(background_l2 - 11.574770) / 11.574770
        print(f"background_forward_l2={background_l2:.15f}")
        print(f"background_forward_l2_relative_error={background_relative:.15e}")
        if background_relative >= 1.0e-4:
            raise AssertionError("background forward L2 regression failed")

        born_norm = np.linalg.norm(born)
        finite_difference_errors = []
        for eps in (1.0e-4, 1.0e-5, 1.0e-6):
            plus_payload = dict(base)
            plus_payload["background"] = (background + eps * perturbation).tolist()
            minus_payload = dict(base)
            minus_payload["background"] = (background - eps * perturbation).tolist()
            plus_experiment = temporary / f"plus-{eps:.0e}.json"
            minus_experiment = temporary / f"minus-{eps:.0e}.json"
            plus_experiment.write_text(json.dumps(plus_payload))
            minus_experiment.write_text(json.dumps(minus_payload))
            plus_output = temporary / f"plus-{eps:.0e}"
            minus_output = temporary / f"minus-{eps:.0e}"
            run_seismic(binary, plus_experiment, "forward", plus_output)
            run_seismic(binary, minus_experiment, "forward", minus_output)
            finite_difference = (
                np.load(plus_output / "traces.npy") - np.load(minus_output / "traces.npy")
            ) / (2.0 * eps)
            difference = born - finite_difference
            maximum = float(np.max(np.abs(difference)))
            relative = float(np.linalg.norm(difference) / born_norm)
            finite_difference_errors.append(relative)
            print(
                f"eps={eps:.1e} max_abs_difference={maximum:.15e} "
                f"relative_l2_error={relative:.15e}"
            )

        if any(error >= 1.0e-6 for error in finite_difference_errors):
            raise AssertionError(f"whole-trajectory finite-difference error is too large: {finite_difference_errors}")
        if not all(left < right for left, right in zip(finite_difference_errors, finite_difference_errors[1:])):
            raise AssertionError(f"finite-difference errors did not converge with epsilon: {finite_difference_errors}")

        print("official Born validation: passed")


if __name__ == "__main__":
    main()
