"""Run the official temporary Born-to-adjoint Week 5 Part 3C validation."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def run_seismic(binary: Path, experiment: Path, mode: str, output: Path, data: Path | None = None) -> str:
    command = [str(binary), "--experiment", str(experiment), "--mode", mode]
    if data is not None:
        command.extend(["--data", str(data)])
    command.extend(["--out", str(output)])
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout, end="")
    return completed.stdout


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    binary = root / "seismic" / "target" / "release" / "seismic"
    official = root / "inputs" / "reflector.json"
    if not binary.exists():
        raise SystemExit(f"release binary is missing: {binary}")

    with official.open() as handle:
        experiment = json.load(handle)

    with tempfile.TemporaryDirectory(prefix="amat5315-adjoint-") as temporary:
        temporary = Path(temporary)
        born_output = temporary / "born"
        adjoint_output = temporary / "adjoint"
        born_stdout = run_seismic(binary, official, "born", born_output)
        adjoint_stdout = run_seismic(binary, official, "adjoint", adjoint_output, born_output / "born_data.npy")
        if "shot\tmode\tdata_l2_norm" not in born_stdout or "\tborn\t" not in born_stdout:
            raise AssertionError("Born stdout contract failed")
        if "shot\tmode\tdata_l2_norm" not in adjoint_stdout or "\tadjoint\t" not in adjoint_stdout:
            raise AssertionError("adjoint stdout contract failed")

        born = np.load(born_output / "born_data.npy")
        image = np.load(adjoint_output / "image.npy")
        with (adjoint_output / "result.json").open() as handle:
            result = json.load(handle)
        statistics = result["statistics"]

        if born.shape != (3, 240, 14) or born.dtype != np.float64:
            raise AssertionError(f"unexpected born data: {born.shape}, {born.dtype}")
        if not np.isfinite(born).all() or np.all(born == 0.0):
            raise AssertionError("Born data must be finite and nonzero")
        born_sum_squared = float(np.sum(born * born))
        if abs(born_sum_squared - 0.0348479) / 0.0348479 >= 1.0e-3:
            raise AssertionError(f"Born scale differs from the handout reference: {born_sum_squared}")
        if image.shape != (41, 41) or image.dtype != np.float64:
            raise AssertionError(f"unexpected image: {image.shape}, {image.dtype}")
        if not np.isfinite(image).all() or np.all(image == 0.0):
            raise AssertionError("adjoint image must be finite and nonzero")
        if statistics != {
            "storage": "full",
            "checkpoints": None,
            "reverse_calls": 720,
            "scheduler_forward_calls": 720,
            "peak_saved_states": 241,
            "peak_saved_bytes": 6481936,
            "per_shot": [
                {"reverse_calls": 240, "scheduler_forward_calls": 240, "peak_saved_states": 241},
                {"reverse_calls": 240, "scheduler_forward_calls": 240, "peak_saved_states": 241},
                {"reverse_calls": 240, "scheduler_forward_calls": 240, "peak_saved_states": 241},
            ],
        }:
            raise AssertionError(f"unexpected full-history statistics: {statistics}")

        born_files = sorted(path.name for path in born_output.iterdir())
        adjoint_files = sorted(path.name for path in adjoint_output.iterdir())
        if born_files != ["born_data.npy", "result.json", "run.json"]:
            raise AssertionError(f"unexpected Born output files: {born_files}")
        if adjoint_files != ["image.npy", "result.json", "run.json"]:
            raise AssertionError(f"unexpected adjoint output files: {adjoint_files}")

        background_output = temporary / "background-forward"
        background_stdout = run_seismic(binary, official, "forward", background_output)
        if "shot\tmode\tdata_l2_norm" not in background_stdout or "\tforward\t" not in background_stdout:
            raise AssertionError("forward stdout contract failed")
        background_traces = np.load(background_output / "traces.npy")
        background_l2 = float(np.linalg.norm(background_traces))
        background_relative = abs(background_l2 - 11.574770) / 11.574770
        if background_relative >= 1.0e-4:
            raise AssertionError("background forward L2 regression failed")

        left = float(np.sum(born * born))
        perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)
        right = float(np.sum(perturbation * image))
        relative = abs(left - right) / max(abs(left), abs(right))
        if relative >= 1.0e-9:
            raise AssertionError(f"transpose identity failed: left={left}, right={right}, relative={relative}")

        print(f"image_shape={image.shape} image_dtype={image.dtype}")
        print(f"born_sum_squared={born_sum_squared:.15e}")
        print(f"background_forward_l2={background_l2:.15f}")
        print(f"background_forward_l2_relative_error={background_relative:.15e}")
        print(f"image_l2={np.linalg.norm(image):.15e}")
        print(f"image_min={np.min(image):.15e} image_max={np.max(image):.15e}")
        print(f"image_max_abs={np.max(np.abs(image)):.15e}")
        print(f"transpose_left={left:.15e}")
        print(f"transpose_right={right:.15e}")
        print(f"transpose_relative_difference={relative:.15e}")
        print("official adjoint validation: passed")


if __name__ == "__main__":
    main()
