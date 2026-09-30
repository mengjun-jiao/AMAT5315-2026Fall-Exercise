"""Validate official Part 2 forward and recording evidence."""

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "artifacts" / "forward"
    with (output_dir / "run.json").open() as handle:
        run = json.load(handle)
    with (output_dir / "result.json").open() as handle:
        result = json.load(handle)
    traces = np.load(output_dir / "traces.npy")
    wavefield = np.load(output_dir / "wavefield.npy")
    echo = np.load(output_dir / "echo.npy")
    experiment = run["experiment"]

    if traces.shape != (3, 240, 14) or traces.dtype != np.float64:
        raise AssertionError(f"unexpected traces array: {traces.shape}, {traces.dtype}")
    global_l2 = float(np.linalg.norm(traces))
    global_reference = 11.574770
    global_relative_error = abs(global_l2 - global_reference) / global_reference
    if global_relative_error >= 1.0e-4:
        raise AssertionError(f"global L2 relative error too large: {global_relative_error}")

    print(f"traces_shape={traces.shape} traces_dtype={traces.dtype}")
    print(f"global_l2={global_l2:.12f}")
    print(f"global_l2_relative_error={global_relative_error:.12e}")

    expected_maxima = [0.60809514, 0.59271397, 0.60809514]
    for shot_index, source in enumerate(experiment["shots"]):
        shot = traces[shot_index]
        step_index, receiver_index = np.unravel_index(np.argmax(np.abs(shot)), shot.shape)
        maximum = float(shot[step_index, receiver_index])
        receiver = experiment["receivers"][receiver_index]
        time_reduced = (step_index + 1) * experiment["dt"]
        time_physical = time_reduced * experiment["time_unit_s"]
        relative_error = abs(abs(maximum) - expected_maxima[shot_index]) / expected_maxima[shot_index]
        if relative_error >= 1.0e-4 or step_index != 83:
            raise AssertionError(
                f"shot {shot_index} maximum failed: {abs(maximum)}, index {step_index}"
            )
        if abs(receiver[0] - source[0]) != 1 or receiver[1] != source[1]:
            raise AssertionError(f"shot {shot_index} maximum receiver is not one cell from source")
        print(
            f"shot={shot_index} source={source} max_abs={abs(maximum):.12f} "
            f"trace_index={step_index} reduced_time={time_reduced:.12f} "
            f"physical_time={time_physical:.12f} receiver_index={receiver_index} receiver={receiver}"
        )

    if result != {
        "mode": "forward",
        "nx": 41,
        "nz": 41,
        "dx": 1.0,
        "dt": 0.2,
        "steps": 240,
        "shots": [[10, 8], [20, 8], [30, 8]],
        "receivers": [[7, 8], [9, 8], [11, 8], [13, 8], [15, 8], [17, 8], [19, 8], [21, 8], [23, 8], [25, 8], [27, 8], [29, 8], [31, 8], [33, 8]],
    }:
        raise AssertionError("result.json does not match the forward contract")

    recording = run.get("recording")
    if recording is None or recording["every"] != 3:
        raise AssertionError("recording metadata is missing or incorrect")
    expected_steps = list(range(0, 241, 3))
    if recording["steps"] != expected_steps or len(recording["times"]) != 81:
        raise AssertionError("recording steps or times are incorrect")
    if abs(recording["times"][50] - 30.0) >= 1.0e-12:
        raise AssertionError("step-150 reduced time is incorrect")
    if wavefield.shape != (81, 41, 41) or wavefield.dtype != np.float32:
        raise AssertionError(f"unexpected wavefield array: {wavefield.shape}, {wavefield.dtype}")
    if echo.shape != (81, 41, 41) or echo.dtype != np.float32:
        raise AssertionError(f"unexpected echo array: {echo.shape}, {echo.dtype}")

    frame = 50
    background_max = float(np.max(np.abs(wavefield[frame])))
    echo_max = float(np.max(np.abs(echo[frame])))
    print(f"wavefield_shape={wavefield.shape} wavefield_dtype={wavefield.dtype}")
    print(f"echo_shape={echo.shape} echo_dtype={echo.dtype}")
    print("recording_every=3 frame_count=81 first_step=0 last_step=240 step_150_frame=50")
    print("step_150_reduced_time=30.000000000000 physical_time=3.000000000000")
    print(f"step_150_background_max_abs={background_max:.12f}")
    print(f"step_150_echo_max_abs={echo_max:.12f}")
    print(f"step_150_echo_background_ratio={echo_max / background_max:.12f}")
    print("forward evidence validation: passed")


if __name__ == "__main__":
    main()
