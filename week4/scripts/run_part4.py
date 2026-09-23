"""Run Part 4 convergence experiments with the installed Rust solver."""

import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEEK4 = ROOT / "week4"
ARTIFACTS = WEEK4 / "artifacts" / "part4"
EVIDENCE = WEEK4 / "evidence"


def run_fluid(field_data, directory, method, nu, dt, t_end):
    directory.mkdir(parents=True, exist_ok=True)
    stdout_path = directory.parent / f"{directory.name}.tsv"
    command = ["fluid", "--method", method, "--nu", str(nu), "--dt", str(dt), "--t-end", str(t_end), "--every", str(t_end), "--out", str(directory)]
    result = subprocess.run(command, input=field_data, cwd=WEEK4, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_path.write_bytes(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"fluid failed for {directory}: {result.stderr.decode()}")
    return json.loads((directory / "fields.jsonl").read_text().splitlines()[-1])


def relative_velocity_error(actual, exact):
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual["u"], exact["u"])) + sum((a - b) ** 2 for a, b in zip(actual["v"], exact["v"])))
    denominator = math.sqrt(sum(value * value for value in exact["u"]) + sum(value * value for value in exact["v"]))
    return numerator / denominator


def relative_omega_error(actual, reference):
    numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual["omega"], reference["omega"])))
    denominator = math.sqrt(sum(value * value for value in reference["omega"]))
    return numerator / denominator


def fit_slope(points):
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    x_bar, y_bar = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sum((x - x_bar) ** 2 for x in xs)


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    tg_initial = subprocess.check_output(["field", "taylor-green", "--n", "8"], cwd=WEEK4)
    tg_exact = json.loads(subprocess.check_output(["field", "taylor-green", "--n", "8", "--nu", "0.5", "--t", "2"], cwd=WEEK4))
    tg_errors = []
    for dt in [0.4, 0.25, 0.2]:
        final = run_fluid(tg_initial, ARTIFACTS / f"order/rk4-dt{dt}", "rk4", 0.5, dt, 2.0)
        error = relative_velocity_error(final, tg_exact)
        tg_errors.append({"dt": dt, "relative_velocity_error": error})
    tg_points = [(item["dt"], item["relative_velocity_error"]) for item in tg_errors]
    tg_slope = fit_slope(tg_points)
    tg_reduction = tg_errors[0]["relative_velocity_error"] / tg_errors[-1]["relative_velocity_error"]
    print("Taylor-Green RK4 order study:")
    print("dt\trelative velocity error")
    for item in tg_errors:
        print(f"{item['dt']:.6f}\t{item['relative_velocity_error']:.12e}")
    print(f"Fitted RK4 slope: {tg_slope:.8f}")
    print(f"Error reduction from dt=0.4 to dt=0.2: {tg_reduction:.8f}")
    (EVIDENCE / "order.json").write_text(json.dumps({"dt_errors": tg_errors, "fitted_slope": tg_slope, "error_reduction_0.4_to_0.2": tg_reduction}, indent=2) + "\n")

    random_initial = subprocess.check_output(["field", "random", "--n", "128", "--seed", "2026", "--k-min", "2", "--k-max", "6"], cwd=WEEK4)
    candidate_dts = [0.02, 0.0125, 0.01]
    reference_dt = 0.0025
    reference = run_fluid(random_initial, ARTIFACTS / "convergence/rk4-dt0.0025", "rk4", 0.004, reference_dt, 2.0)
    errors = []
    for dt in candidate_dts:
        final = run_fluid(random_initial, ARTIFACTS / f"convergence/rk4-dt{dt}", "rk4", 0.004, dt, 2.0)
        errors.append({"dt": dt, "relative_omega_error": relative_omega_error(final, reference)})
    points = [(item["dt"], item["relative_omega_error"]) for item in errors]
    slope = fit_slope(points)
    h = 0.01
    dt_values = {item["dt"]: item for item in errors}
    field_2h = json.loads((ARTIFACTS / "convergence/rk4-dt0.02/fields.jsonl").read_text().splitlines()[-1])
    field_h = json.loads((ARTIFACTS / "convergence/rk4-dt0.01/fields.jsonl").read_text().splitlines()[-1])
    norm_difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(field_2h["omega"], field_h["omega"])))
    norm_h = math.sqrt(sum(value * value for value in field_h["omega"]))
    richardson_base = norm_difference / (15.0 * norm_h)
    predicted = [{"dt": dt, "predicted_relative_error": richardson_base * (dt / h) ** 4} for dt in candidate_dts]
    threshold = 5e-6
    qualifying = [item for item in predicted if item["predicted_relative_error"] < threshold]
    selected = max(qualifying, key=lambda item: item["dt"])
    selected_measured = dt_values[selected["dt"]]["relative_omega_error"]
    print("\nRandom-flow self-convergence:")
    print("dt\trelative omega error")
    for item in errors:
        print(f"{item['dt']:.7f}\t{item['relative_omega_error']:.12e}")
    print(f"Fitted self-convergence slope: {slope:.8f}")
    print(f"Error reduction from dt=0.02 to dt=0.01: {errors[0]['relative_omega_error'] / errors[-1]['relative_omega_error']:.8f}")
    print(f"Richardson estimated relative error at dt=0.01: {richardson_base:.12e}")
    for item in predicted:
        print(f"Predicted relative error at dt={item['dt']:.7f}: {item['predicted_relative_error']:.12e}")
    print(f"Selected dt: {selected['dt']:.7f}")
    print(f"Selected predicted error: {selected['predicted_relative_error']:.12e}")
    print(f"Selected measured error: {selected_measured:.12e}")
    result = {
        "grid_size": 128,
        "viscosity": 0.004,
        "final_time": 2.0,
        "seed": 2026,
        "k_min": 2,
        "k_max": 6,
        "method": "rk4",
        "reference_dt": reference_dt,
        "candidate_dt": candidate_dts,
        "measured_relative_omega_errors": errors,
        "fitted_log_log_slope": slope,
        "richardson_base_h": h,
        "richardson_estimated_error_at_h": richardson_base,
        "predicted_errors": predicted,
        "threshold": threshold,
        "selected_dt": selected["dt"],
        "selected_predicted_error": selected["predicted_relative_error"],
        "selected_measured_error": selected_measured,
    }
    (EVIDENCE / "convergence.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
