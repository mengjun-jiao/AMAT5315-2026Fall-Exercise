"""Run the Part 3 Rust simulations and record English diagnostics."""

import json
import math
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEEK4 = ROOT / "week4"
ARTIFACTS = WEEK4 / "artifacts"


def field_command(kind, n, extra):
    return ["field", kind, "--n", str(n), *extra]


def generate_field(command, destination):
    data = subprocess.check_output(command, cwd=WEEK4)
    destination.write_bytes(data)
    return data


def run_fluid(field_data, output_dir, stdout_path, method, nu, dt, t_end, every, expected_failure=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "fluid", "--method", method, "--nu", str(nu), "--dt", str(dt),
        "--t-end", str(t_end), "--every", str(every), "--out", str(output_dir),
    ]
    result = subprocess.run(command, input=field_data, cwd=WEEK4, text=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_path.write_bytes(result.stdout)
    if result.returncode != 0 and not expected_failure:
        raise RuntimeError(f"unexpected fluid failure for {output_dir}: {result.stderr.decode()}")
    return result.returncode


def diagnostic_rows(path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 3:
            rows.append(tuple(float(value) for value in fields))
    return rows


def run_scan(field_data, prefix, method, dt, t_end=10.0, every=0.5):
    output_dir = ARTIFACTS / "scan" / f"{prefix}-{method}-dt-{dt:.3f}"
    stdout_path = output_dir.parent / f"{output_dir.name}.tsv"
    code = run_fluid(field_data, output_dir, stdout_path, method, 0.004, dt, t_end, every, expected_failure=True)
    rows = diagnostic_rows(stdout_path)
    return output_dir, stdout_path, code, rows


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "scan").mkdir(exist_ok=True)
    (ARTIFACTS / "sensitivity").mkdir(exist_ok=True)
    (ARTIFACTS / "unstable").mkdir(exist_ok=True)

    random_command = field_command("random", 128, ["--seed", "2026", "--k-min", "2", "--k-max", "6"])
    random_data = generate_field(random_command, ARTIFACTS / "random-initial.json")
    random_code = run_fluid(random_data, ARTIFACTS / "random", ARTIFACTS / "random.tsv", "rk4", 0.004, 0.01, 10.0, 0.1)
    random_rows = diagnostic_rows(ARTIFACTS / "random.tsv")
    print(f"Random reference exit code: {random_code}")
    print(f"Random reference first row: {random_rows[0]}")
    print(f"Random reference final row: {random_rows[-1]}")

    tg_data = generate_field(field_command("taylor-green", 64, []), ARTIFACTS / "taylor-green-initial.json")
    unstable_dir = ARTIFACTS / "unstable" / "taylor-green"
    unstable_code = run_fluid(tg_data, unstable_dir, ARTIFACTS / "unstable" / "taylor-green.tsv", "rk4", 0.1, 0.04, 4.0, 0.1, expected_failure=True)
    unstable_rows = diagnostic_rows(ARTIFACTS / "unstable" / "taylor-green.tsv")
    print(f"Taylor-Green dt=0.04 exit code: {unstable_code}; final diagnostic row: {unstable_rows[-1]}")

    diff_bound = 2.785 / (0.1 * 2 * 21**2)
    print(f"Predicted Taylor-Green diffusive stability limit: {diff_bound:.12f}")
    for dt in [0.032, 0.033]:
        directory = ARTIFACTS / "scan" / f"taylor-green-dt-{dt:.3f}"
        stdout_path = directory.parent / f"{directory.name}.tsv"
        code = run_fluid(tg_data, directory, stdout_path, "rk4", 0.1, dt, 8.0, 0.5, expected_failure=True)
        rows = diagnostic_rows(stdout_path)
        print(f"Taylor-Green dt={dt:.3f}: exit code {code}, last finite/diagnostic time {rows[-1][0]:.15g}")

    random_json = json.loads(random_data)
    u, v = random_json["u"], random_json["v"]
    umax = max(math.hypot(a, b) for a, b in zip(u, v))
    advective_bound = 2.83 / (umax * math.sqrt(2.0) * 42.0)
    print(f"Random initial Umax: {umax:.12f}")
    print(f"Predicted random-flow advective bound: {advective_bound:.12f}")

    scan_results = []
    for dt in [0.038, 0.040]:
        result = run_scan(random_data, "random", "rk4", dt)
        scan_results.append(result)
        print(f"Random RK4 dt={dt:.3f}: exit code {result[2]}, last time {result[3][-1][0]:.15g}")
    if all(result[2] == 0 for result in scan_results):
        for dt in [0.045, 0.050, 0.055, 0.060, 0.070]:
            result = run_scan(random_data, "random", "rk4", dt)
            scan_results.append(result)
            print(f"Random RK4 dt={dt:.3f}: exit code {result[2]}, last time {result[3][-1][0]:.15g}")
            if result[2] != 0:
                break
    elif all(result[2] != 0 for result in scan_results):
        for dt in [0.035, 0.032, 0.030, 0.028, 0.025]:
            result = run_scan(random_data, "random", "rk4", dt)
            scan_results.append(result)
            print(f"Random RK4 dt={dt:.3f}: exit code {result[2]}, last time {result[3][-1][0]:.15g}")
            if result[2] == 0:
                break
    euler_result = run_scan(random_data, "random", "euler", 0.01)
    print(f"Random Euler dt=0.010: exit code {euler_result[2]}, last time {euler_result[3][-1][0]:.15g}")

    tg_perturbed = subprocess.check_output(["perturb_field"], input=tg_data, cwd=WEEK4)
    random_perturbed = subprocess.check_output(["perturb_field"], input=random_data, cwd=WEEK4)
    pairs = [
        ("taylor-green-original", tg_data),
        ("taylor-green-perturbed", tg_perturbed),
        ("random-original", random_data),
        ("random-perturbed", random_perturbed),
    ]
    for name, data in pairs:
        directory = ARTIFACTS / "sensitivity" / name
        stdout_path = directory.parent / f"{directory.name}.tsv"
        code = run_fluid(data, directory, stdout_path, "rk4", 0.004 if name.startswith("random") else 0.1, 0.01, 20.0, 0.5)
        print(f"Sensitivity run {name}: exit code {code}")

    (ARTIFACTS / "part3-parameters.json").write_text(json.dumps({"diffusive_bound": diff_bound, "umax": umax, "advective_bound": advective_bound}, indent=2) + "\n")


if __name__ == "__main__":
    main()
