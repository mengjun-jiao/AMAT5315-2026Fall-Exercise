"""Re-evaluate saved contraction plans independently with NumPy."""
from pathlib import Path
import itertools
import json
import math
import numpy as np
from audit_bp_traces import read_model

ROOT = Path(__file__).resolve().parent.parent


def condition(scopes, tables, observations):
    factors = []
    log_scale = 0.
    for scope, table in zip(scopes, tables):
        selection = tuple(observations.get(v, slice(None)) for v in scope)
        data = np.asarray(table[selection])
        remaining = [v for v in scope if v not in observations]
        if not remaining:
            assert data > 0
            log_scale += math.log(float(data))
        else:
            factors.append((remaining, data))
    return factors, log_scale


def contract(plan_path, scopes, tables, observations):
    plan = json.loads(plan_path.read_text())
    factors, log_scale = condition(scopes, tables, observations)
    assert plan["input_scopes"] == [v for v, _ in factors]
    live = {}
    for i, (variables, data) in enumerate(factors):
        scale = float(np.max(data))
        assert scale > 0
        live[i] = (variables, data / scale)
        log_scale += math.log(scale)
    for step in plan["steps"]:
        inputs = [live.pop(i) for i in step["inputs"]]
        labels = sorted({x for variables, _ in inputs for x in variables})
        letters = dict(zip(labels, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        eq = ",".join("".join(letters[x] for x in variables) for variables, _ in inputs)
        eq += "->" + "".join(letters[x] for x in step["retained_labels"])
        data = np.asarray(np.einsum(eq, *(array for _, array in inputs), optimize=False))
        scale = float(np.max(data))
        assert scale > 0
        live[step["output"]] = (step["retained_labels"], data / scale)
        log_scale += math.log(scale)
    assert len(live) == 1
    variables, result = next(iter(live.values()))
    assert variables == plan["query"]
    return result, log_scale


def enumerate_asia(scopes, tables, obs):
    z = 0.
    marginals = np.zeros((8, 2))
    joint = np.zeros((2, 2))
    for bits in itertools.product(range(2), repeat=8):
        if any(bits[v] != state for v, state in obs.items()):
            continue
        w = math.prod(float(t[tuple(bits[v] for v in s)]) for s, t in zip(scopes, tables))
        z += w
        for v, state in enumerate(bits):
            marginals[v, state] += w
        joint[bits[1], bits[3]] += w
    return z, marginals / z, joint / z


if __name__ == "__main__":
    _, scopes, tables = read_model(ROOT / "asia.uai")
    obs = {0: 0, 7: 0}
    z, marginals, joint = enumerate_asia(scopes, tables, obs)
    report = {}
    for name in ["random", "greedy", "replay"]:
        run_dir = ROOT / "runs" / name
        result, scale = contract(run_dir / "order.json", scopes, tables, obs)
        scalar = float(result) * math.exp(scale)
        run = json.loads((run_dir / "run.json").read_text())
        got = np.array([run["marginals"][str(v)] for v in range(8)])
        joint_got = np.array(run["joint_TL"])
        z_err = abs(scalar - z) / z
        marginal_err = float(np.max(np.abs(got - marginals)))
        joint_err = float(np.max(np.abs(joint_got - joint)))
        assert z_err < 1e-10 and marginal_err < 1e-10 and joint_err < 1e-10
        assert np.allclose(got.sum(axis=1), 1, atol=1e-14)
        assert abs(joint_got.sum() - 1) < 1e-14
        assert np.max(np.abs(joint_got.sum(axis=1) - got[1])) < 1e-14
        assert np.max(np.abs(joint_got.sum(axis=0) - got[3])) < 1e-14
        report[name] = {"independent_scalar_relative_error": z_err,
                        "max_marginal_abs_error": marginal_err, "max_joint_abs_error": joint_err}
        print("independent ASIA contraction:", name, report[name], flush=True)

    _, scopes, tables = read_model(ROOT / "promedus/Promedus_14.uai")
    evidence = list(map(int, (ROOT / "promedus/Promedus_14.uai.evid").read_text().split()))
    obs = dict(zip(evidence[1::2], evidence[2::2]))
    run_dir = ROOT / "artifacts/promedus-treesa"
    run = json.loads((run_dir / "run.json").read_text())
    scalar, scale = contract(run_dir / "order.json", scopes, tables, obs)
    log10_z = (math.log(float(scalar)) + scale) / math.log(10.)
    assert abs(log10_z - run["log10_partition"]) < 1e-10
    errors = {}
    for v in [59, 102, 139]:
        result, _ = contract(run_dir / f"query-{v}-order.json", scopes, tables, obs)
        independent = result / result.sum()
        errors[v] = float(np.max(np.abs(independent - np.array(run["marginals"][str(v)]))))
        assert errors[v] < 1e-10
    report["promedus_treesa"] = {"independent_log10_partition": log10_z,
        "log10_partition_abs_error": abs(log10_z - run["log10_partition"]),
        "marginal_abs_errors": errors}
    print("independent Promedus contractions:", report["promedus_treesa"], flush=True)
    (ROOT / "artifacts/audit-exact-paths.json").write_text(json.dumps(report, indent=2))
