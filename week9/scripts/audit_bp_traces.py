"""Independently recompute every synchronous BP sweep from the source UAI data."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def read_model(path):
    words = iter(path.read_text().split())
    assert next(words) == "MARKOV"
    n = int(next(words))
    assert all(int(next(words)) == 2 for _ in range(n))
    scopes = [[int(next(words)) for _ in range(int(next(words)))]
              for _ in range(int(next(words)))]
    factors = []
    for scope in scopes:
        size = int(next(words))
        assert size == 2 ** len(scope)
        factors.append(np.array([float(next(words)) for _ in range(size)])
                       .reshape((2,) * len(scope)))
    return n, scopes, factors


def check(name, model_path, observations):
    n, scopes, raw_factors = read_model(model_path)
    run = json.loads((ROOT / "artifacts" / name / "run.json").read_text())
    assert run["observations"] == {str(k): v for k, v in observations.items()}
    assert run["damping"] == .5 and run["tolerance"] == 1e-10 and run["max_sweeps"] == 1000
    factors = []
    adjacent = [[] for _ in range(n)]
    for fi, (scope, table) in enumerate(zip(scopes, raw_factors)):
        table = table.copy()
        for pos, var in enumerate(scope):
            adjacent[var].append((fi, pos))
            if var in observations:
                axis = [slice(None)] * len(scope)
                axis[pos] = 1 - observations[var]
                table[tuple(axis)] = 0
        factors.append(table)

    vf = [[np.array([.5, .5]) for _ in scope] for scope in scopes]
    fv = [[np.array([.5, .5]) for _ in scope] for scope in scopes]
    per_sweep_work = sum(2 * len(edges) * (len(edges) - 1) for edges in adjacent)
    per_sweep_work += sum(len(scope) ** 2 * 2 ** len(scope) for scope in scopes)
    max_belief_diff = max_residual_diff = 0.
    trace = run["trace"]
    assert len(trace) == run["sweeps"] + 1
    for sweep, recorded in enumerate(trace):
        next_vf = [[None] * len(scope) for scope in scopes]
        next_fv = [[None] * len(scope) for scope in scopes]
        beliefs = np.empty((n, 2))
        for v, edges in enumerate(adjacent):
            belief = np.ones(2)
            for fi, pos in edges:
                belief *= fv[fi][pos]
            beliefs[v] = belief / belief.sum()
            for fi, pos in edges:
                message = np.ones(2)
                for other_fi, other_pos in edges:
                    if other_fi != fi:
                        message *= fv[other_fi][other_pos]
                next_vf[fi][pos] = message / message.sum()
        for fi, (scope, table) in enumerate(zip(scopes, factors)):
            for pos in range(len(scope)):
                weighted = table.copy()
                for other_pos in range(len(scope)):
                    if other_pos != pos:
                        shape = [1] * len(scope)
                        shape[other_pos] = 2
                        weighted *= vf[fi][other_pos].reshape(shape)
                axes = tuple(k for k in range(len(scope)) if k != pos)
                message = weighted.sum(axis=axes) if axes else weighted
                next_fv[fi][pos] = message / message.sum()
        residual = max(
            (max(float(np.max(np.abs(next_vf[fi][pos] - vf[fi][pos]))),
                 float(np.max(np.abs(next_fv[fi][pos] - fv[fi][pos]))))
             for fi, scope in enumerate(scopes) for pos in range(len(scope))),
            default=0.)
        max_belief_diff = max(max_belief_diff,
                              float(np.max(np.abs(beliefs - np.array(recorded["beliefs"])))))
        max_residual_diff = max(max_residual_diff, abs(residual - recorded["residual"]))
        assert recorded["sweep"] == sweep
        assert recorded["cumulative_work"] == sweep * per_sweep_work
        assert max_belief_diff < 1e-11 and max_residual_diff < 1e-11, (name, sweep, max_belief_diff, max_residual_diff)
        if sweep + 1 < len(trace):
            assert recorded["residual"] > run["tolerance"]
            for fi, scope in enumerate(scopes):
                for pos in range(len(scope)):
                    vf[fi][pos] = .5 * vf[fi][pos] + .5 * next_vf[fi][pos]
                    fv[fi][pos] = .5 * fv[fi][pos] + .5 * next_fv[fi][pos]
    assert run["converged"] == (run["residual"] <= run["tolerance"])
    assert np.max(np.abs(beliefs - np.array(run["marginals"]))) < 1e-11
    assert abs(residual - run["residual"]) < 1e-11
    return {"sweeps": run["sweeps"], "per_sweep_update_work": per_sweep_work,
            "max_belief_difference": max_belief_diff,
            "max_residual_difference": max_residual_diff}


if __name__ == "__main__":
    result = {}
    model = ROOT / "asia.uai"
    for case, obs in [("prior", {}), ("AD", {0: 0, 7: 0}), ("ADX", {0: 0, 7: 0, 6: 0})]:
        result["bp-" + case] = check("bp-" + case, model, obs)
        print("verified", case, result["bp-" + case], flush=True)
    evidence = list(map(int, (ROOT / "promedus/Promedus_14.uai.evid").read_text().split()))
    obs = dict(zip(evidence[1::2], evidence[2::2]))
    assert len(obs) == evidence[0] == 9
    result["promedus-bp"] = check("promedus-bp", ROOT / "promedus/Promedus_14.uai", obs)
    print("verified Promedus", result["promedus-bp"], flush=True)
    (ROOT / "artifacts" / "audit-bp-traces.json").write_text(json.dumps(result, indent=2))
