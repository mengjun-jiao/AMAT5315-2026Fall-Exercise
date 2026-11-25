"""Independently check the worksheet runs, energy witnesses, labels and JAX checkpoint."""

import argparse
import hashlib
import itertools
import json
import struct
import tomllib
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

ROOT = Path("artifacts")
MASK = (1 << 64) - 1
EPS = 1e-6
NAMES = "x_val frac_dist field_h abs_field_h deg_norm bond_bias eff_field abs_eff_field mean_bond_dual mean_bond_gap mean_nbr_frac gap_ratio".split()


def read(rel):
    return json.loads((ROOT / rel).read_text())


def equal(x, y, tol=1e-8):
    assert_allclose(x, y, atol=tol, rtol=0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeded(seed, instance, n, m):
    def stream(state):
        while True:
            state = (state + 0x9E3779B97F4A7C15) & MASK
            v = state
            v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & MASK
            v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & MASK
            yield v ^ (v >> 31)

    r = stream(next(stream(seed ^ (instance * 0xD1B54A32D192ED03 & MASK))))
    pairs = set()
    for i in range(n):
        while sum(i in edge for edge in pairs) < min(2, n - 1):
            j = next(r) % n
            if i != j:
                pairs.add(tuple(sorted((i, j))))
    while len(pairs) < m:
        i, j = next(r) % n, next(r) % n
        if i != j:
            pairs.add(tuple(sorted((i, j))))
    edges = [[i, j, 1 - 2 * (next(r) & 1)] for i, j in sorted(pairs)]
    return {"h": [next(r) % 3 - 1 for _ in range(n)], "edges": edges}


def energy(graph, bits):
    spin = 1 - 2 * np.asarray(bits, dtype=int)
    assert set(bits) <= {0, 1}
    return sum(j * spin[u] * spin[v] for u, v, j in graph["edges"]) + np.dot(
        graph["h"], spin
    )


def eliminate(graph):
    """Exact min-sum variable elimination on Ising factors, using integer arithmetic only."""
    spin = np.array([1, -1], dtype=np.int64)
    factors = [((i,), int(h) * spin) for i, h in enumerate(graph["h"])]
    factors += [
        (tuple(sorted((u, v))), int(j) * np.outer(spin, spin))
        for u, v, j in graph["edges"]
    ]
    active = set(range(len(graph["h"])))
    while active:
        scopes = {i: set().union(*(s for s, _ in factors if i in s)) for i in active}
        i = min(active, key=lambda v: (len(scopes[v]), v))
        scope = sorted(scopes[i])
        remaining, combined = [], 0
        for indices, array in factors:
            if i in indices:
                combined = combined + array.reshape(
                    [2 if v in indices else 1 for v in scope]
                )
            else:
                remaining.append((indices, array))
        axis = scope.index(i)
        scope.pop(axis)
        remaining.append((tuple(scope), np.min(combined, axis=axis)))
        factors = remaining
        active.remove(i)
    return int(sum(int(v) for _, v in factors))


def pred_examples():
    for slug, target in (("square", -2), ("r100", -138)):
        d = read(f"pred/{slug}.json")["data"]
        graph = {
            "h": d["fields"],
            "edges": [[*e, w] for e, w in zip(d["graph"]["edges"], d["couplings"])],
        }
        log = (ROOT / f"pred/{slug}.solve.log").read_text()
        record = json.JSONDecoder().raw_decode(log[log.index("{") :])[0]
        assert record["solver"] == "ilp" and record["evaluation"] == f"Min({target})"
        equal(energy(graph, record["solution"]), target)
        if slug == "square":
            equal(
                min(energy(graph, s) for s in itertools.product((0, 1), repeat=4)),
                target,
            )
        else:
            assert len(graph["h"]) == 100
            equal(eliminate(graph), target)


def check_gap(run):
    e = run["evidence"]
    sq, d = e["square"], e["diet"]
    equal([sq["lp"], sq["exact"], sq["rounded_energy"], sq["excess"]], [-4, -2, 2, 4])
    equal(sq["lp_x"], [0.5] * 4)
    equal(sq["lp_z"], [1, 1, 1, 0])
    equal(d["x"], [89 / 28, 8 / 7])
    equal(d["cost"], 323 / 14)
    rows = e["instances"]
    exact_ref = [
        -37,
        -39,
        -40,
        -37,
        -42,
        -38,
        -36,
        -35,
        -37,
        -37,
        -36,
        -44,
        -38,
        -41,
        -39,
        -36,
    ]
    excess_ref = [28, 34, 22, 44, 36, 40, 32, 26, 32, 40, 30, 60, 30, 46, 18, 30]
    for instance, key, ex in zip(rows, exact_ref, excess_ref, strict=True):
        graph, r = instance["graph"], instance["result"]
        assert graph == seeded(2026, instance["id"], 22, 52)
        equal(r["lp_x"], [0.5] * 22)
        equal(r["lp_z"], [int(j > 0) for _, _, j in graph["edges"]])
        equal(
            [r["lp"], r["exact"], r["gap"], r["fractional"], r["excess"]],
            [-52, key, key + 52, 22, ex],
        )
        equal(energy(graph, r["exact_bits"]), key)
        equal(energy(graph, r["rounded_bits"]), r["rounded_energy"])
        equal(r["rounded_bits"], np.asarray(r["lp_x"]) >= 0.5)
        equal(r["relative_gap"], r["gap"] / abs(key))
    s = run["summary"]
    equal(
        [
            s[k]
            for k in (
                "mean_lp",
                "mean_exact",
                "mean_gap",
                "mean_fractional",
                "suboptimal_roundings",
                "mean_rounding_excess",
            )
        ],
        [-52, -38.25, 13.75, 22, 16, 34.25],
    )


def trace_check(trace):
    equal([v["bound"] for v in trace], [-6, -4, -3, -3, -2])
    assert [v["path"] for v in trace] == [
        [],
        [[0, 1]],
        [[0, 1], [1, 1]],
        [[0, 1], [1, 0]],
        [[0, 0]],
    ]
    assert [v["status"] for v in trace] == [
        "branch",
        "branch",
        "incumbent",
        "pruned by bound",
        "pruned by bound",
    ]
    equal(
        [trace[k]["x"] for k in (0, 1, 2, 4)],
        [[0.5] * 4, [1, 0.5, 0.5, 0.5], [1, 1, 0, 0], [0, 0.5, 0.5, 0.5]],
    )


def product_lp(graph):
    """Use y=x_u*x_v (McCormick inequalities) in a separate SciPy formulation."""
    n, m = len(graph["h"]), len(graph["edges"])
    mat = np.zeros((3 * m, n + m))
    rhs = np.tile([0.0, 0.0, 1.0], m)
    c = np.r_[-2 * np.asarray(graph["h"], float), np.zeros(m)]
    offset = sum(graph["h"])
    for e, (u, v, w) in enumerate(graph["edges"]):
        q = n + e
        mat[3 * e, [q, u]] = [1, -1]
        mat[3 * e + 1, [q, v]] = [1, -1]
        mat[3 * e + 2, [u, v, q]] = [1, 1, -1]
        c[[u, v]] -= 2 * w
        c[q] = 4 * w
        offset += w
    mat = csr_matrix(mat)

    @lru_cache(maxsize=None)
    def bound(fixed):
        bounds = [(0, 1) if f < 0 else (f, f) for f in fixed] + [(0, 1)] * m
        sol = linprog(
            c,
            A_ub=mat,
            b_ub=rhs,
            bounds=bounds,
            method="highs-ds",
            options={"presolve": False},
        )
        assert sol.success
        return sol.fun + offset

    return bound


def check_labels(dataset, strong, trials):
    assert dataset["feature_names"] == NAMES
    total_children = 0
    for row in strong["evidence"]["instances"]:
        graph, id = row["graph"], row["id"]
        nodes = [r for r in dataset["nodes"] if r["instance"] == id]
        search = row["policies"]["strong"]["search"]
        assert 0 < len(nodes) <= 80 and len(nodes) == (search["nodes"] - 1) // 2
        solve = product_lp(graph) if trials else None
        h = np.array(graph["h"])
        edges = np.asarray(graph["edges"])
        u, v = edges[:, :2].astype(int).T
        w = edges[:, 2]
        deg = np.bincount(np.r_[u, v], minlength=len(h))
        child_count = 0
        for label in nodes:
            assert label["split"] == ("train" if id < 36 else "val")
            fixed = np.asarray(label["fixed"])
            x, z, duals = (np.array(label[k]) for k in ("x", "z", "row_duals"))
            duals = duals.reshape(-1, 4)
            candidates = label["candidates"]
            assert [c["spin"] for c in candidates] == np.flatnonzero(
                (fixed < 0) & (x > EPS) & (x < 1 - EPS)
            ).tolist()
            p, b = label["lower_bound"], label["incumbent"]
            if solve:
                equal(solve(tuple(fixed)), p)
            equal(np.dot(h, 1 - 2 * x) + np.dot(w, 1 - 2 * z), p)
            constraints = np.stack(
                (
                    z - x[u] - x[v],
                    z + x[u] + x[v] - 2,
                    x[u] - x[v] - z,
                    x[v] - x[u] - z,
                ),
                axis=1,
            )
            assert np.max(constraints) < EPS and np.max(duals) < EPS
            equal(constraints * duals, 0.0)
            spin = 1 - 2 * np.clip(x, 0, 1)
            frac = 1 - abs(spin)
            for c in candidates:
                i = c["spin"]
                loc = (u == i) | (v == i)
                nbr = np.where(u[loc] == i, v[loc], u[loc])
                eff = (h[i] + np.dot(w[loc], spin[nbr])) / (deg[i] + 1)
                expected = [
                    x[i],
                    frac[i],
                    h[i],
                    abs(h[i]),
                    deg[i] / deg.max(),
                    np.sign(w[loc]).mean(),
                    eff,
                    abs(eff),
                    np.abs(duals[loc]).sum(axis=1).mean() / max(1e-9, abs(duals).max()),
                    abs(z[loc] - (1 - spin[i] * spin[nbr]) / 2).mean(),
                    np.where(fixed[nbr] < 0, frac[nbr], 0).mean(),
                    1 if b is None else np.clip((b - p) / max(1, abs(b)), 0, 2),
                ]
                equal(c["features"], expected)
                gains = np.array(c["gains"])
                equal(c["score"], np.maximum(gains, 1e-6).prod())
                assert c["pruned"] == sum(gains == 1e6)
                if solve:
                    for side in range(2):
                        child = fixed.copy()
                        child[i] = side
                        val = solve(tuple(child))
                        equal(
                            gains[side],
                            1e6 if b is not None and val >= b - EPS else val - p,
                        )
                child_count += 2
            assert label["chosen"] == max(candidates, key=lambda c: c["score"])["spin"]
        assert search["lp_solves"] == search["nodes"] + child_count
        total_children += child_count
    return total_children


def check_checkpoint(data, m):
    assert m["jax_version"] == "0.12.1" and m["dtype"] == "float64"
    assert m["seed"] == 2026 and m["dataset_sha256"] == sha(
        ROOT / "bnb_strong/dataset.json"
    )
    tr = [n for n in data["nodes"] if n["split"] == "train"]
    va = [n for n in data["nodes"] if n["split"] == "val"]
    assert [len(tr), len(va)] == [276, 84] == [m["train_nodes"], m["val_nodes"]]
    assert set(m["train_instances"]) == {r["instance"] for r in tr} == set(range(36))
    assert set(m["val_instances"]) == {r["instance"] for r in va} == set(range(36, 48))
    all_x = np.array([c["features"] for n in tr for c in n["candidates"]])
    mean, std = all_x.mean(axis=0), all_x.std(axis=0)
    equal(mean, m["mean"])
    equal(std, m["std"])
    w1, w2 = np.array(m["w1"]), np.array(m["w2"])
    assert w1.shape == (12, 48) and w2.shape == (48,)

    def predict(f):
        raw = (
            np.tanh((np.asarray(f) - mean) / np.maximum(std, 1e-6) @ w1 + m["b1"]) @ w2
            + m["b2"]
        )
        e = np.exp(raw - raw.max())
        return raw, e / e.sum()

    def ce(rows):
        losses = []
        for n in rows:
            _, p = predict([c["features"] for c in n["candidates"]])
            scores = np.array([c["score"] for c in n["candidates"]])
            tied = abs(scores - scores.max()) <= 1e-9 * max(1, abs(scores.max()))
            losses.append(-np.log(p[tied]).mean())
        return np.mean(losses)

    hist = m["history"]
    assert [r["epoch"] for r in hist] == list(range(1, 121))
    best = min(
        (r for r in hist if r["epoch"] == 1 or r["epoch"] % 5 == 0),
        key=lambda r: r["val_ce"],
    )
    assert m["best_epoch"] == best["epoch"]
    equal(
        [ce(tr), ce(va), m["best_val_ce"]],
        [best["train_ce"], best["val_ce"], best["val_ce"]],
        1e-11,
    )
    assert len(m["probes"]) == 6
    for probe in m["probes"]:
        n = next(
            n
            for n in va
            if (n["instance"], n["node"]) == (probe["instance"], probe["node"])
        )
        equal(probe["features"], [c["features"] for c in n["candidates"]])
        raw, p = predict(probe["features"])
        equal(raw, probe["logits"], 1e-12)
        equal(p, probe["probabilities"], 1e-12)


def main(trials):
    for design in ("branchopt.design.toml", "train_branch.design.toml"):
        with open(design, "rb") as f:
            tomllib.load(f)
    pred_examples()
    gap = read("gap/run.json")
    check_gap(gap)
    plain = read("bnb_strong/run_no_labels.json")
    strong = read("bnb_strong/run.json")
    most = read("bnb_most/run.json")
    held, transfer = read("eval_test/run.json"), read("eval_transfer/run.json")
    exact = {}
    max_error = 0.0
    error_by_mode = {"bnb": 0.0, "eval": 0.0}
    for run, n, m, count, seed in (
        (gap, 22, 52, 16, 2026),
        (most, 22, 52, 48, 2026),
        (plain, 22, 52, 48, 2026),
        (strong, 22, 52, 48, 2026),
        (held, 22, 52, 16, 3026),
        (transfer, 30, 72, 8, 4026),
    ):
        a = run["parameters"]
        assert [a["spins"], a["edges"], a["count"], a["seed"]] == [n, m, count, seed]
        records = run["evidence"]["instances"]
        assert [row["id"] for row in records] == list(range(count))
        if run is not gap:
            trace_check(run["evidence"]["tetrahedron"]["trace"])
        for row in records:
            g, id = row["graph"], row["id"]
            assert g == seeded(seed, id, n, m)
            key = (seed, id)
            if key not in exact:
                exact[key] = eliminate(g)
            opt = exact[key]
            if run is gap:
                equal(row["result"]["exact"], opt)
                continue
            tolerance = 1e-9 if run["mode"] == "bnb" else 1e-6
            equal(row["oracle"], opt, tolerance)
            equal(energy(g, row["oracle_bits"]), opt)
            for rule, record in row["policies"].items():
                r = record["search"]
                equal([energy(g, r["bits"]), r["energy"]], [opt, opt])
                equal(record["error"], abs(row["oracle"] - r["energy"]))
                max_error = max(max_error, record["error"])
                assert record["error"] <= tolerance
                error_by_mode[run["mode"]] = max(
                    error_by_mode[run["mode"]], record["error"]
                )
                assert (r["lp_solves"] == r["nodes"]) == (rule != "strong")
                assert 0 <= r["lp_ms"] + r["inference_ms"] <= r["wall_ms"] + 1e-6
        if run is not gap:
            for p, s in run["summary"].items():
                metrics = {
                    "mean_nodes": "nodes",
                    "mean_lp_solves": "lp_solves",
                    "mean_wall_ms": "wall_ms",
                    "mean_lp_ms": "lp_ms",
                    "mean_inference_ms": "inference_ms",
                }
                for key, metric in metrics.items():
                    equal(
                        s[key],
                        np.mean([r["policies"][p]["search"][metric] for r in records]),
                    )
                equal(s["max_error"], max(r["policies"][p]["error"] for r in records))
    assert len(exact) == 72
    equal(
        [
            most["summary"]["most-frac"]["mean_nodes"],
            plain["summary"]["strong"]["mean_nodes"],
            plain["summary"]["strong"]["mean_lp_solves"],
        ],
        [50.875, 16, 287 + 5 / 12],
    )
    for a, b in zip(
        plain["evidence"]["instances"], strong["evidence"]["instances"], strict=True
    ):
        for k in ("energy", "nodes", "lp_solves", "first_splits"):
            assert (
                a["policies"]["strong"]["search"][k]
                == b["policies"]["strong"]["search"][k]
            )
    verified_model_digests = 0
    for r, folder in ((held, "eval_test"), (transfer, "eval_transfer")):
        digest_file = ROOT / folder / "model.sha256"
        if (
            digest_file.exists()
        ):  # The reproduction shell script records this extra provenance.
            assert digest_file.read_text().split()[0] == sha(ROOT / "model.json")
            verified_model_digests += 1
        else:
            assert sha(Path(r["parameters"]["model"])) == sha(ROOT / "model.json")
        s = r["summary"]
        for p in ("neural", "root-neural"):
            assert s[p]["mean_nodes"] < s["most-frac"]["mean_nodes"]
            assert s[p]["mean_wall_ms"] < s["strong"]["mean_wall_ms"] / 2
        parity = r["evidence"]["parity"]
        assert (
            len(parity["probability_errors"]) == 6
            and max(parity["probability_errors"]) < 1e-4
        )
        sample = r["evidence"]["largest_saving"]
        largest = max(
            r["evidence"]["instances"],
            key=lambda a: (
                a["policies"]["most-frac"]["search"]["nodes"]
                - a["policies"]["neural"]["search"]["nodes"]
            ),
        )
        assert largest["id"] == sample["id"]
        for rule in ("neural", "most-frac"):
            assert largest["policies"][rule]["search"] == sample[rule]
    dataset = read("bnb_strong/dataset.json")
    evaluated = check_labels(dataset, strong, trials)
    check_checkpoint(dataset, read("model.json"))
    for name, dims in (
        ("gap", (1655, 603)),
        ("bnb", (1655, 603)),
        ("neural_bnb", (1656, 603)),
        ("eval", (1692, 639)),
    ):
        header = (ROOT / f"{name}.png").read_bytes()[:24]
        assert (
            header[:8] == b"\x89PNG\r\n\x1a\n"
            and struct.unpack(">II", header[16:24]) == dims
        )
    results = dict(
        verified_unique_graphs=len(exact),
        pred_optima=[-2, -138],
        tetrahedron_nodes=5,
        strong_label_nodes=len(dataset["nodes"]),
        trial_LPs_accounted_for=evaluated,
        trial_LPs_independently_resolved=evaluated if trials else 0,
        max_error_against_HiGHS=max_error,
        max_error_by_mode=error_by_mode,
        evaluation_model_files_verified_by_digest=verified_model_digests,
        model_sha256=sha(ROOT / "model.json"),
        figures_checked=4,
    )
    target = ROOT / "audit"
    target.mkdir(exist_ok=True)
    (target / "numeric.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip the independent re-solve of Strong Branching trial LPs",
    )
    main(not parser.parse_args().quick)
