"""Imitate full Strong Branching labels with one shared, masked candidate scorer."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

os.environ["JAX_PLATFORMS"] = "cpu"
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.example_libraries import optimizers

SEED, EPOCHS, BATCH = 2026, 120, 64
RATE, PENALTY = 4e-3, 2e-4


def arrays(nodes, width, mean, std):
    features = np.zeros((len(nodes), width, len(mean)))
    mask = np.zeros((len(nodes), width), dtype=bool)
    target = np.zeros_like(mask, dtype=float)
    for index, node in enumerate(nodes):
        candidates = node["candidates"]
        k = len(candidates)
        raw = np.array([c["features"] for c in candidates])
        scores = np.array([c["score"] for c in candidates])
        best = scores.max()
        ties = np.abs(scores - best) <= 1e-9 * max(1, abs(best))
        features[index, :k] = (raw - mean) / np.maximum(std, 1e-6)
        mask[index, :k] = True
        target[index, :k] = ties / ties.sum()
    return tuple(map(jnp.asarray, (features, mask, target)))


def logits(params, features):
    w1, b1, w2, b2 = params
    return jnp.tanh(features @ w1 + b1) @ w2 + b2


@jax.jit
def cross_entropy(params, data):
    x, mask, target = data
    scores = jnp.where(mask, logits(params, x), -jnp.inf)
    log_p = jax.nn.log_softmax(scores, axis=-1)
    return -jnp.sum(target * jnp.where(mask, log_p, 0.0), axis=-1).mean()


def objective(params, data):
    return cross_entropy(params, data) + PENALTY * (
        jnp.square(params[0]).sum() + jnp.square(params[2]).sum()
    )


def train(dataset, output):
    raw = json.loads(dataset.read_text())
    names = raw["feature_names"]
    train_nodes = [r for r in raw["nodes"] if r["split"] == "train"]
    val_nodes = [r for r in raw["nodes"] if r["split"] == "val"]
    train_ids = sorted({r["instance"] for r in train_nodes})
    val_ids = sorted({r["instance"] for r in val_nodes})
    if (
        set(train_ids) & set(val_ids)
        or train_ids != list(range(36))
        or val_ids != list(range(36, 48))
    ):
        raise ValueError(
            "Expected disjoint training instances 0..35 and validation instances 36..47"
        )
    train_rows = np.asarray(
        [c["features"] for n in train_nodes for c in n["candidates"]]
    )
    mean, std = train_rows.mean(axis=0), train_rows.std(axis=0)
    width = max(len(n["candidates"]) for n in train_nodes + val_nodes)
    tr = arrays(train_nodes, width, mean, std)
    va = arrays(val_nodes, width, mean, std)
    # Glorot initialization uses the same seed as the deterministic minibatch shuffle.
    k1, k2 = jax.random.split(jax.random.key(SEED))
    init = jax.nn.initializers.glorot_uniform()
    params = (
        init(k1, (12, 48), jnp.float64),
        jnp.zeros(48),
        init(k2, (48, 1), jnp.float64)[:, 0],
        jnp.zeros(()),
    )
    initialize, update, values = optimizers.adam(RATE)
    state = initialize(params)

    @jax.jit
    def step(index, state, batch):
        grad = jax.grad(objective)(values(state), batch)
        return update(index, grad, state)

    rng = np.random.default_rng(SEED)
    history, iteration = [], 0
    best_ce, checkpoint, best_params = float("inf"), None, None
    tic = perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_nodes))
        for start in range(0, len(order), BATCH):
            ids = order[start : start + BATCH]
            batch = tuple(x[ids] for x in tr)
            state = step(iteration, state, batch)
            iteration += 1
        current = values(state)
        tr_ce = float(cross_entropy(current, tr))
        va_ce = float(cross_entropy(current, va))
        history.append(dict(epoch=epoch, train_ce=tr_ce, val_ce=va_ce))
        if (epoch == 1 or epoch % 5 == 0) and va_ce < best_ce:
            best_ce, checkpoint, best_params = va_ce, epoch, current
    elapsed = perf_counter() - tic
    probes = []
    for idx in np.linspace(0, len(val_nodes) - 1, 6, dtype=int):
        node = val_nodes[idx]
        feats = np.array([c["features"] for c in node["candidates"]])
        scores = logits(
            best_params, jnp.asarray((feats - mean) / np.maximum(std, 1e-6))
        )
        probes.append(
            dict(
                instance=node["instance"],
                node=node["node"],
                features=feats.tolist(),
                logits=scores.tolist(),
                probabilities=jax.nn.softmax(scores).tolist(),
            )
        )
    params = {
        k: np.asarray(v).tolist() for k, v in zip(("w1", "b1", "w2", "b2"), best_params)
    }
    result = dict(
        schema=1,
        seed=SEED,
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        feature_names=names,
        mean=mean.tolist(),
        std=std.tolist(),
        train_instances=train_ids,
        val_instances=val_ids,
        train_nodes=len(train_nodes),
        val_nodes=len(val_nodes),
        best_epoch=checkpoint,
        best_val_ce=best_ce,
        history=history,
        probes=probes,
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
        dtype=str(tr[0].dtype),
        training_seconds=elapsed,
        **params,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, allow_nan=False))
    tmp.replace(output)
    print(
        f"Train {len(train_nodes)} nodes / {len(train_ids)} graphs; validation {len(val_nodes)} nodes / {len(val_ids)} graphs; overlap 0"
    )
    print(
        f"Best checkpoint at epoch {checkpoint}, validation CE {best_ce:.6f}. Training time {elapsed:.3f} s."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    train(args.dataset, args.out)
