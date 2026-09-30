#!/usr/bin/env python3
"""Validate official reflector Treeverse runs and their action logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (1, 3, 5, 10)
SHOTS = (0, 1, 2)
EXPECTED_FORWARD = {1: 28680, 3: 1695, 5: 990, 10: 642}
EXPECTED_STATES = {1: 2, 3: 4, 5: 6, 10: 11}
EXPECTED_BYTES = {1: 53792, 3: 107584, 5: 161376, 10: 295856}


def audit_actions(path: Path, steps: int, delta: int) -> dict[str, int]:
    actions = json.loads(path.read_text())
    saved = {0}
    working = 0
    expected_grads = list(range(steps - 1, -1, -1))
    observed_grads: list[int] = []
    errors: list[str] = []
    grad_order_errors = 0
    invalid_restores = 0
    budget_overruns = 0

    for index, entry in enumerate(actions):
        if set(entry) != {"action", "step", "saved_states"}:
            errors.append(f"action {index}: unexpected JSON fields")
            continue
        kind = entry["action"]
        step = entry["step"]
        claimed = entry["saved_states"]
        if not isinstance(step, int) or not isinstance(claimed, int):
            errors.append(f"action {index}: non-integer step/count")
            continue

        if kind == "store":
            if step != working:
                errors.append(f"action {index}: store at {step}, working at {working}")
            if step in saved:
                errors.append(f"action {index}: duplicate store {step}")
            else:
                saved.add(step)
        elif kind == "restore":
            if step not in saved:
                invalid_restores += 1
                errors.append(f"action {index}: restore of unsaved {step}")
            else:
                working = step
        elif kind == "call":
            if step != working:
                errors.append(f"action {index}: call at {step}, working at {working}")
            if not 0 <= step < steps:
                errors.append(f"action {index}: call outside interval")
            else:
                working = step + 1
        elif kind == "grad":
            if step not in saved:
                errors.append(f"action {index}: grad of unsaved {step}")
            observed_grads.append(step)
        elif kind == "fetch":
            if step not in saved:
                errors.append(f"action {index}: fetch of unsaved {step}")
            elif step == 0:
                errors.append(f"action {index}: attempted fetch of s_0")
            else:
                saved.remove(step)
        else:
            errors.append(f"action {index}: unknown action {kind!r}")

        if len(saved) > delta + 1:
            budget_overruns += 1
        if claimed != len(saved):
            errors.append(
                f"action {index}: saved_states={claimed}, reconstructed={len(saved)}"
            )

    for index, (actual, expected) in enumerate(zip(observed_grads, expected_grads)):
        if actual != expected:
            grad_order_errors += 1
            errors.append(f"grad {index}: got {actual}, expected {expected}")
    grad_order_errors += abs(len(observed_grads) - len(expected_grads))
    if len(observed_grads) != len(expected_grads):
        errors.append("grad sequence has missing or extra entries")
    if saved != {0}:
        errors.append(f"final saved set is {sorted(saved)}, expected [0]")

    return {
        "grad_order_errors": grad_order_errors,
        "invalid_restores": invalid_restores,
        "budget_overruns": budget_overruns,
        "extended_errors": len(errors),
        "actions": len(actions),
        "forward_calls": sum(a.get("action") == "call" for a in actions),
        "reverse_calls": sum(a.get("action") == "grad" for a in actions),
        "peak_saved_states": max([1] + [a["saved_states"] for a in actions]),
    }


def main() -> None:
    full = np.load(ROOT / "artifacts/adjoint/image.npy")
    born = np.load(ROOT / "artifacts/born/born_data.npy")
    born_sum_squares = float(np.sum(born * born))
    expected_born_sum_squares = 3.484789021516375e-02
    if not np.isclose(born_sum_squares, expected_born_sum_squares, rtol=0, atol=1e-15):
        raise AssertionError(f"Born sum of squares changed: {born_sum_squares:.17e}")

    print("delta shot actions forward reverse peak grad_order_errors invalid_restores budget_overruns")
    print("----- ---- ------- ------- ------- ---- ------------------ ----------------- --------------")
    rows = []
    for delta in BUDGETS:
        directory = ROOT / f"artifacts/checkpoint-{delta}"
        result = json.loads((directory / "result.json").read_text())
        stats = result["statistics"]
        assert stats["storage"] == "treeverse"
        assert stats["checkpoints"] == delta
        assert stats["scheduler_forward_calls"] == EXPECTED_FORWARD[delta] * 3
        assert stats["reverse_calls"] == 720
        assert stats["peak_saved_states"] == EXPECTED_STATES[delta]
        assert stats["peak_saved_bytes"] == EXPECTED_BYTES[delta]
        for shot in SHOTS:
            audit = audit_actions(directory / f"actions-{shot}.json", 240, delta)
            per_shot = stats["per_shot"][shot]
            assert audit["grad_order_errors"] == 0
            assert audit["invalid_restores"] == 0
            assert audit["budget_overruns"] == 0
            assert audit["extended_errors"] == 0, audit
            assert audit["forward_calls"] == EXPECTED_FORWARD[delta]
            assert audit["reverse_calls"] == 240
            assert audit["peak_saved_states"] == EXPECTED_STATES[delta]
            assert per_shot["scheduler_forward_calls"] == EXPECTED_FORWARD[delta]
            assert per_shot["reverse_calls"] == 240
            assert per_shot["peak_saved_states"] == EXPECTED_STATES[delta]
            print(delta, shot, audit["actions"], audit["forward_calls"], audit["reverse_calls"], audit["peak_saved_states"], audit["grad_order_errors"], audit["invalid_restores"], audit["budget_overruns"])

        image = np.load(directory / "image.npy")
        difference = image - full
        relative = float(np.linalg.norm(difference) / np.linalg.norm(full))
        maximum = float(np.max(np.abs(difference)))
        assert image.shape == full.shape == (41, 41)
        assert image.dtype == np.float64
        assert np.isfinite(image).all()
        assert relative < 1e-9
        rows.append((delta, relative, maximum, EXPECTED_FORWARD[delta], 240, EXPECTED_STATES[delta], EXPECTED_BYTES[delta]))

    print("\nNumerical checkpoint validation")
    print("delta relative_l2_error max_abs_difference forward/shot reverse/shot peak_states peak_bytes")
    for row in rows:
        print(f"{row[0]:5d} {row[1]:.6e} {row[2]:.6e} {row[3]:14d} {row[4]:11d} {row[5]:11d} {row[6]:10d}")

    full_result = json.loads((ROOT / "artifacts/adjoint/result.json").read_text())
    full_stats = full_result["statistics"]
    assert full_stats["storage"] == "full"
    assert full_stats["checkpoints"] is None
    assert full_stats["scheduler_forward_calls"] == 720
    assert full_stats["reverse_calls"] == 720
    assert full_stats["peak_saved_states"] == 241
    assert full_stats["peak_saved_bytes"] == 6481936
    experiment = json.loads((ROOT / "inputs/reflector.json").read_text())
    perturbation = np.asarray(experiment["perturbation"], dtype=np.float64)
    left = float(np.sum(born * born))
    right = float(np.sum(perturbation * full))
    transpose_relative = abs(left - right) / max(abs(left), abs(right))
    assert transpose_relative < 1e-9
    print(f"\nBorn sum of squares: {born_sum_squares:.17e}")
    print(f"Full-history transpose: left={left:.17e} right={right:.17e} relative={transpose_relative:.17e}")
    print("All checkpoint JSON, images, action logs, and independent audits passed.")


if __name__ == "__main__":
    main()
