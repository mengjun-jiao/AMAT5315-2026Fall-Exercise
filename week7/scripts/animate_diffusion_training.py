"""Compare DDPM trajectories across the saved diffusion checkpoints."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ddpm import ddpm_trajectory
from smoke_test_ddpm import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "artifacts" / "part1_diffusion_training.json"
CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
EVIDENCE_PATH = ROOT / "evidence"
OUTPUT_PATH = ROOT / "artifacts" / "part1_diffusion_training_animation.json"
GIF_PATH = EVIDENCE_PATH / "part1_diffusion_training.gif"
STAGES_PATH = EVIDENCE_PATH / "part1_diffusion_training_stages.png"

CHECKPOINT_UPDATES = (0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000)
SEED = 7401
PATH_COUNT = 10
STEPS = 1000
DISPLAY_LIMITS = ((-10.0, 10.0), (-10.0, 10.0))
FPS = 2


def checkpoint_path(update: int) -> Path:
    return CHECKPOINT_DIR / f"update_{update:06d}.npz"


def shared_noise(seed: int) -> tuple[np.ndarray, np.ndarray]:
    key = jax.random.key(seed)
    initial_key, reverse_key = jax.random.split(key)
    initial = jax.random.normal(initial_key, (PATH_COUNT, 2), dtype=jax.numpy.float32)
    reverse_keys = jax.random.split(reverse_key, STEPS)
    reverse = jax.vmap(
        lambda key: jax.random.normal(key, (PATH_COUNT, 2), dtype=jax.numpy.float32)
    )(reverse_keys)
    return np.asarray(initial), np.asarray(reverse)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def path_limits(paths: dict[int, np.ndarray]) -> dict[str, list[float]]:
    all_values = np.concatenate([path.reshape(-1, 2) for path in paths.values()])
    return {
        "x": [float(all_values[:, 0].min()), float(all_values[:, 0].max())],
        "y": [float(all_values[:, 1].min()), float(all_values[:, 1].max())],
    }


def draw_common_axis(axis, title: str, include_centers: bool = True) -> None:
    axis.set_xlim(*DISPLAY_LIMITS[0])
    axis.set_ylim(*DISPLAY_LIMITS[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    if include_centers:
        centers = np.asarray(((3, 0), (0, 3), (-3, 0), (0, -3)), dtype=float)
        axis.scatter(
            centers[:, 0], centers[:, 1], marker="x", s=80, c="black", label="target centers"
        )


def save_stages(paths: dict[int, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for axis, update in zip(axes.flat, CHECKPOINT_UPDATES):
        path = paths[update]
        for path_index in range(PATH_COUNT):
            axis.plot(path[:, path_index, 0], path[:, path_index, 1], alpha=0.35, linewidth=0.7)
        axis.scatter(path[-1, :, 0], path[-1, :, 1], s=18, c="tab:red", label="endpoints")
        draw_common_axis(axis, f"update {update}")
    figure.suptitle("DDPM paths across diffusion training checkpoints")
    figure.savefig(STAGES_PATH, dpi=150)
    plt.close(figure)


def save_animation(paths: dict[int, np.ndarray], loss_history: list[float]) -> None:
    figure, (path_axis, loss_axis) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    lines = [path_axis.plot([], [], alpha=0.45, linewidth=1.0)[0] for _ in range(PATH_COUNT)]
    endpoints = path_axis.scatter([], [], s=28, c="tab:red", label="endpoints")
    draw_common_axis(path_axis, "")
    path_axis.legend(loc="upper right", fontsize=8)
    loss_line = loss_axis.plot([], [], color="tab:blue", label="batch loss")[0]
    marker = loss_axis.axvline(0, color="tab:red", linestyle="--", label="checkpoint")
    loss_axis.set_xlim(0, len(loss_history))
    loss_axis.set_ylim(0.0, max(loss_history) * 1.05)
    loss_axis.set_xlabel("update")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(alpha=0.2)
    loss_axis.legend(loc="upper right", fontsize=8)
    suptitle = figure.suptitle("")

    def update(frame_index: int):
        update_number = CHECKPOINT_UPDATES[frame_index]
        path = paths[update_number]
        for line, path_index in zip(lines, range(PATH_COUNT)):
            line.set_data(path[:, path_index, 0], path[:, path_index, 1])
        endpoints.set_offsets(path[-1])
        draw_common_axis(path_axis, f"checkpoint update {update_number}", include_centers=False)
        if update_number:
            loss_line.set_data(np.arange(1, update_number + 1), loss_history[:update_number])
        else:
            loss_line.set_data([], [])
        marker.set_xdata([update_number, update_number])
        suptitle.set_text(f"Shared-noise DDPM comparison — update {update_number}")
        return (*lines, endpoints, loss_line, marker, suptitle)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(CHECKPOINT_UPDATES),
        interval=1000 / FPS,
        blit=False,
        repeat=True,
    )
    animation.save(GIF_PATH, writer=PillowWriter(fps=FPS))
    plt.close(figure)


def main() -> None:
    started = time.perf_counter()
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    if len(training["loss_history"]) != 25000:
        raise AssertionError("training history does not contain 25000 updates")
    if tuple(training["checkpoint_updates"]) != CHECKPOINT_UPDATES:
        raise AssertionError("training checkpoint list does not match the required list")
    for update in CHECKPOINT_UPDATES:
        if not checkpoint_path(update).is_file():
            raise FileNotFoundError(checkpoint_path(update))

    initial_noise, reverse_noise = shared_noise(SEED)
    paths = {}
    for update in CHECKPOINT_UPDATES:
        parameters = load_checkpoint(checkpoint_path(update))
        trajectory = ddpm_trajectory(parameters, initial_noise, reverse_noise, STEPS)
        paths[update] = np.asarray(trajectory)
    save_stages(paths)
    save_animation(paths, training["loss_history"])

    endpoints = {
        str(update): paths[update][-1].tolist() for update in CHECKPOINT_UPDATES
    }
    output = {
        "seed": SEED,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "path_count": PATH_COUNT,
        "ddpm_steps": STEPS,
        "shared_noise": {
            "reused_across_checkpoints": True,
            "initial_noise_shape": list(initial_noise.shape),
            "reverse_noise_shape": list(reverse_noise.shape),
            "initial_noise_sha256": digest(initial_noise),
            "reverse_noise_sha256": digest(reverse_noise),
        },
        "animation": {
            "frame_count": len(CHECKPOINT_UPDATES),
            "fps": FPS,
            "display_limits": {
                "x": list(DISPLAY_LIMITS[0]),
                "y": list(DISPLAY_LIMITS[1]),
            },
            "raw_path_extrema": path_limits(paths),
            "display_clipping": True,
            "display_clipping_note": "Raw states are unchanged; only plot limits are fixed at [-10, 10].",
        },
        "loss_at_checkpoint": {
            str(update): (None if update == 0 else training["loss_history"][update - 1])
            for update in CHECKPOINT_UPDATES
        },
        "final_endpoints": endpoints,
        "runtime_seconds": time.perf_counter() - started,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"loaded checkpoints: {len(CHECKPOINT_UPDATES)}")
    print(f"shared initial noise digest: {digest(initial_noise)}")
    print(f"shared reverse noise digest: {digest(reverse_noise)}")
    print(f"raw path extrema: {output['animation']['raw_path_extrema']}")
    print(f"wrote {GIF_PATH}")
    print(f"wrote {STAGES_PATH}")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
