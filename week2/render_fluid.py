"""Render Rust-produced frame/RDF data; no force or RDF recomputation here."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(source, destination):
    with Path(source).open() as stream:
        frames = [json.loads(line) for line in stream]
    if not frames:
        raise ValueError("No saved frames")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.8), dpi=100)
    box = frames[0]["box"]
    left.set(xlim=(0, box[0]), ylim=(0, box[1]), xlabel="x", ylabel="y")
    left.set_aspect("equal")
    atoms = left.scatter([], [], s=24, c="#176b9c", edgecolors="white", linewidths=0.3)
    line, = right.plot([], [], color="#c25320", linewidth=1.6)
    ymax = max(float(np.max(f["rdf_g"])) for f in frames)
    right.set(xlim=(0, min(box)/2), ylim=(0, max(1.2, 1.08*ymax)), xlabel="r", ylabel="g(r)")
    right.grid(alpha=0.2)
    fig.suptitle("2D Lennard-Jones fluid | saved trajectory")
    left.set_title("t=0.00")
    right.set_title("g(r): last 20/20 saved frames")
    fig.tight_layout()
    bitrate = min(900_000, int(1_700_000*8/(len(frames)/20)))
    try:
        for attempt in range(4):
            with tempfile.TemporaryFile() as errors:
                command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
                           "-pixel_format", "rgba", "-video_size", "960x480",
                           "-framerate", "20", "-i", "pipe:0", "-an",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-b:v", str(bitrate), "-movflags", "+faststart", str(destination)]
                encoder = subprocess.Popen(command, stdin=subprocess.PIPE,
                                           stdout=subprocess.DEVNULL, stderr=errors)
                try:
                    for frame in frames:
                        atoms.set_offsets(np.asarray(frame["pos"]))
                        line.set_data(frame["rdf_r"], frame["rdf_g"])
                        left.set_title(f"t={frame['t']:.2f}")
                        right.set_title(f"g(r): last {frame['window_frames']}/{frame['window_limit']} saved frames")
                        fig.canvas.draw()
                        encoder.stdin.write(fig.canvas.buffer_rgba())
                    encoder.stdin.close()
                    status = encoder.wait()
                except BaseException:
                    encoder.kill()
                    encoder.wait()
                    errors.seek(0)
                    print(errors.read().decode(errors="replace"), file=sys.stderr)
                    raise
                if status:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode(errors="replace"))
            if Path(destination).stat().st_size < 2_000_000:
                return
            bitrate = max(1, int(bitrate*0.7))
        raise RuntimeError("MP4 exceeds 2,000,000 bytes after four encoding attempts")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
