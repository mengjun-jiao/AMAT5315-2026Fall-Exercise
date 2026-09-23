"""Render Part 1 line-stability evidence using data emitted by Rust."""

import io
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week4" / "evidence"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def color_map(value, low, high):
    value = max(0.0, min(1.0, (value - low) / (high - low)))
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    position = value * (len(stops) - 1)
    left = int(position)
    fraction = position - left
    if left == len(stops) - 1:
        return stops[-1]
    a, b = stops[left], stops[left + 1]
    return tuple(int(a[i] * (1 - fraction) + b[i] * fraction) for i in range(3))


def diverging(value, maximum):
    scale = math.copysign(math.log1p(abs(value) / 1e-3) / math.log1p(maximum / 1e-3), value)
    if scale < 0:
        t = scale + 1
        return (int(40 * t + 49 * (1 - t)), int(100 * t + 54 * (1 - t)), int(190 * t + 149 * (1 - t)))
    t = scale
    return (int(255 * t + 255 * (1 - t)), int(90 * t + 245 * (1 - t)), int(80 * t + 245 * (1 - t)))


def plot_xy(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    px = left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)
    py = bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)
    return int(px), int(py)


def label_axes(draw, box, xlabel, ylabel, title, xlim, ylim):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) // 2, bottom + 24), xlabel, fill="black", anchor="mm", font=font(16))
    draw.text((max(20, left - 42), (top + bottom) // 2), ylabel, fill="black", anchor="mm", font=font(16))
    draw.text(((left + right) // 2, top - 25), title, fill="black", anchor="mm", font=font(17))
    for value in [xlim[0], 0, xlim[1]]:
        px, _ = plot_xy(box, value, ylim[0], xlim, ylim)
        draw.text((px, bottom + 7), f"{value:g}", fill="black", anchor="ma", font=font(12))
    for value in [ylim[0], 0, ylim[1]]:
        _, py = plot_xy(box, xlim[0], value, xlim, ylim)
        draw.text((left - 7, py), f"{value:g}", fill="black", anchor="rm", font=font(12))


def boundary_points(function, xlim, ylim):
    points = []
    for row in range(241):
        y = ylim[0] + (ylim[1] - ylim[0]) * row / 240
        previous = function(complex(xlim[0], y)) - 1.0
        for col in range(1, 241):
            x = xlim[0] + (xlim[1] - xlim[0]) * col / 240
            current = function(complex(x, y)) - 1.0
            if previous * current <= 0:
                points.append((x, y))
            previous = current
    return points


def main():
    EVIDENCE.mkdir(exist_ok=True)
    command = ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "week4" / "Cargo.toml"), "--bin", "line_stability_data"]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    grid, spectra, profiles, critical = [], {0.045: [], 0.056: []}, {0.045: [], 0.056: []}, None
    for line in io.StringIO(output):
        fields = line.rstrip().split(",")
        if fields[0] == "CRITICAL":
            critical = float(fields[1])
        elif fields[0] == "GRID":
            grid.append(tuple(map(float, fields[1:])))
        elif fields[0] == "SPECTRUM":
            spectra[float(fields[1])].append(tuple(map(float, fields[2:])))
        elif fields[0] == "LINE":
            profiles[float(fields[1])].append(tuple(map(float, fields[2:])))
    print(f"Computed RK4 line stability limit: dt_crit = {critical:.12f}")

    image = Image.new("RGB", (1600, 690), "white")
    draw = ImageDraw.Draw(image)
    panels = [(35, 75, 515, 560), (565, 75, 1015, 560), (1065, 75, 1515, 560)]
    xlim, ylim = (-4.0, 1.0), (-4.0, 4.0)
    for real, imaginary, growth in grid:
        px, py = plot_xy(panels[0], real, imaginary, xlim, ylim)
        draw.point((px, py), fill=color_map(math.log10(max(growth, 1e-3)), -3.0, 2.0))
    for function, color in [
        (lambda z: abs(1 + z), "black"),
        (lambda z: abs(1 + z + z**2 / 2), "orange"),
        (lambda z: abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24), "red"),
    ]:
        for point in boundary_points(function, xlim, ylim):
            draw.point(plot_xy(panels[0], *point, xlim, ylim), fill=color)
    for dt, color in [(0.045, "cyan"), (0.056, "magenta")]:
        for _, real, imaginary, _ in spectra[dt]:
            px, py = plot_xy(panels[0], real * dt, imaginary * dt, xlim, ylim)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
    label_axes(draw, panels[0], "Re(z)", "Im(z)", "Measured RK4 stability map", xlim, ylim)
    legend = [("black", "Euler"), ("orange", "midpoint"), ("red", "RK4")]
    x_position = 45
    for color, label in legend:
        draw.line((x_position, 610, x_position + 22, 610), fill=color, width=3)
        draw.text((x_position + 28, 610), label, fill="black", anchor="lm", font=font(12))
        x_position += 92
    for color, label in [("cyan", "dt = 0.045"), ("magenta", "dt = 0.056")]:
        draw.ellipse((x_position, 606, x_position + 8, 614), fill=color)
        draw.text((x_position + 15, 610), label, fill="black", anchor="lm", font=font(12))
        x_position += 105
    draw.text((55, 642), "RK4 crossings: real ≈ -2.785, imaginary ≈ ±2.83", fill="black", font=font(12))

    for panel, dt in zip(panels[1:], [0.045, 0.056]):
        rows = profiles[dt]
        times = sorted(set(row[0] for row in rows))
        values = {(row[0], int(row[1])): row[3] for row in rows}
        maximum = max(abs(value) for value in values.values())
        left, top, right, bottom = panel
        cell_width = (right - left) / 64
        cell_height = (bottom - top) / len(times)
        for ti, time in enumerate(times):
            for row in range(64):
                value = values[(time, row)]
                px = int(left + row * cell_width)
                py = int(top + ti * cell_height)
                draw.rectangle((px, py, int(px + cell_width + 1), int(py + cell_height + 1)), fill=diverging(value, maximum))
        label_axes(draw, panel, "x", "time", f"RK4 Gaussian, dt = {dt}", (0, 2 * math.pi), (6, 0))
        draw.text((left + 8, bottom + 25), "time increases downward", fill="black", font=font(12))
    draw.text((35, 10), "Part 1 line stability: spectral map and time-space histories", fill="black", font=font(23))
    image.save(EVIDENCE / "line-stability.png")


if __name__ == "__main__":
    main()
