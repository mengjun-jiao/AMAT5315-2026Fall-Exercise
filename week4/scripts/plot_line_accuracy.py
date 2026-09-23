"""Render compact line-accuracy evidence from the Rust data producer."""

import io
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week4" / "evidence"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def map_xy(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    return (int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)),
            int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)))


def regression(values):
    xs = [math.log(item[0]) for item in values]
    ys = [math.log(item[1]) for item in values]
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)


def scientific(value):
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def draw_axes(draw, box, xlabel, ylabel, title):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, top - 18), title, fill="black", anchor="mm", font=font(16))
    draw.text(((left + right) / 2, bottom + 25), xlabel, fill="black", anchor="ma", font=font(14))
    draw.text((left - 42, (top + bottom) / 2), ylabel, fill="black", anchor="mm", font=font(14))


def main():
    EVIDENCE.mkdir(exist_ok=True)
    command = ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "week4" / "Cargo.toml"), "--bin", "line_accuracy_data"]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    profiles, errors, convergence = {}, {}, {}
    for text in io.StringIO(output):
        fields = text.rstrip().split(",")
        if fields[0] == "PROFILE":
            errors[fields[1]] = float(fields[2])
        elif fields[0] == "PROFILE_VALUE":
            profiles.setdefault(fields[1], []).append(tuple(map(float, fields[2:])))
        elif fields[0] == "CONVERGENCE":
            convergence.setdefault(fields[1], []).append((float(fields[2]), float(fields[3])))
    print("Pulse propagation maximum absolute errors:")
    for name, value in errors.items():
        print(f"  {name}: {value:.8e}")
    slopes = {name: regression(values) for name, values in convergence.items()}
    print("\nTemporal convergence errors and fitted slopes:")
    print("Method                  dt          max_error")
    for name, values in convergence.items():
        for dt, value in values:
            print(f"{name:22s} {dt:8.5f}   {value:.8e}")
        print(f"{name:22s} fitted slope = {slopes[name]:.6f}")

    image = Image.new("RGB", (1280, 570), "white")
    draw = ImageDraw.Draw(image)
    left_box, right_box = (70, 62, 585, 390), (690, 62, 1205, 390)
    colors = {"RK4 Fourier": "#1769aa", "RK4 centered finite difference": "#e07a00", "Euler Fourier": "#16803c", "Exact solution": "black"}
    y_values = [row[2] for rows in profiles.values() for row in rows] + [row[3] for row in profiles["RK4 Fourier"]]
    ylim = (min(y_values) - 0.03, max(y_values) + 0.03)
    xlim = (0, 2 * math.pi)
    draw_axes(draw, left_box, "x", "u", "Pulse propagation at t = 2π")
    for name, rows in profiles.items():
        points = [map_xy(left_box, row[1], row[2], xlim, ylim) for row in rows]
        draw.line(points, fill=colors[name], width=2, joint="curve")
    exact = [map_xy(left_box, row[1], row[3], xlim, ylim) for row in profiles["RK4 Fourier"]]
    for start in range(0, len(exact) - 1, 2):
        draw.line((exact[start], exact[start + 1]), fill="black", width=2)
    for value, label in [(0, "0"), (math.pi / 2, "π/2"), (math.pi, "π"), (3 * math.pi / 2, "3π/2"), (2 * math.pi, "2π")]:
        px, _ = map_xy(left_box, value, ylim[0], xlim, ylim)
        draw.text((px, left_box[3] + 7), label, fill="black", anchor="ma", font=font(11))
    for value in [0, 0.5, 1.0]:
        _, py = map_xy(left_box, 0, value, xlim, ylim)
        draw.text((left_box[0] - 7, py), f"{value:g}", fill="black", anchor="rm", font=font(11))
    legend = [(colors[name], name) for name in colors]
    for index, (color, name) in enumerate(legend):
        x, y = 95 + (index % 2) * 235, 425 + (index // 2) * 22
        draw.line((x, y, x + 20, y), fill=color, width=3)
        draw.text((x + 27, y), name, fill="black", anchor="lm", font=font(10))

    xmin, xmax = math.log(0.0025), math.log(0.02)
    ymin = min(math.log(value) for values in convergence.values() for _, value in values)
    ymax = max(math.log(value) for values in convergence.values() for _, value in values)
    right_ylim = (ymin - 0.25, ymax + 0.25)
    draw_axes(draw, right_box, "dt", "error", "Temporal convergence")
    colors_conv = ["#1769aa", "#e07a00", "#b00020", "#16803c"]
    for index, (name, values) in enumerate(convergence.items()):
        color = colors_conv[index]
        points = [map_xy(right_box, math.log(dt), math.log(value), (xmin, xmax), right_ylim) for dt, value in values]
        draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
        slope = slopes[name]
        x0, x1 = xmin, xmax
        y0 = math.log(values[0][1]) + slope * (x0 - math.log(values[0][0]))
        y1 = math.log(values[0][1]) + slope * (x1 - math.log(values[0][0]))
        draw.line((map_xy(right_box, x0, y0, (xmin, xmax), right_ylim), map_xy(right_box, x1, y1, (xmin, xmax), right_ylim)), fill=color, width=1)
    for dt in [0.0025, 0.005, 0.01, 0.02]:
        px, _ = map_xy(right_box, math.log(dt), right_ylim[0], (xmin, xmax), right_ylim)
        draw.text((px, right_box[3] + 7), f"{dt:g}", fill="black", anchor="ma", font=font(11))
    for exponent in [-10, -8, -6, -4, -2]:
        value = 10**exponent
        if right_ylim[0] <= math.log(value) <= right_ylim[1]:
            _, py = map_xy(right_box, xmin, math.log(value), (xmin, xmax), right_ylim)
            draw.text((right_box[0] - 8, py), scientific(value), fill="black", anchor="rm", font=font(10))
    for index, (name, slope) in enumerate(slopes.items()):
        x, y = 720 + (index % 2) * 220, 425 + (index // 2) * 22
        draw.line((x, y, x + 18, y), fill=colors_conv[index], width=3)
        draw.text((x + 24, y), f"{name}: slope {slope:.2f}", fill="black", anchor="lm", font=font(10))
    image.save(EVIDENCE / "line-accuracy.png", dpi=(180, 180))


if __name__ == "__main__":
    main()
