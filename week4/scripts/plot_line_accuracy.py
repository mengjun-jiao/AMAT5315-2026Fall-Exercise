"""Render compact pulse-accuracy and temporal-convergence evidence."""

import io
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week4" / "evidence"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def f(size):
    return ImageFont.truetype(FONT, size)


def xy(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    return (int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)),
            int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)))


def logxy(box, x, y, xlim, ylim):
    return xy(box, math.log(x), math.log(y), (math.log(xlim[0]), math.log(xlim[1])), (math.log(ylim[0]), math.log(ylim[1])))


def regression(values):
    xs = [math.log(x) for x, _ in values]
    ys = [math.log(y) for _, y in values]
    xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
    return slope, yb - slope * xb


def axes(draw, box, title, xlabel, ylabel):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, top - 16), title, fill="black", anchor="mm", font=f(14))
    draw.text(((left + right) / 2, bottom + 19), xlabel, fill="black", anchor="ma", font=f(10))
    draw.multiline_text((left - 25, (top + bottom) / 2), ylabel.replace(" ", "\n"), fill="black", anchor="mm", align="center", spacing=0, font=f(9))


def legend(draw, entries, x, y, columns=1):
    for index, (color, label) in enumerate(entries):
        col = index % columns
        row = index // columns
        px = x + col * 190
        py = y + row * 14
        draw.line((px, py, px + 15, py), fill=color, width=2)
        draw.text((px + 19, py), label, fill="black", anchor="lm", font=f(8))


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
    slopes = {name: regression(values)[0] for name, values in convergence.items()}
    print("\nTemporal convergence errors and fitted slopes:")
    print("Method                  dt          max_error")
    for name, values in convergence.items():
        for dt, value in values:
            print(f"{name:22s} {dt:8.5f}   {value:.8e}")
        print(f"{name:22s} fitted slope = {slopes[name]:.6f}")

    image = Image.new("RGB", (1330, 505), "white")
    draw = ImageDraw.Draw(image)
    left_box, right_box = (58, 55, 625, 365), (700, 55, 1270, 365)
    colors = {"RK4 Fourier": "#1769aa", "RK4 centered finite difference": "#e07a00", "Euler Fourier": "#16803c", "Exact solution": "black"}
    all_y = [row[2] for rows in profiles.values() for row in rows] + [row[3] for row in profiles["RK4 Fourier"]]
    ylim = (min(all_y) - 0.03, max(all_y) + 0.03)
    axes(draw, left_box, "Pulse propagation at t = 2π", "x", "u")
    for name, rows in profiles.items():
        points = [xy(left_box, row[1], row[2], (0, 2 * math.pi), ylim) for row in rows]
        draw.line(points, fill=colors[name], width=2, joint="curve")
    exact = [xy(left_box, row[1], row[3], (0, 2 * math.pi), ylim) for row in profiles["RK4 Fourier"]]
    for i in range(0, len(exact) - 1, 2):
        draw.line((exact[i], exact[i + 1]), fill="black", width=2)
    for value, label in [(0, "0"), (math.pi / 2, "π/2"), (math.pi, "π"), (3 * math.pi / 2, "3π/2"), (2 * math.pi, "2π")]:
        px, _ = xy(left_box, value, ylim[0], (0, 2 * math.pi), ylim)
        draw.text((px, left_box[3] + 5), label, fill="black", anchor="ma", font=f(9))
    for value in [0, 0.5, 1.0]:
        _, py = xy(left_box, 0, value, (0, 2 * math.pi), ylim)
        draw.text((left_box[0] - 7, py), str(value), fill="black", anchor="rm", font=f(8))
    legend(draw, [(colors[name], name) for name in colors], 75, 115, columns=1)

    xlim = (0.0025, 0.02)
    values = [value for rows in convergence.values() for _, value in rows]
    ylim_log = (min(values) * 0.5, max(values) * 2)
    axes(draw, right_box, "Temporal convergence", "dt", "maximum error")
    conv_colors = ["#1769aa", "#e07a00", "#b00020", "#16803c"]
    entries = []
    for index, (name, values_for_method) in enumerate(convergence.items()):
        color = conv_colors[index]
        points = [logxy(right_box, dt, value, xlim, ylim_log) for dt, value in values_for_method]
        draw.line(points, fill=color, width=2)
        for px, py in points:
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        slope, intercept = regression(values_for_method)
        line_points = [logxy(right_box, dt, math.exp(intercept) * dt**slope, xlim, ylim_log) for dt in xlim]
        draw.line(line_points, fill=color, width=1)
        short = {"Forward Euler": "Euler", "Midpoint": "midpoint", "RK4": "RK4", "EqualWeightFourStage": "equal-weight"}[name]
        entries.append((color, f"{short}, q = {slope:.2f}"))
    for dt in [0.0025, 0.005, 0.01, 0.02]:
        px, _ = logxy(right_box, dt, ylim_log[0], xlim, ylim_log)
        draw.text((px, right_box[3] + 5), f"{dt:g}", fill="black", anchor="ma", font=f(9))
    for exponent in [-10, -8, -6, -4, -2]:
        value = 10**exponent
        if ylim_log[0] <= value <= ylim_log[1]:
            _, py = logxy(right_box, xlim[0], value, xlim, ylim_log)
            draw.text((right_box[0] - 7, py), f"1e{exponent:+d}", fill="black", anchor="rm", font=f(8))
    legend(draw, entries, 720, 112, columns=1)
    image.crop((0, 0, image.width, 405)).save(EVIDENCE / "line-accuracy.png", dpi=(140, 140))


if __name__ == "__main__":
    main()
