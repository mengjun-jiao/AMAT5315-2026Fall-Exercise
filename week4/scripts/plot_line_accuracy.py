"""Render Part 1 line-accuracy evidence using data emitted by Rust."""

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
    return (int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)), int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)))


def line(draw, points, color, width=2):
    if len(points) > 1:
        draw.line(points, fill=color, width=width, joint="curve")


def regression(values):
    xs = [math.log(item[0]) for item in values]
    ys = [math.log(item[1]) for item in values]
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum((x - xbar) ** 2 for x in xs)


def main():
    EVIDENCE.mkdir(exist_ok=True)
    command = ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "week4" / "Cargo.toml"), "--bin", "line_accuracy_data"]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    profiles, errors, convergence = {}, {}, {}
    for line_text in io.StringIO(output):
        fields = line_text.rstrip().split(",")
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

    image = Image.new("RGB", (1500, 700), "white")
    draw = ImageDraw.Draw(image)
    left_box, right_box = (80, 105, 710, 590), (810, 105, 1440, 590)
    colors = {"RK4 Fourier": "blue", "RK4 centered finite difference": "orange", "Euler Fourier": "green", "Exact solution": "black"}
    y_values = [row[2] for rows in profiles.values() for row in rows] + [row[3] for row in profiles["RK4 Fourier"]]
    ylim = (min(y_values) - 0.03, max(y_values) + 0.03)
    for name, rows in profiles.items():
        line(draw, [map_xy(left_box, row[1], row[2], (0, 2 * math.pi), ylim) for row in rows], colors[name])
    line(draw, [map_xy(left_box, row[1], row[3], (0, 2 * math.pi), ylim) for row in profiles["RK4 Fourier"]], "black")
    draw.rectangle(left_box, outline="black")
    draw.text((395, 65), "Pulse propagation at t = 2π", fill="black", anchor="mm", font=font(20))
    draw.text((395, 625), "x", fill="black", anchor="mm", font=font(16))
    draw.text((30, 350), "u", fill="black", anchor="mm", font=font(16))
    for index, (name, color) in enumerate(colors.items()):
        draw.line((100, 640 + index * 17, 125, 640 + index * 17), fill=color, width=3)
        draw.text((135, 640 + index * 17), name, fill="black", anchor="lm", font=font(12))

    xmin, xmax = math.log(0.0025), math.log(0.02)
    ymin = min(math.log(value) for values in convergence.values() for _, value in values)
    ymax = max(math.log(value) for values in convergence.values() for _, value in values)
    draw.rectangle(right_box, outline="black")
    for index, (name, values) in enumerate(convergence.items()):
        color = ["blue", "orange", "red", "green"][index]
        points = [map_xy(right_box, math.log(dt), math.log(value), (xmin, xmax), (ymin - 0.2, ymax + 0.2)) for dt, value in values]
        line(draw, points, color)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
    draw.text((1125, 65), "Temporal convergence", fill="black", anchor="mm", font=font(20))
    draw.text((1125, 625), "dt (log scale)", fill="black", anchor="mm", font=font(16))
    draw.text((760, 350), "error", fill="black", anchor="mm", font=font(16))
    for index, (name, slope) in enumerate(slopes.items()):
        color = ["blue", "orange", "red", "green"][index]
        draw.line((820, 640 + index * 17, 845, 640 + index * 17), fill=color, width=3)
        draw.text((855, 640 + index * 17), f"{name}, slope {slope:.2f}", fill="black", anchor="lm", font=font(12))
    draw.text((55, 25), "Part 1 line accuracy", fill="black", font=font(23))
    image.save(EVIDENCE / "line-accuracy.png")


if __name__ == "__main__":
    main()
