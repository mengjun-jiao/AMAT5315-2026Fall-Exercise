"""Analyze stored Part 3 Rust runs and render the three evidence figures."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week4" / "artifacts"
EVIDENCE = ROOT / "week4" / "evidence"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def color(value, limit):
    value = max(-limit, min(limit, value)) / limit
    if value < 0:
        t = value + 1
        return (int(40 * t + 245 * (1 - t)), int(80 * t + 245 * (1 - t)), int(190 * t + 245 * (1 - t)))
    return (int(245 * value + 245 * (1 - value)), int(90 * value + 245 * (1 - value)), int(40 * value + 245 * (1 - value)))


def read_fields(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def read_diagnostics(path):
    rows = []
    for line in path.read_text().splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 3:
            rows.append(tuple(float(value) for value in fields))
    return rows


def xy(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    return (int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)), int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)))


def plot_line(draw, points, colour, width=2):
    finite = [point for point in points if point is not None]
    if len(finite) > 1:
        draw.line(finite, fill=colour, width=width, joint="curve")


def random_figure():
    frames = read_fields(ARTIFACTS / "random" / "fields.jsonl")
    selected = [min(frames, key=lambda frame: abs(frame["t"] - target)) for target in [0.0, 2.0, 5.0, 10.0]]
    limit = max(abs(value) for value in selected[0]["omega"])
    image = Image.new("RGB", (1800, 550), "white")
    draw = ImageDraw.Draw(image)
    for panel_index, frame in enumerate(selected):
        left, top, right, bottom = 35 + panel_index * 440, 95, 425 + panel_index * 440, 485
        n = int(round(math.sqrt(len(frame["omega"]))))
        raster = Image.new("RGB", (n, n))
        pixels = raster.load()
        for row in range(n):
            for col in range(n):
                pixels[col, row] = color(frame["omega"][row * n + col], limit)
        raster = raster.resize((right - left, bottom - top), Image.Resampling.BILINEAR)
        image.paste(raster, (left, top))
        draw.rectangle((left, top, right, bottom), outline="black")
        z_value = 0.5 * sum(value * value for value in frame["omega"]) / len(frame["omega"])
        draw.text(((left + right) / 2, top - 28), f"t = {frame['t']:g}, E = {energy(frame):.4f}, Z = {z_value:.4f}", fill="black", anchor="mm", font=font(13))
        draw.text((left, bottom + 10), "0", fill="black", anchor="ra", font=font(12))
        draw.text((right, bottom + 10), "2π", fill="black", anchor="la", font=font(12))
        draw.text((left - 8, top), "0", fill="black", anchor="ra", font=font(12))
        draw.text((left - 8, bottom), "2π", fill="black", anchor="rd", font=font(12))
    for x in range(650, 1050):
        fraction = (x - 650) / 400
        draw.line((x, 480, x, 492), fill=color(-limit + 2 * limit * fraction, limit))
    draw.rectangle((650, 480, 1050, 492), outline="black")
    draw.text((650, 495), f"−{limit:.3f}", fill="black", anchor="la", font=font(10))
    draw.text((850, 495), "0", fill="black", anchor="ma", font=font(10))
    draw.text((1050, 495), f"{limit:.3f}", fill="black", anchor="ra", font=font(10))
    draw.text((1080, 486), "vorticity", fill="black", anchor="lm", font=font(11))
    image.save(EVIDENCE / "random.png")


def energy(frame):
    return 0.5 * sum(u * u + v * v for u, v in zip(frame["u"], frame["v"])) / len(frame["u"])


def chart_axes(draw, box, title, xlabel, ylabel, ylabel_inside=True):
    draw.rectangle(box, outline="black")
    left, top, right, bottom = box
    draw.text(((left + right) / 2, top - 25), title, fill="black", anchor="mm", font=font(19))
    draw.text(((left + right) / 2, bottom + 28), xlabel, fill="black", anchor="mm", font=font(15))
    if ylabel_inside:
        draw.text((left + 5, (top + bottom) / 2), ylabel, fill="black", anchor="lm", font=font(13))
    else:
        draw.text((left - 8, (top + bottom) / 2), ylabel, fill="black", anchor="rm", font=font(13))


def log_ticks(draw, box, x_max, y_min, y_max):
    left, top, right, bottom = box
    for time in [0, x_max / 2, x_max]:
        px, _ = xy(box, time, math.log(y_min), (0, x_max), (math.log(y_min), math.log(y_max)))
        draw.line((px, bottom, px, bottom + 5), fill="black")
        draw.text((px, bottom + 7), f"{time:g}", fill="black", anchor="ma", font=font(10))
    first = math.ceil(math.log10(y_min))
    last = math.floor(math.log10(y_max))
    for exponent in range(first, last + 1):
        value = 10 ** exponent
        _, py = xy(box, 0, math.log(value), (0, x_max), (math.log(y_min), math.log(y_max)))
        draw.line((left - 5, py, left, py), fill="black")
        draw.text((left - 8, py), f"1e{exponent:+d}", fill="black", anchor="rm", font=font(10))


def log_point(box, time, value, x_max, y_min, y_max):
    if not math.isfinite(value) or value <= 0:
        return None
    return xy(box, time, math.log(value), (0.0, x_max), (math.log(y_min), math.log(y_max)))


def blowup_figure():
    image = Image.new("RGB", (1600, 720), "white")
    draw = ImageDraw.Draw(image)
    boxes = [(85, 105, 725, 600), (875, 105, 1515, 600)]
    tg_runs = [("RK4 dt=0.032", "blue", read_diagnostics(ARTIFACTS / "scan/taylor-green-dt-0.032.tsv")), ("RK4 dt=0.033", "red", read_diagnostics(ARTIFACTS / "scan/taylor-green-dt-0.033.tsv"))]
    tg_y = [value for _, _, rows in tg_runs for _, value, _ in rows if math.isfinite(value)]
    tg_y.extend(0.25 * math.exp(-0.4 * t) for t in [0, 8])
    chart_axes(draw, boxes[0], "Taylor-Green diffusive stability", "time", "energy (log scale)")
    for name, colour, rows in tg_runs:
        points = [log_point(boxes[0], t, e, 8, min(tg_y) * 0.5, max(tg_y) * 2) for t, e, _ in rows]
        plot_line(draw, points, colour)
    exact = [log_point(boxes[0], t, 0.25 * math.exp(-0.4 * t), 8, min(tg_y) * 0.5, max(tg_y) * 2) for t in [i * 0.05 for i in range(161)]]
    for index in range(0, len(exact) - 1, 2):
        draw.line((exact[index], exact[index + 1]), fill="black", width=2)
    stop = next((t for t, e, _ in read_diagnostics(ARTIFACTS / "scan/taylor-green-dt-0.033.tsv") if not math.isfinite(e)), None)
    if stop is not None:
        px, _ = xy(boxes[0], stop, math.log(min(tg_y)), (0, 8), (math.log(min(tg_y) * 0.5), math.log(max(tg_y) * 2)))
        draw.line((px, boxes[0][1], px, boxes[0][3]), fill="red", width=1)
        draw.text((px + 5, boxes[0][1] + 8), f"stop {stop:.3g}", fill="red", font=font(12))
    log_ticks(draw, boxes[0], 8, min(tg_y) * 0.5, max(tg_y) * 2)
    draw.text((100, 125), "predicted diffusive limit = 0.0316", fill="black", font=font(12))
    random_runs = [("RK4 dt=0.035", "blue", read_diagnostics(ARTIFACTS / "scan/random-rk4-dt-0.035.tsv")), ("RK4 dt=0.038", "red", read_diagnostics(ARTIFACTS / "scan/random-rk4-dt-0.038.tsv")), ("Euler dt=0.010", "green", read_diagnostics(ARTIFACTS / "scan/random-euler-dt-0.010.tsv"))]
    random_y = [e for _, _, rows in random_runs for _, e, _ in rows if math.isfinite(e)]
    chart_axes(draw, boxes[1], "Random-flow stability", "time", "energy (log scale)")
    for name, colour, rows in random_runs:
        plot_line(draw, [log_point(boxes[1], t, e, 10, min(random_y) * 0.5, max(random_y) * 2) for t, e, _ in rows], colour)
    for name, colour, rows in random_runs[1:]:
        stop = next((t for t, e, _ in rows if not math.isfinite(e)), None)
        if stop is not None:
            px, _ = xy(boxes[1], stop, math.log(min(random_y)), (0, 10), (math.log(min(random_y) * 0.5), math.log(max(random_y) * 2)))
            draw.line((px, boxes[1][1], px, boxes[1][3]), fill=colour, width=1)
            draw.text((px + 5, boxes[1][1] + 8 + (0 if colour == "red" else 18)), f"stop {stop:.3g}", fill=colour, font=font(12))
    log_ticks(draw, boxes[1], 10, min(random_y) * 0.5, max(random_y) * 2)
    draw.text((890, 125), "advective bound = 0.02066; bracket = [0.035, 0.038]", fill="black", font=font(12))
    legend_runs = tg_runs + [("exact", "black", [])] + random_runs
    for index, (name, colour, _) in enumerate(legend_runs):
        x = 105 + index * 115 if index < 3 else 900 + (index - 3) * 130
        y = 655 if index < 2 else 655
        draw.line((x, y, x + 20, y), fill=colour, width=3)
        draw.text((x + 26, y), name, fill="black", anchor="lm", font=font(11))
    image.save(EVIDENCE / "blowup.png")


def sensitivity_figure():
    pairs = [("Taylor-Green", "taylor-green-original", "taylor-green-perturbed", "blue"), ("Random flow", "random-original", "random-perturbed", "red")]
    distances = {}
    for label, original_name, perturbed_name, colour in pairs:
        original = read_fields(ARTIFACTS / "sensitivity" / original_name / "fields.jsonl")
        perturbed = read_fields(ARTIFACTS / "sensitivity" / perturbed_name / "fields.jsonl")
        values = []
        for first, second in zip(original, perturbed):
            numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(first["omega"], second["omega"])))
            denominator = math.sqrt(sum(a * a for a in first["omega"]))
            values.append((first["t"], numerator / denominator))
        distances[label] = values
        print(f"{label} sensitivity distance: initial={values[0][1]:.8e}, final={values[-1][1]:.8e}, growth factor={values[-1][1] / values[0][1]:.8e}")
        original_energy = [energy(frame) for frame in original]
        perturbed_energy = [energy(frame) for frame in perturbed]
        print(f"{label} original energy: initial={original_energy[0]:.8f}, final={original_energy[-1]:.8f}")
        print(f"{label} perturbed energy: initial={perturbed_energy[0]:.8f}, final={perturbed_energy[-1]:.8f}")
    image = Image.new("RGB", (1000, 650), "white")
    draw = ImageDraw.Draw(image)
    box = (100, 90, 900, 540)
    all_values = [value for values in distances.values() for _, value in values if value > 0]
    chart_axes(draw, box, "Sensitivity to initial vorticity ripple", "time", "relative vorticity distance (log scale)", False)
    for label, colour in [("Taylor-Green", "blue"), ("Random flow", "red")]:
        points = [log_point(box, t, value, 20, min(all_values) * 0.5, max(all_values) * 2) for t, value in distances[label]]
        plot_line(draw, points, colour, 3)
    draw.line((120, 590, 145, 590), fill="blue", width=3)
    draw.text((155, 590), "Taylor-Green", fill="black", anchor="lm", font=font(13))
    draw.line((300, 590, 325, 590), fill="red", width=3)
    draw.text((335, 590), "Random flow", fill="black", anchor="lm", font=font(13))
    image.save(EVIDENCE / "sensitivity.png")


def main():
    EVIDENCE.mkdir(exist_ok=True)
    reference = read_diagnostics(ARTIFACTS / "random.tsv")
    print(f"Random reference E(0)={reference[0][1]:.8f}, Z(0)={reference[0][2]:.8f}, E(10)={reference[-1][1]:.8f}, Z(10)={reference[-1][2]:.8f}")
    print(f"Random enstrophy reduction factor: {reference[0][2] / reference[-1][2]:.8f}")
    print(f"Random retained energy fraction: {reference[-1][1] / reference[0][1]:.8f}")
    random_figure()
    blowup_figure()
    sensitivity_figure()


if __name__ == "__main__":
    main()
