"""Render compact random-flow stability and sensitivity evidence."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "week4" / "artifacts"
EVIDENCE = ROOT / "week4" / "evidence"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def f(size):
    return ImageFont.truetype(FONT, size)


def field_color(value, limit):
    scale = max(-1.0, min(1.0, value / max(limit, 1e-15)))
    if scale < 0:
        q = scale + 1
        return (int(45 * (1 - q) + 70 * q), int(80 * (1 - q) + 110 * q), int(180 * (1 - q) + 205 * q))
    return (255, int(245 * (1 - scale) + 75 * scale), int(245 * (1 - scale) + 45 * scale))


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
    return (int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)),
            int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)))


def log_point(box, time, value, x_max, y_min, y_max):
    if not math.isfinite(value) or value <= 0:
        return None
    return xy(box, time, math.log(value), (0, x_max), (math.log(y_min), math.log(y_max)))


def axes(draw, box, title, xlabel, ylabel, x_max, y_min, y_max):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, top - 15), title, fill="black", anchor="mm", font=f(13))
    draw.text(((left + right) / 2, bottom + 18), xlabel, fill="black", anchor="ma", font=f(9))
    draw.multiline_text((left - 25, (top + bottom) / 2), ylabel.replace(" ", "\n"), fill="black", anchor="mm", align="center", spacing=0, font=f(8))
    for time in [0, x_max / 2, x_max]:
        px, _ = xy(box, time, math.log(y_min), (0, x_max), (math.log(y_min), math.log(y_max)))
        draw.text((px, bottom + 4), f"{time:g}", fill="black", anchor="ma", font=f(8))
    for exponent in range(math.ceil(math.log10(y_min)), math.floor(math.log10(y_max)) + 1):
        value = 10**exponent
        _, py = xy(box, 0, math.log(value), (0, x_max), (math.log(y_min), math.log(y_max)))
        draw.text((left - 6, py), f"1e{exponent:+d}", fill="black", anchor="rm", font=f(8))


def legend(draw, entries, x, y, columns=1):
    for index, (color, label) in enumerate(entries):
        col, row = index % columns, index // columns
        px, py = x + col * 170, y + row * 14
        draw.line((px, py, px + 15, py), fill=color, width=2)
        draw.text((px + 19, py), label, fill="black", anchor="lm", font=f(8))


def energy(frame):
    return 0.5 * sum(u * u + v * v for u, v in zip(frame["u"], frame["v"])) / len(frame["u"])


def random_figure():
    frames = read_fields(ARTIFACTS / "random" / "fields.jsonl")
    selected = [min(frames, key=lambda frame: abs(frame["t"] - target)) for target in [0, 2, 5, 10]]
    limit = max(abs(value) for value in selected[0]["omega"])
    image = Image.new("RGB", (1540, 405), "white")
    draw = ImageDraw.Draw(image)
    for index, frame in enumerate(selected):
        left, top, right, bottom = (35 + index * 365, 55, 335 + index * 365, 355)
        n = int(round(math.sqrt(len(frame["omega"]))))
        raw = Image.new("RGB", (n, n))
        pixels = raw.load()
        for row in range(n):
            for col in range(n):
                pixels[col, row] = field_color(frame["omega"][row * n + col], limit)
        image.paste(raw.resize((right - left, bottom - top), Image.Resampling.BILINEAR), (left, top))
        draw.rectangle((left, top, right, bottom), outline="black")
        z_value = 0.5 * sum(value * value for value in frame["omega"]) / len(frame["omega"])
        draw.text(((left + right) / 2, top - 25), f"t = {frame['t']:g}", fill="black", anchor="mm", font=f(11))
        draw.text(((left + right) / 2, top - 11), f"E = {energy(frame):.4f}, Z = {z_value:.4f}", fill="black", anchor="mm", font=f(9))
        draw.text((left, bottom + 4), "0", fill="black", anchor="ra", font=f(8))
        draw.text((right, bottom + 4), "2π", fill="black", anchor="la", font=f(8))
        if index == 0:
            draw.text((left - 6, top), "0", fill="black", anchor="ra", font=f(8))
            draw.text((left - 6, bottom), "2π", fill="black", anchor="rd", font=f(8))
    cbar = (1490, 75, 1503, 335)
    for y in range(cbar[1], cbar[3]):
        draw.line((cbar[0], y, cbar[2], y), fill=field_color(limit - 2 * limit * (y - cbar[1]) / (cbar[3] - cbar[1]), limit))
    draw.rectangle(cbar, outline="black")
    for value in [-limit, 0, limit]:
        py = cbar[3] - int((value + limit) / (2 * limit) * (cbar[3] - cbar[1]))
        draw.text((cbar[2] + 5, py), f"{value:.2f}", fill="black", anchor="lm", font=f(8))
    draw.text((1496, 61), "vorticity", fill="black", anchor="ma", font=f(9))
    image.crop((0, 0, image.width, 380)).save(EVIDENCE / "random.png", dpi=(140, 140))


def blowup_figure():
    image = Image.new("RGB", (1260, 505), "white")
    draw = ImageDraw.Draw(image)
    boxes = [(58, 55, 585, 365), (675, 55, 1202, 365)]
    tg_runs = [("RK4 dt=0.032", "#123fca", read_diagnostics(ARTIFACTS / "scan/taylor-green-dt-0.032.tsv")), ("RK4 dt=0.033", "#d01818", read_diagnostics(ARTIFACTS / "scan/taylor-green-dt-0.033.tsv"))]
    tg_values = [e for _, _, rows in tg_runs for _, e, _ in rows if math.isfinite(e)] + [0.25 * math.exp(-0.4 * t) for t in [0, 8]]
    tg_min, tg_max = min(tg_values) * 0.5, max(tg_values) * 2
    axes(draw, boxes[0], "Taylor-Green diffusive stability", "t", "E(t)", 8, tg_min, tg_max)
    for _, color, rows in tg_runs:
        points = [log_point(boxes[0], t, e, 8, tg_min, tg_max) for t, e, _ in rows]
        points = [point for point in points if point]
        draw.line(points, fill=color, width=2)
    exact = [log_point(boxes[0], i * 0.05, 0.25 * math.exp(-0.4 * i * 0.05), 8, tg_min, tg_max) for i in range(161)]
    for i in range(0, len(exact) - 1, 2):
        draw.line((exact[i], exact[i + 1]), fill="black", width=2)
    stop = next((t for t, e, _ in tg_runs[1][2] if not math.isfinite(e)), None)
    if stop is not None:
        px, _ = xy(boxes[0], stop, math.log(tg_min), (0, 8), (math.log(tg_min), math.log(tg_max)))
        draw.line((px, boxes[0][1], px, boxes[0][3]), fill="#d01818")
        draw.text((px - 4, boxes[0][1] + 10), f"stop {stop:.2f}", fill="#d01818", anchor="ra", font=f(8))
    draw.text((boxes[0][0] + 10, boxes[0][1] + 12), "predicted limit = 0.0316", fill="black", font=f(8))
    legend(draw, [("#123fca", "RK4, dt=.032"), ("#d01818", "RK4, dt=.033"), ("black", "exact")], boxes[0][0] + 12, boxes[0][3] - 35)

    random_runs = [("RK4 dt=0.035", "#123fca", read_diagnostics(ARTIFACTS / "scan/random-rk4-dt-0.035.tsv")), ("RK4 dt=0.038", "#d01818", read_diagnostics(ARTIFACTS / "scan/random-rk4-dt-0.038.tsv")), ("Euler dt=0.010", "#16803c", read_diagnostics(ARTIFACTS / "scan/random-euler-dt-0.010.tsv"))]
    random_values = [e for _, _, rows in random_runs for _, e, _ in rows if math.isfinite(e)]
    random_min, random_max = min(random_values) * 0.5, max(random_values) * 2
    axes(draw, boxes[1], "Random-flow stability", "t", "E(t)", 10, random_min, random_max)
    for _, color, rows in random_runs:
        points = [log_point(boxes[1], t, e, 10, random_min, random_max) for t, e, _ in rows]
        draw.line([point for point in points if point], fill=color, width=2)
    for _, color, rows in random_runs[1:]:
        stop = next((t for t, e, _ in rows if not math.isfinite(e)), None)
        if stop is not None:
            px, _ = xy(boxes[1], stop, math.log(random_min), (0, 10), (math.log(random_min), math.log(random_max)))
            draw.line((px, boxes[1][1], px, boxes[1][3]), fill=color)
            draw.text((px + 4, boxes[1][1] + 10), f"stop {stop:.2f}", fill=color, font=f(8))
    draw.text((boxes[1][0] + 10, boxes[1][1] + 12), "advective bound = 0.02066", fill="black", font=f(8))
    legend(draw, [(color, name) for name, color, _ in random_runs], boxes[1][0] + 12, boxes[1][3] - 35)
    image.crop((0, 0, image.width, 405)).save(EVIDENCE / "blowup.png", dpi=(140, 140))


def sensitivity_figure():
    pairs = [("Taylor-Green", "taylor-green-original", "taylor-green-perturbed", "#123fca"), ("Random flow", "random-original", "random-perturbed", "#d01818")]
    distances = {}
    for label, original_name, perturbed_name, color in pairs:
        original = read_fields(ARTIFACTS / "sensitivity" / original_name / "fields.jsonl")
        perturbed = read_fields(ARTIFACTS / "sensitivity" / perturbed_name / "fields.jsonl")
        values = []
        for first, second in zip(original, perturbed):
            numerator = math.sqrt(sum((a - b) ** 2 for a, b in zip(first["omega"], second["omega"])))
            denominator = math.sqrt(sum(a * a for a in first["omega"]))
            values.append((first["t"], numerator / denominator))
        distances[label] = values
        print(f"{label} sensitivity distance: initial={values[0][1]:.8e}, final={values[-1][1]:.8e}, growth factor={values[-1][1] / values[0][1]:.8e}")
        print(f"{label} original energy: initial={energy(original[0]):.8f}, final={energy(original[-1]):.8f}")
        print(f"{label} perturbed energy: initial={energy(perturbed[0]):.8f}, final={energy(perturbed[-1]):.8f}")
    image = Image.new("RGB", (770, 532), "white")
    draw = ImageDraw.Draw(image)
    box = (76, 55, 735, 425)
    all_values = [value for values in distances.values() for _, value in values if value > 0]
    ymin, ymax = min(all_values) * 0.5, max(all_values) * 2
    axes(draw, box, "Sensitivity to initial vorticity", "t", "relative vorticity difference", 20, ymin, ymax)
    for label, color in [("Taylor-Green", "#123fca"), ("Random flow", "#d01818")]:
        points = [log_point(box, t, value, 20, ymin, ymax) for t, value in distances[label] if value > 0]
        draw.line(points, fill=color, width=2)
    legend(draw, [("#123fca", "Taylor-Green"), ("#d01818", "Random flow")], box[0] + 15, box[1] + 18)
    image.crop((0, 0, image.width, 465)).save(EVIDENCE / "sensitivity.png", dpi=(140, 140))


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
