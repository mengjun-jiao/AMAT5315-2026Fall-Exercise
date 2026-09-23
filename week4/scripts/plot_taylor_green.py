"""Compare stored Rust Taylor-Green fields and render the validation figure."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "week4" / "artifacts" / "taylor-green"
EVIDENCE = ROOT / "week4" / "evidence"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def vorticity_color(value):
    value = max(-2.0, min(2.0, value)) / 2.0
    if value < 0:
        t = value + 1.0
        return (int(35 * t + 245 * (1 - t)), int(80 * t + 245 * (1 - t)), int(180 * t + 245 * (1 - t)))
    return (int(245 * value + 245 * (1 - value)), int(80 * value + 245 * (1 - value)), int(35 * value + 245 * (1 - value)))


def arrow(draw, start, end, color):
    draw.line((*start, *end), fill=color, width=2)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 7
    points = [end, (end[0] - size * math.cos(angle - 0.5), end[1] - size * math.sin(angle - 0.5)), (end[0] - size * math.cos(angle + 0.5), end[1] - size * math.sin(angle + 0.5))]
    draw.polygon(points, fill=color)


def main():
    with (RUN / "fields.jsonl").open() as stream:
        frames = [json.loads(line) for line in stream]
    with (RUN / "exact-t1.json").open() as stream:
        exact = json.load(stream)
    initial, final = frames[0], frames[-1]
    numerical_norm = math.sqrt(sum(value * value for value in final["u"]) + sum(value * value for value in final["v"]))
    difference_norm = math.sqrt(sum((a - b) ** 2 for a, b in zip(final["u"], exact["u"])) + sum((a - b) ** 2 for a, b in zip(final["v"], exact["v"])))
    print(f"Relative velocity error at t=1: {difference_norm / math.sqrt(sum(value * value for value in exact['u']) + sum(value * value for value in exact['v'])):.8e}")
    print(f"Stored max |omega|: t=0 {max(abs(value) for value in initial['omega']):.8f}, t=1 {max(abs(value) for value in final['omega']):.8f}")

    n = int(round(math.sqrt(len(initial["omega"]))))
    image = Image.new("RGB", (1400, 720), "white")
    draw = ImageDraw.Draw(image)
    panels = [(80, 105, 610, 610), (730, 105, 1260, 610)]
    for panel, frame in zip(panels, [initial, final]):
        left, top, right, bottom = panel
        cell_width = (right - left) / n
        cell_height = (bottom - top) / n
        for row in range(n):
            for col in range(n):
                value = frame["omega"][row * n + col]
                x0 = int(left + col * cell_width)
                y0 = int(bottom - (row + 1) * cell_height)
                x1 = int(left + (col + 1) * cell_width + 1)
                y1 = int(bottom - row * cell_height + 1)
                draw.rectangle((x0, y0, x1, y1), fill=vorticity_color(value))
        spacing = 4
        max_velocity = max(math.hypot(u, v) for u, v in zip(frame["u"], frame["v"]))
        arrow_scale = 0.06 * min(right - left, bottom - top) / max_velocity
        for row in range(0, n, spacing):
            for col in range(0, n, spacing):
                index = row * n + col
                start = (left + (col + 0.5) * cell_width, bottom - (row + 0.5) * cell_height)
                end = (start[0] + frame["u"][index] * arrow_scale, start[1] - frame["v"][index] * arrow_scale)
                arrow(draw, start, end, "black")
        draw.rectangle(panel, outline="black", width=2)
        draw.text(((left + right) / 2, top - 32), f"Taylor-Green flow, t = {frame['t']:g}", fill="black", anchor="mm", font=font(20))
        draw.text(((left + right) / 2, bottom + 30), "x", fill="black", anchor="mm", font=font(16))
        draw.text((left - 32, (top + bottom) / 2), "y", fill="black", anchor="mm", font=font(16))
        draw.text((left, bottom + 8), "0", fill="black", anchor="ra", font=font(12))
        draw.text((right, bottom + 8), "2π", fill="black", anchor="la", font=font(12))
        draw.text((left - 8, bottom), "0", fill="black", anchor="ra", font=font(12))
        draw.text((left - 8, top), "2π", fill="black", anchor="rd", font=font(12))

    draw.text((70, 35), "Part 2 Taylor-Green validation", fill="black", font=font(25))
    draw.text((80, 660), "Vorticity colour scale: fixed range [-2, 2]; black arrows show velocity", fill="black", font=font(15))
    legend_left = 1010
    for index, value in enumerate([-2, -1, 0, 1, 2]):
        x = legend_left + index * 45
        draw.rectangle((x, 650, x + 45, 666), fill=vorticity_color(value))
        draw.text((x + 22, 678), f"{value:g}", fill="black", anchor="ma", font=font(11))
    image.save(EVIDENCE / "taylor-green.png")


if __name__ == "__main__":
    main()
