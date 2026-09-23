"""Render the compact Taylor-Green validation figure from stored Rust fields."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "week4" / "artifacts" / "taylor-green"
EVIDENCE = ROOT / "week4" / "evidence"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def f(size):
    return ImageFont.truetype(FONT, size)


def colour(value):
    scale = max(-1.0, min(1.0, value / 2.0))
    if scale < 0:
        q = scale + 1
        return (int(45 * (1 - q) + 70 * q), int(80 * (1 - q) + 110 * q), int(180 * (1 - q) + 205 * q))
    return (255, int(245 * (1 - scale) + 75 * scale), int(245 * (1 - scale) + 45 * scale))


def arrow(draw, start, end):
    draw.line((*start, *end), fill="black", width=2)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 6
    points = [end, (end[0] - size * math.cos(angle - 0.5), end[1] - size * math.sin(angle - 0.5)), (end[0] - size * math.cos(angle + 0.5), end[1] - size * math.sin(angle + 0.5))]
    draw.polygon(points, fill="black")


def main():
    with (RUN / "fields.jsonl").open() as stream:
        frames = [json.loads(line) for line in stream]
    with (RUN / "exact-t1.json").open() as stream:
        exact = json.load(stream)
    initial, final = frames[0], frames[-1]
    exact_norm = math.sqrt(sum(value * value for value in exact["u"]) + sum(value * value for value in exact["v"]))
    difference = math.sqrt(sum((a - b) ** 2 for a, b in zip(final["u"], exact["u"])) + sum((a - b) ** 2 for a, b in zip(final["v"], exact["v"])))
    print(f"Relative velocity error at t=1: {difference / exact_norm:.8e}")
    print(f"Stored max |omega|: t=0 {max(abs(value) for value in initial['omega']):.8f}, t=1 {max(abs(value) for value in final['omega']):.8f}")

    n = int(round(math.sqrt(len(initial["omega"]))))
    image = Image.new("RGB", (1190, 532), "white")
    draw = ImageDraw.Draw(image)
    panels = [(55, 48, 455, 448), (535, 48, 935, 448)]
    common_velocity = max(math.hypot(u, v) for frame in [initial, final] for u, v in zip(frame["u"], frame["v"]))
    for panel, frame in zip(panels, [initial, final]):
        left, top, right, bottom = panel
        raw = Image.new("RGB", (n, n))
        pixels = raw.load()
        for row in range(n):
            for col in range(n):
                pixels[col, row] = colour(frame["omega"][row * n + col])
        raster = raw.resize((right - left, bottom - top), Image.Resampling.BILINEAR)
        image.paste(raster, (left, top))
        cell_width, cell_height = (right - left) / n, (bottom - top) / n
        arrow_scale = 0.055 * min(right - left, bottom - top) / common_velocity
        for row in range(0, n, 4):
            for col in range(0, n, 4):
                index = row * n + col
                start = (left + (col + 0.5) * cell_width, bottom - (row + 0.5) * cell_height)
                target = (start[0] + frame["u"][index] * arrow_scale, start[1] - frame["v"][index] * arrow_scale)
                target = (max(left + 2, min(right - 2, target[0])), max(top + 2, min(bottom - 2, target[1])))
                arrow(draw, start, target)
        draw.rectangle(panel, outline="black")
        draw.text(((left + right) / 2, top - 17), f"t = {frame['t']:g}", fill="black", anchor="mm", font=f(14))
        draw.text(((left + right) / 2, bottom + 17), "x", fill="black", anchor="ma", font=f(10))
        if panel == panels[0]:
            draw.text((left - 24, (top + bottom) / 2), "y", fill="black", anchor="mm", font=f(10))
        for value, label in [(0, "0"), (math.pi, "π"), (2 * math.pi, "2π")]:
            px = int(left + value / (2 * math.pi) * (right - left))
            draw.text((px, bottom + 4), label, fill="black", anchor="ma", font=f(8))
            py = int(bottom - value / (2 * math.pi) * (bottom - top))
            if panel == panels[0]:
                draw.text((left - 6, py), label, fill="black", anchor="rm", font=f(8))

    cbar = (1000, 80, 1013, 415)
    for y in range(cbar[1], cbar[3]):
        draw.line((cbar[0], y, cbar[2], y), fill=colour(2 - 4 * (y - cbar[1]) / (cbar[3] - cbar[1])))
    draw.rectangle(cbar, outline="black")
    for value in [-2, -1, 0, 1, 2]:
        py = cbar[3] - int((value + 2) / 4 * (cbar[3] - cbar[1]))
        draw.text((cbar[2] + 6, py), f"{value:g}", fill="black", anchor="lm", font=f(8))
    draw.text((1038, 248), "vorticity", fill="black", anchor="mm", font=f(10))
    image.crop((0, 0, image.width, 465)).save(EVIDENCE / "taylor-green.png", dpi=(140, 140))


if __name__ == "__main__":
    main()
