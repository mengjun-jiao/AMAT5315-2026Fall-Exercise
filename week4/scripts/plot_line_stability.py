"""Render the compact three-panel line-stability evidence figure."""

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


def viridis(value):
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    value = max(0.0, min(1.0, value)) * 4
    i = min(int(value), 3)
    q = value - i
    return tuple(int(stops[i][j] * (1 - q) + stops[i + 1][j] * q) for j in range(3))


def signed_color(value, limit, logarithmic=False):
    if logarithmic:
        scale = math.copysign(math.asinh(abs(value) / 0.02) / math.asinh(max(limit, 0.02) / 0.02), value)
    else:
        scale = max(-1.0, min(1.0, value / max(limit, 1e-15)))
    if scale < 0:
        q = scale + 1
        return (int(48 * (1 - q) + 65 * q), int(65 * (1 - q) + 105 * q), int(160 * (1 - q) + 195 * q))
    return (255, int(245 * (1 - scale) + 75 * scale), int(245 * (1 - scale) + 70 * scale))


def draw_axes(draw, box, xlabel, ylabel, title, xticks, yticks):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, top - 15), title, fill="black", anchor="mm", font=f(14))
    draw.text(((left + right) / 2, bottom + 19), xlabel, fill="black", anchor="ma", font=f(11))
    draw.text((left - 27, (top + bottom) / 2), ylabel, fill="black", anchor="mm", font=f(11))
    for value, label in xticks:
        px, _ = xy(box, value, 0, (xticks[0][0], xticks[-1][0]), (0, 1))
        draw.line((px, bottom, px, bottom + 4), fill="black")
        draw.text((px, bottom + 5), label, fill="black", anchor="ma", font=f(9))
    for value, label in yticks:
        _, py = xy(box, 0, value, (0, 1), (yticks[0][0], yticks[-1][0]))
        draw.line((left - 4, py, left, py), fill="black")
        draw.text((left - 7, py), label, fill="black", anchor="rm", font=f(9))


def boundary(draw, function, box, color, dashed=False):
    xlim, ylim = (-4, 1), (-4, 4)
    previous = None
    for row in range(401):
        y = ylim[0] + 8 * row / 400
        found = None
        for col in range(401):
            x = -4 + 5 * col / 400
            if abs(abs(function(complex(x, y))) - 1) < 0.012:
                found = xy(box, x, y, xlim, ylim)
                break
        if found is not None and previous is not None and abs(found[0] - previous[0]) < 10:
            if not dashed or row % 10 < 6:
                draw.line((previous, found), fill=color, width=2)
        previous = found


def image_from_rows(rows, width, height, mapper):
    raw = Image.new("RGB", (len(rows[0]), len(rows)))
    pixels = raw.load()
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            pixels[x, y] = mapper(value)
    return raw.resize((width, height), Image.Resampling.BILINEAR)


def add_colorbar(draw, box, label):
    left, top, right, bottom = box
    for x in range(left, right):
        draw.line((x, top, x, bottom), fill=viridis((x - left) / max(1, right - left)))
    draw.rectangle(box, outline="black")
    draw.text((left, bottom + 4), "-3", fill="black", anchor="ma", font=f(8))
    draw.text(((left + right) / 2, bottom + 4), "0", fill="black", anchor="ma", font=f(8))
    draw.text((right, bottom + 4), "2", fill="black", anchor="ma", font=f(8))
    draw.text(((left + right) / 2, bottom + 17), label, fill="black", anchor="ma", font=f(9))


def main():
    EVIDENCE.mkdir(exist_ok=True)
    command = ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "week4" / "Cargo.toml"), "--bin", "line_stability_data"]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    grid = []
    spectra = {0.045: [], 0.056: []}
    histories = {0.045: [], 0.056: []}
    critical = None
    for text in io.StringIO(output):
        fields = text.rstrip().split(",")
        if fields[0] == "CRITICAL":
            critical = float(fields[1])
        elif fields[0] == "GRID":
            grid.append(tuple(map(float, fields[1:])))
        elif fields[0] == "SPECTRUM":
            spectra[float(fields[1])].append(tuple(map(float, fields[2:])))
        elif fields[0] == "LINE":
            histories[float(fields[1])].append(tuple(map(float, fields[2:])))
    print(f"Computed RK4 line stability limit: dt_crit = {critical:.12f}")

    image = Image.new("RGB", (1610, 505), "white")
    draw = ImageDraw.Draw(image)
    left, middle, right = (50, 55, 545, 390), (575, 55, 1070, 390), (1100, 55, 1595, 390)

    grid_image = image_from_rows(
        [[math.log10(max(grid[row * 241 + col][2], 1e-3)) for col in range(241)] for row in range(241)],
        left[2] - left[0], left[3] - left[1], lambda value: viridis((value + 3) / 5),
    )
    image.paste(grid_image, left[:2])
    draw_axes(draw, left, "Re(z)", "Im(z)", "RK4 stability map", [(-4, "-4"), (-2, "-2"), (0, "0"), (1, "1")], [(-4, "-4"), (-2, "-2"), (0, "0"), (2, "2"), (4, "4")])
    boundary(draw, lambda z: 1 + z, left, "black", True)
    boundary(draw, lambda z: 1 + z + z**2 / 2, left, "#df7f00", True)
    boundary(draw, lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24, left, "#b00020")
    for dt, color in [(0.045, "#00a8c8"), (0.056, "#d000c8")]:
        for _, real, imaginary, _ in spectra[dt]:
            px, py = xy(left, real * dt, imaginary * dt, (-4, 1), (-4, 4))
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color, outline="black")
    add_colorbar(draw, (205, 68, 370, 77), "log10 growth")
    draw.text((65, 105), "RK4: -2.785, ±2.83i", fill="black", font=f(9))
    legend = [("black", "Euler"), ("#df7f00", "midpoint"), ("#b00020", "RK4"), ("#00a8c8", "dt=.045"), ("#d000c8", "dt=.056")]
    for i, (color, label) in enumerate(legend):
        x = 65 + (i % 2) * 100
        y = 330 + (i // 2) * 15
        draw.line((x, y, x + 16, y), fill=color, width=2)
        draw.text((x + 20, y), label, fill="black", anchor="lm", font=f(9))

    data = {}
    for dt in [0.045, 0.056]:
        rows = histories[dt]
        times = sorted(set(row[0] for row in rows))
        values = {(row[0], int(row[1])): row[3] for row in rows}
        data[dt] = [[values[(time, row)] for row in range(64)] for time in times]
    stable_limit = max(abs(value) for row in data[0.045] for value in row)
    unstable_display_limit = max(abs(value) for row in data[0.056] for value in row if abs(value) < 100)
    for panel, dt, logarithmic, limit in [(middle, 0.045, False, stable_limit), (right, 0.056, True, unstable_display_limit)]:
        raster = image_from_rows(data[dt], panel[2] - panel[0], panel[3] - panel[1], lambda value: signed_color(value, limit, logarithmic))
        image.paste(raster, panel[:2])
        draw_axes(draw, panel, "x", "t", f"RK4, dt = {dt:.3f}", [(0, "0"), (math.pi, "π"), (2 * math.pi, "2π")], [(0, "0"), (3, "3"), (6, "6")])
        if logarithmic:
            draw.text((panel[0] + 8, panel[1] + 10), "symmetric-log display", fill="black", font=f(8))
    image.crop((0, 0, image.width, 430)).save(EVIDENCE / "line-stability.png", dpi=(140, 140))


if __name__ == "__main__":
    main()
