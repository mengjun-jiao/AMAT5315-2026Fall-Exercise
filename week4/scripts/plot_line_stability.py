"""Render compact line-stability evidence from the Rust data producer."""

import io
import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week4" / "evidence"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def viridis(value):
    stops = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    value = max(0.0, min(1.0, value)) * (len(stops) - 1)
    index = min(int(value), len(stops) - 2)
    fraction = value - index
    return tuple(int(stops[index][i] * (1 - fraction) + stops[index + 1][i] * fraction) for i in range(3))


def diverging(value, maximum):
    if maximum == 0:
        return (245, 245, 245)
    scale = math.copysign(math.log1p(abs(value) / 1e-3) / math.log1p(maximum / 1e-3), value)
    if scale < 0:
        t = scale + 1
        return (int(49 * (1 - t) + 40 * t), int(54 * (1 - t) + 100 * t), int(149 * (1 - t) + 190 * t))
    return (255, int(245 * (1 - scale) + 90 * scale), int(245 * (1 - scale) + 80 * scale))


def point(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    return (int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)),
            int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)))


def axes(draw, box, xlabel, ylabel, title, xlim, ylim, xticks=None, yticks=None):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, bottom + 22), xlabel, fill="black", anchor="ma", font=font(14))
    draw.text((left - 38, (top + bottom) / 2), ylabel, fill="black", anchor="mm", font=font(14))
    draw.text(((left + right) / 2, top - 18), title, fill="black", anchor="mm", font=font(16))
    for value, label in (xticks or [(xlim[0], f"{xlim[0]:g}"), (xlim[1], f"{xlim[1]:g}")]):
        px, _ = point(box, value, ylim[0], xlim, ylim)
        draw.line((px, bottom, px, bottom + 4), fill="black")
        draw.text((px, bottom + 6), label, fill="black", anchor="ma", font=font(11))
    for value, label in (yticks or [(ylim[0], f"{ylim[0]:g}"), (ylim[1], f"{ylim[1]:g}")]):
        _, py = point(box, xlim[0], value, xlim, ylim)
        draw.line((left - 4, py, left, py), fill="black")
        draw.text((left - 7, py), label, fill="black", anchor="rm", font=font(11))


def boundary(draw, function, box, xlim, ylim, color, width=2):
    previous = None
    for row in range(801):
        y = ylim[0] + (ylim[1] - ylim[0]) * row / 800
        for col in range(801):
            x = xlim[0] + (xlim[1] - xlim[0]) * col / 800
            if abs(abs(function(complex(x, y))) - 1) < 0.006:
                current = point(box, x, y, xlim, ylim)
                if previous is not None and abs(current[0] - previous[0]) < 5:
                    draw.line((previous, current), fill=color, width=width)
                previous = current
                break
        else:
            previous = None


def colorbar(draw, box, lo, hi, label, logarithmic=False):
    left, top, right, bottom = box
    for x in range(left, right + 1):
        value = (x - left) / max(1, right - left)
        draw.line((x, top, x, bottom), fill=viridis(value))
    draw.rectangle(box, outline="black")
    draw.text((left, bottom + 5), f"{lo:g}", fill="black", anchor="ma", font=font(10))
    draw.text(((left + right) / 2, bottom + 5), "1", fill="black", anchor="ma", font=font(10))
    draw.text((right, bottom + 5), f"{hi:g}", fill="black", anchor="ma", font=font(10))
    draw.text(((left + right) / 2, bottom + 22), label, fill="black", anchor="ma", font=font(11))


def raster(rows, width, height, mapper):
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            pixels[x, y] = mapper(value)
    return image


def main():
    EVIDENCE.mkdir(exist_ok=True)
    command = ["cargo", "run", "--quiet", "--manifest-path", str(ROOT / "week4" / "Cargo.toml"), "--bin", "line_stability_data"]
    output = subprocess.check_output(command, cwd=ROOT, text=True)
    grid = []
    spectra = {0.045: [], 0.056: []}
    histories = {0.045: [], 0.056: []}
    critical = None
    for line in io.StringIO(output):
        fields = line.rstrip().split(",")
        if fields[0] == "CRITICAL":
            critical = float(fields[1])
        elif fields[0] == "GRID":
            grid.append(tuple(map(float, fields[1:])))
        elif fields[0] == "SPECTRUM":
            spectra[float(fields[1])].append(tuple(map(float, fields[2:])))
        elif fields[0] == "LINE":
            histories[float(fields[1])].append(tuple(map(float, fields[2:])))
    print(f"Computed RK4 line stability limit: dt_crit = {critical:.12f}")

    image = Image.new("RGB", (1500, 590), "white")
    draw = ImageDraw.Draw(image)
    panels = [(55, 58, 475, 375), (535, 58, 955, 375), (1015, 58, 1435, 375)]
    xlim, ylim = (-4.0, 1.0), (-4.0, 4.0)
    growth = raster([[math.log10(max(value[2], 1e-3)) for value in grid[row * 241:(row + 1) * 241]] for row in range(241)], 241, 241,
                    lambda value: viridis((value + 3.0) / 5.0)).resize((420, 317), Image.Resampling.BILINEAR)
    image.paste(growth, panels[0][:2])
    axes(draw, panels[0], "Re(z)", "Im(z)", "Measured RK4 growth factor", xlim, ylim,
         [(-4, "-4"), (-2, "-2"), (0, "0"), (1, "1")], [(-4, "-4"), (-2, "-2"), (0, "0"), (2, "2"), (4, "4")])
    boundary(draw, lambda z: 1 + z, panels[0], xlim, ylim, "black", 2)
    boundary(draw, lambda z: 1 + z + z**2 / 2, panels[0], xlim, ylim, "#e07a00", 2)
    boundary(draw, lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24, panels[0], xlim, ylim, "#b00020", 3)
    for dt, color in [(0.045, "cyan"), (0.056, "magenta")]:
        for _, real, imaginary, _ in spectra[dt]:
            px, py = point(panels[0], real * dt, imaginary * dt, xlim, ylim)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color, outline="black")
    colorbar(draw, (150, 405, 380, 417), -3, 2, "log10 measured growth factor")
    draw.text((55, 455), "crossings: Re ≈ -2.785; Im ≈ ±2.83", fill="black", font=font(12))
    legend = [("black", "Euler"), ("#e07a00", "midpoint"), ("#b00020", "RK4"), ("cyan", "dt = 0.045"), ("magenta", "dt = 0.056")]
    x = 55
    for color, label in legend:
        draw.line((x, 490, x + 20, 490), fill=color, width=3)
        draw.text((x + 27, 490), label, fill="black", anchor="lm", font=font(11))
        x += 92 if "dt" not in label else 110

    fields_by_dt = {}
    for panel, dt in zip(panels[1:], [0.045, 0.056]):
        rows = histories[dt]
        times = sorted(set(row[0] for row in rows))
        values = {(row[0], int(row[1])): row[3] for row in rows}
        field_rows = [[values[(time, row)] for row in range(64)] for time in times]
        fields_by_dt[dt] = field_rows
    maximum = max(abs(value) for field in fields_by_dt.values() for row in field for value in row)
    for panel, dt in zip(panels[1:], [0.045, 0.056]):
        rows = raster(fields_by_dt[dt], 64, len(fields_by_dt[dt]), lambda value: diverging(value, maximum)).resize((420, 317), Image.Resampling.BILINEAR)
        image.paste(rows, panel[:2])
        axes(draw, panel, "x", "time", f"RK4, dt = {dt}", (0, 2 * math.pi), (6, 0),
             [(0, "0"), (math.pi, "π"), (2 * math.pi, "2π")], [(6, "0"), (3, "3"), (0, "6")])
    draw.text((1015, 405), f"shared field scale: ±{maximum:.3g}", fill="black", font=font(11))
    image.save(EVIDENCE / "line-stability.png", dpi=(180, 180))


if __name__ == "__main__":
    main()
