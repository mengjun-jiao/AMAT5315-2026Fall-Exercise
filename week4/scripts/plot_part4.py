"""Render compact Part 4 order and self-convergence figures."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week4" / "evidence"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def f(size):
    return ImageFont.truetype(FONT, size)


def point(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    return (int(left + (math.log(x) - math.log(xlim[0])) / (math.log(xlim[1]) - math.log(xlim[0])) * (right - left)),
            int(bottom - (math.log(y) - math.log(ylim[0])) / (math.log(ylim[1]) - math.log(ylim[0])) * (bottom - top)))


def axes(draw, box, title, xlabel, ylabel, x_ticks, y_ticks):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, top - 15), title, fill="black", anchor="mm", font=f(13))
    draw.text(((left + right) / 2, bottom + 18), xlabel, fill="black", anchor="ma", font=f(10))
    draw.multiline_text((left - 25, (top + bottom) / 2), ylabel.replace(" ", "\n"), fill="black", anchor="mm", align="center", spacing=0, font=f(8))
    for value, label in x_ticks:
        px, _ = point(box, value, y_ticks[0][0], (x_ticks[0][0], x_ticks[-1][0]), (y_ticks[0][0], y_ticks[-1][0]))
        draw.text((px, bottom + 4), label, fill="black", anchor="ma", font=f(8))
    for value, label in y_ticks:
        _, py = point(box, x_ticks[0][0], value, (x_ticks[0][0], x_ticks[-1][0]), (y_ticks[0][0], y_ticks[-1][0]))
        draw.text((left - 7, py), label, fill="black", anchor="rm", font=f(8))


def line(draw, box, values, xlim, ylim, color, width=2):
    draw.line([point(box, x, y, xlim, ylim) for x, y in values], fill=color, width=width)


def legend(draw, entries, x, y):
    for index, (color, label) in enumerate(entries):
        py = y + index * 14
        draw.line((x, py, x + 15, py), fill=color, width=2)
        draw.text((x + 19, py), label, fill="black", anchor="lm", font=f(8))


def main():
    order = json.loads((EVIDENCE / "order.json").read_text())
    convergence = json.loads((EVIDENCE / "convergence.json").read_text())
    image = Image.new("RGB", (1420, 505), "white")
    draw = ImageDraw.Draw(image)

    order_values = [(item["dt"], item["relative_velocity_error"]) for item in order["dt_errors"]]
    xs = [math.log(x) for x, _ in order_values]
    ys = [math.log(y) for _, y in order_values]
    slope = sum((x - sum(xs) / len(xs)) * (y - sum(ys) / len(ys)) for x, y in zip(xs, ys)) / sum((x - sum(xs) / len(xs)) ** 2 for x in xs)
    intercept = sum(ys) / len(ys) - slope * sum(xs) / len(xs)
    order_box = (65, 55, 655, 365)
    order_lim = (0.19, 0.42), (1e-5, 2e-3)
    axes(draw, order_box, "Taylor-Green RK4 temporal order", "dt", "relative velocity error", [(0.2, "0.2"), (0.25, "0.25"), (0.4, "0.4")], [(1e-5, "1e-5"), (1e-4, "1e-4"), (1e-3, "1e-3")])
    line(draw, order_box, order_values, *order_lim, "#1769aa", 1)
    for x, y in order_values:
        px, py = point(order_box, x, y, *order_lim)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill="#1769aa")
    fitted = [(order_lim[0][0], math.exp(intercept) * order_lim[0][0] ** slope), (order_lim[0][1], math.exp(intercept) * order_lim[0][1] ** slope)]
    line(draw, order_box, fitted, *order_lim, "#b00020", 2)
    legend(draw, [("#1769aa", "measured"), ("#b00020", f"fit, q = {slope:.2f}")], 435, 90)

    conv_values = [(item["dt"], item["relative_omega_error"]) for item in convergence["measured_relative_omega_errors"]]
    xs = [math.log(x) for x, _ in conv_values]
    ys = [math.log(y) for _, y in conv_values]
    conv_slope = sum((x - sum(xs) / len(xs)) * (y - sum(ys) / len(ys)) for x, y in zip(xs, ys)) / sum((x - sum(xs) / len(xs)) ** 2 for x in xs)
    conv_intercept = sum(ys) / len(ys) - conv_slope * sum(xs) / len(xs)
    conv_box = (765, 55, 1355, 365)
    conv_lim = (0.0095, 0.021), (5e-7, 5e-5)
    axes(draw, conv_box, "Random-flow RK4 self-convergence", "dt", "relative vorticity error", [(0.01, "0.01"), (0.0125, "0.0125"), (0.02, "0.02")], [(1e-6, "1e-6"), (1e-5, "1e-5"), (1e-4, "1e-4")])
    for x, y in conv_values:
        px, py = point(conv_box, x, y, *conv_lim)
        color = "#d01818" if x == convergence["selected_dt"] else "#1769aa"
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)
    fitted = [(conv_lim[0][0], math.exp(conv_intercept) * conv_lim[0][0] ** conv_slope), (conv_lim[0][1], math.exp(conv_intercept) * conv_lim[0][1] ** conv_slope)]
    line(draw, conv_box, fitted, *conv_lim, "#333333", 2)
    threshold_y = point(conv_box, conv_lim[0][0], convergence["threshold"], *conv_lim)[1]
    for x in range(conv_box[0], conv_box[2], 12):
        draw.line((x, threshold_y, min(x + 7, conv_box[2]), threshold_y), fill="#df7f00", width=1)
    legend(draw, [("#333333", f"fit, q = {conv_slope:.3f}"), ("#df7f00", "threshold = 5e-6"), ("#d01818", "selected dt")], 1040, 90)
    image.crop((0, 0, 710, 420)).save(EVIDENCE / "order.png", dpi=(140, 140))
    image.crop((710, 0, 1420, 420)).save(EVIDENCE / "convergence.png", dpi=(140, 140))


if __name__ == "__main__":
    main()
