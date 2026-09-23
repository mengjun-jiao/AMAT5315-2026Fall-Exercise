"""Plot Part 4 order and convergence results from audit JSON files."""

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "week4" / "evidence"


def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def point(box, x, y, xlim, ylim):
    left, top, right, bottom = box
    return (int(left + (math.log(x) - math.log(xlim[0])) / (math.log(xlim[1]) - math.log(xlim[0])) * (right - left)), int(bottom - (math.log(y) - math.log(ylim[0])) / (math.log(ylim[1]) - math.log(ylim[0])) * (bottom - top)))


def slope_and_intercept(points):
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    x_bar, y_bar = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys)) / sum((x - x_bar) ** 2 for x in xs)
    return slope, y_bar - slope * x_bar


def axes(draw, box, title, xlabel, ylabel, x_ticks, y_ticks):
    left, top, right, bottom = box
    draw.rectangle(box, outline="black")
    draw.text(((left + right) / 2, top - 26), title, fill="black", anchor="mm", font=font(19))
    draw.text(((left + right) / 2, bottom + 28), xlabel, fill="black", anchor="mm", font=font(15))
    draw.text((left + 5, (top + bottom) / 2), ylabel, fill="black", anchor="lm", font=font(14))
    xlim = (min(x_ticks), max(x_ticks))
    ylim = (min(y_ticks), max(y_ticks))
    for value in x_ticks:
        px, _ = point(box, value, ylim[0], xlim, ylim)
        draw.line((px, bottom, px, bottom + 7), fill="black")
        draw.text((px, bottom + 12), f"{value:g}", fill="black", anchor="ma", font=font(12))
    for value in y_ticks:
        _, py = point(box, xlim[0], value, xlim, ylim)
        draw.line((left - 7, py, left, py), fill="black")
        draw.text((left - 12, py), f"{value:.0e}", fill="black", anchor="rm", font=font(11))


def draw_series(draw, box, points, xlim, ylim, colour, width=2):
    draw.line([point(box, x, y, xlim, ylim) for x, y in points], fill=colour, width=width, joint="curve")


def main():
    order = json.loads((EVIDENCE / "order.json").read_text())
    convergence = json.loads((EVIDENCE / "convergence.json").read_text())
    image = Image.new("RGB", (1500, 700), "white")
    draw = ImageDraw.Draw(image)
    order_points = [(item["dt"], item["relative_velocity_error"]) for item in order["dt_errors"]]
    order_slope, order_intercept = slope_and_intercept(order_points)
    order_y = [value for _, value in order_points]
    order_box = (100, 105, 700, 570)
    order_x = (min(x for x, _ in order_points), max(x for x, _ in order_points))
    order_ylim = (min(order_y) * 0.5, max(order_y) * 2)
    axes(draw, order_box, "Taylor-Green RK4 temporal order", "dt", "relative velocity error", [0.2, 0.25, 0.4], [1e-6, 1e-5, 1e-4, 1e-3])
    draw_series(draw, order_box, order_points, order_x, order_ylim, "blue", 0)
    for x, y in order_points:
        px, py = point(order_box, x, y, order_x, order_ylim)
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="blue")
    fitted = [(order_x[0], math.exp(order_intercept) * order_x[0] ** order_slope), (order_x[1], math.exp(order_intercept) * order_x[1] ** order_slope)]
    draw_series(draw, order_box, fitted, order_x, order_ylim, "red", 2)
    draw.text((120, 600), f"Fitted slope = {order_slope:.3f}; storage scale ≈ 1e-6", fill="black", font=font(13))
    draw.line((120, 635, 145, 635), fill="blue", width=3)
    draw.text((155, 635), "measured errors", fill="black", anchor="lm", font=font(12))
    draw.line((280, 635, 305, 635), fill="red", width=3)
    draw.text((315, 635), "fitted log-log line", fill="black", anchor="lm", font=font(12))

    convergence_points = [(item["dt"], item["relative_omega_error"]) for item in convergence["measured_relative_omega_errors"]]
    convergence_slope, convergence_intercept = slope_and_intercept(convergence_points)
    convergence_box = (800, 105, 1400, 570)
    x_ticks = [0.01, 0.0125, 0.02]
    y_values = [value for _, value in convergence_points] + [convergence["threshold"]]
    convergence_ylim = (min(y_values) * 0.5, max(y_values) * 2)
    axes(draw, convergence_box, "Random-flow RK4 self-convergence", "dt", "relative omega error", x_ticks, [1e-7, 1e-6, 1e-5, 1e-4])
    for x, y in convergence_points:
        px, py = point(convergence_box, x, y, (min(x_ticks), max(x_ticks)), convergence_ylim)
        colour = "red" if x == convergence["selected_dt"] else "blue"
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=colour)
    fitted = [(min(x_ticks), math.exp(convergence_intercept) * min(x_ticks) ** convergence_slope), (max(x_ticks), math.exp(convergence_intercept) * max(x_ticks) ** convergence_slope)]
    draw_series(draw, convergence_box, fitted, (min(x_ticks), max(x_ticks)), convergence_ylim, "black", 2)
    threshold_y = point(convergence_box, min(x_ticks), convergence["threshold"], (min(x_ticks), max(x_ticks)), convergence_ylim)[1]
    draw.line((convergence_box[0], threshold_y, convergence_box[2], threshold_y), fill="orange", width=2)
    draw.text((820, 600), f"Fitted slope = {convergence_slope:.3f}; orange threshold = 5e-6", fill="black", font=font(13))
    draw.text((820, 635), f"Selected dt = {convergence['selected_dt']:g} (red point)", fill="black", font=font(13))
    draw.text((55, 28), "Part 4 temporal order and convergence validation", fill="black", font=font(25))
    image.crop((0, 0, 750, 700)).save(EVIDENCE / "order.png")
    image.crop((750, 0, 1500, 700)).save(EVIDENCE / "convergence.png")


if __name__ == "__main__":
    main()
