#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot DirectMapping training history without third-party plotting dependencies.

Outputs an SVG figure that can be opened directly in a browser.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "DirectMapping_baseline" / "DirectMapping_DD_history.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "DirectMapping_baseline" / "DirectMapping_DD_history_plot.svg"


def polyline_points(xs: Iterable[float], ys: Iterable[float]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))


def scale_linear(values: List[float], vmin: float, vmax: float, out_min: float, out_max: float) -> List[float]:
    if math.isclose(vmin, vmax):
        mid = 0.5 * (out_min + out_max)
        return [mid for _ in values]
    return [out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin) for v in values]


def scale_log10(values: List[float], vmin: float, vmax: float, out_min: float, out_max: float) -> List[float]:
    safe = [max(v, 1.0e-30) for v in values]
    lvmin = math.log10(max(vmin, 1.0e-30))
    lvmax = math.log10(max(vmax, 1.0e-30))
    if math.isclose(lvmin, lvmax):
        mid = 0.5 * (out_min + out_max)
        return [mid for _ in safe]
    return [out_min + (math.log10(v) - lvmin) * (out_max - out_min) / (lvmax - lvmin) for v in safe]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DirectMapping training/validation loss history as SVG.")
    parser.add_argument("--history", type=str, default=str(DEFAULT_HISTORY), help="Path to DirectMapping_DD_history.json")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output SVG path")
    parser.add_argument("--logy", action="store_true", help="Use log10 scaling on the y-axis")
    args = parser.parse_args()

    history_path = Path(args.history)
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_loss = [float(v) for v in history.get("train_loss", [])]
    val_loss = [float(v) for v in history.get("val_loss", [])]
    if not train_loss or not val_loss:
        raise ValueError(f"History file does not contain usable train_loss/val_loss arrays: {history_path}")

    n_epochs = min(len(train_loss), len(val_loss))
    train_loss = train_loss[:n_epochs]
    val_loss = val_loss[:n_epochs]
    epochs = list(range(1, n_epochs + 1))
    best_val_epoch = min(range(n_epochs), key=lambda i: val_loss[i]) + 1
    best_val = val_loss[best_val_epoch - 1]

    width, height = 1200, 720
    left, right, top, bottom = 110, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_vals = scale_linear(epochs, 1.0, float(max(epochs)), left, left + plot_w)
    y_source = train_loss + val_loss
    y_min = min(y_source)
    y_max = max(y_source)
    if args.logy:
        y_train = scale_log10(train_loss, y_min, y_max, top + plot_h, top)
        y_val = scale_log10(val_loss, y_min, y_max, top + plot_h, top)
    else:
        y_train = scale_linear(train_loss, y_min, y_max, top + plot_h, top)
        y_val = scale_linear(val_loss, y_min, y_max, top + plot_h, top)

    best_x = scale_linear([float(best_val_epoch)], 1.0, float(max(epochs)), left, left + plot_w)[0]

    bg = "#FFFFFF"
    axis = "#1F2937"
    grid = "#D1D5DB"
    blue = "#2563EB"
    orange = "#D97706"
    green = "#059669"

    x_ticks = [1, max(1, n_epochs // 4), max(1, n_epochs // 2), max(1, 3 * n_epochs // 4), n_epochs]
    x_ticks = sorted(set(x_ticks))
    x_tick_pos = scale_linear([float(v) for v in x_ticks], 1.0, float(max(epochs)), left, left + plot_w)

    if args.logy:
        ly_min = math.log10(max(y_min, 1.0e-30))
        ly_max = math.log10(max(y_max, 1.0e-30))
        y_tick_vals = [10 ** (ly_min + i * (ly_max - ly_min) / 4.0) for i in range(5)]
        y_tick_pos = scale_log10(y_tick_vals, y_min, y_max, top + plot_h, top)
        y_tick_labels = [f"{v:.1e}" for v in y_tick_vals]
    else:
        y_tick_vals = [y_min + i * (y_max - y_min) / 4.0 for i in range(5)]
        y_tick_pos = scale_linear(y_tick_vals, y_min, y_max, top + plot_h, top)
        y_tick_labels = [f"{v:.2e}" for v in y_tick_vals]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{bg}"/>
  <text x="{width/2:.0f}" y="38" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="28" fill="{axis}">DirectMapping training history</text>
  <text x="{width-40}" y="38" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="18" fill="{axis}">Best val loss = {best_val:.3e}</text>
  <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FFFFFF" stroke="{axis}" stroke-width="1.5"/>
"""

    for x_tick, x_pos in zip(x_ticks, x_tick_pos):
        svg += f'  <line x1="{x_pos:.2f}" y1="{top}" x2="{x_pos:.2f}" y2="{top+plot_h}" stroke="{grid}" stroke-width="1"/>\n'
        svg += f'  <text x="{x_pos:.2f}" y="{top+plot_h+32}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{axis}">{x_tick}</text>\n'

    for y_lab, y_pos in zip(y_tick_labels, y_tick_pos):
        svg += f'  <line x1="{left}" y1="{y_pos:.2f}" x2="{left+plot_w}" y2="{y_pos:.2f}" stroke="{grid}" stroke-width="1"/>\n'
        svg += f'  <text x="{left-16}" y="{y_pos+6:.2f}" text-anchor="end" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{axis}">{y_lab}</text>\n'

    svg += f'  <polyline fill="none" stroke="{blue}" stroke-width="3" points="{polyline_points(x_vals, y_train)}"/>\n'
    svg += f'  <polyline fill="none" stroke="{orange}" stroke-width="3" stroke-dasharray="10 8" points="{polyline_points(x_vals, y_val)}"/>\n'
    svg += f'  <line x1="{best_x:.2f}" y1="{top}" x2="{best_x:.2f}" y2="{top+plot_h}" stroke="{green}" stroke-width="2" stroke-dasharray="4 6"/>\n'

    svg += f"""
  <text x="{left + plot_w/2:.0f}" y="{height-24}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="{axis}">Epoch</text>
  <text x="34" y="{top + plot_h/2:.0f}" text-anchor="middle" transform="rotate(-90 34 {top + plot_h/2:.0f})" font-family="Arial, Helvetica, sans-serif" font-size="20" fill="{axis}">Scaled MSE loss{" (log10 scale)" if args.logy else ""}</text>
  <rect x="{left+24}" y="{top+18}" width="22" height="4" fill="{blue}"/>
  <text x="{left+56}" y="{top+24}" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{axis}">Train loss</text>
  <line x1="{left+190}" y1="{top+20}" x2="{left+212}" y2="{top+20}" stroke="{orange}" stroke-width="3" stroke-dasharray="10 8"/>
  <text x="{left+224}" y="{top+24}" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{axis}">Validation loss</text>
  <line x1="{left+410}" y1="{top+20}" x2="{left+432}" y2="{top+20}" stroke="{green}" stroke-width="2" stroke-dasharray="4 6"/>
  <text x="{left+444}" y="{top+24}" font-family="Arial, Helvetica, sans-serif" font-size="16" fill="{axis}">Best validation epoch = {best_val_epoch}</text>
</svg>
"""

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
