#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_plot_style.py

Unified plotting utilities for the SCR BMN paper.

Purpose
-------
1. Provide a single location to define figure sizes, font sizes and export formats;
2. Keep the visual style of all chapter/section figures consistent;
3. Allow later analysis scripts (4.3, 4.4, 4.5, etc.) to reuse the same style.

Usage
-----
from paper_plot_style import (
    FigureStyle,
    PlotExportConfig,
    apply_paper_style,
    create_subplots,
    save_figure,
)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


# =============================================================================
# 1. Figure/plot style definitions
# =============================================================================


@dataclass
class FigureStyle:
    """Global figure style for the paper."""

    # ---- Font sizes ----
    font_family: str = "DejaVu Sans"
    font_size: float = 10.0
    axes_label_size: float = 10.0
    axes_title_size: float = 10.5
    tick_label_size: float = 9.0
    legend_font_size: float = 8.5

    # ---- Line / marker / grid ----
    line_width: float = 1.6
    marker_size: float = 4.5
    grid_alpha: float = 0.25
    grid_line_style: str = "--"

    # ---- DPI / export ----
    dpi_screen: int = 150
    dpi_export: int = 300

    # ---- Size presets (inch) ----
    width_single: float = 6.6
    width_double: float = 13.6
    height_small: float = 4.2
    height_medium: float = 5.0
    height_large: float = 6.4
    height_tall: float = 8.0

    # ---- Subplot spacing ----
    subplot_wspace: float = 0.28
    subplot_hspace: float = 0.36
    subplot_top: float = 0.90
    subplot_bottom: float = 0.12


@dataclass
class PlotExportConfig:
    """Export settings for all figures."""

    save_png: bool = True
    save_pdf: bool = True
    transparent: bool = False
    bbox_inches: str = "tight"
    pad_inches: float = 0.03


DEFAULT_FIG_STYLE = FigureStyle()
DEFAULT_EXPORT_CONFIG = PlotExportConfig()


# =============================================================================
# 2. Style application and figure creation helpers
# =============================================================================


def apply_paper_style(style: FigureStyle | None = None) -> FigureStyle:
    """Apply a consistent matplotlib style for the whole manuscript."""
    style = style or DEFAULT_FIG_STYLE
    plt.rcParams.update(
        {
            "font.family": style.font_family,
            "font.size": style.font_size,
            "axes.labelsize": style.axes_label_size,
            "axes.titlesize": style.axes_title_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_font_size,
            "figure.dpi": style.dpi_screen,
            "savefig.dpi": style.dpi_export,
            "axes.grid": True,
            "grid.alpha": style.grid_alpha,
            "grid.linestyle": style.grid_line_style,
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
        }
    )
    return style


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    kind: str = "single",
    style: FigureStyle | None = None,
    sharex: bool = False,
    sharey: bool = False,
):
    """Create figure with common size presets.

    Parameters
    ----------
    kind : str
        Preset name:
        - "single"       : 1-column figure
        - "wide"         : 2-column wide figure
        - "quad"         : 2x2 style figure
        - "tall"         : tall figure for multiple rows
        - "custom"       : use default fallback
    """
    style = style or DEFAULT_FIG_STYLE

    if kind == "single":
        figsize = (style.width_single, style.height_medium)
    elif kind == "wide":
        figsize = (style.width_double, style.height_medium)
    elif kind == "quad":
        figsize = (style.width_double, style.height_large)
    elif kind == "tall":
        figsize = (style.width_single, style.height_tall)
    else:
        figsize = (style.width_single, style.height_medium)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
    )
    if nrows > 1 or ncols > 1:
        fig.subplots_adjust(
            wspace=style.subplot_wspace,
            hspace=style.subplot_hspace,
            top=style.subplot_top,
            bottom=style.subplot_bottom,
        )
    return fig, axes


def save_figure(
    fig,
    save_dir: str | Path,
    stem: str,
    *,
    style: FigureStyle | None = None,
    export: PlotExportConfig | None = None,
) -> List[Path]:
    """Save a figure to PNG/PDF using unified export options."""
    style = style or DEFAULT_FIG_STYLE
    export = export or DEFAULT_EXPORT_CONFIG
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    if export.save_png:
        png_path = save_dir / f"{stem}.png"
        fig.savefig(
            png_path,
            bbox_inches=export.bbox_inches,
            pad_inches=export.pad_inches,
            transparent=export.transparent,
            dpi=style.dpi_export,
        )
        paths.append(png_path)
    if export.save_pdf:
        pdf_path = save_dir / f"{stem}.pdf"
        fig.savefig(
            pdf_path,
            bbox_inches=export.bbox_inches,
            pad_inches=export.pad_inches,
            transparent=export.transparent,
        )
        paths.append(pdf_path)
    return paths


# =============================================================================
# 3. Common axis formatting helpers
# =============================================================================


def finalize_axes(
    axes,
    *,
    legend: bool = True,
    grid: bool = True,
):
    """Apply final clean-up to a single axis or a list/array of axes."""
    if not isinstance(axes, (list, tuple)):
        try:
            axes_iter = list(axes.flat)  # numpy array of axes
        except Exception:
            axes_iter = [axes]
    else:
        axes_iter = list(axes)

    for ax in axes_iter:
        if grid:
            ax.grid(True)
        if legend:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(frameon=True)

