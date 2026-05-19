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
import math
import numbers
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import zipfile
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
from cycler import cycler


# =============================================================================
# 1. Figure/plot style definitions
# =============================================================================


@dataclass
class FigureStyle:
    """Global figure style for the paper."""

    # ---- Font sizes ----
    font_family: str = "DejaVu Serif"
    font_size: float = 12.0
    axes_label_size: float = 12.0
    axes_title_size: float = 12.0
    tick_label_size: float = 12.0
    legend_font_size: float = 12.0

    # ---- Line / marker / grid ----
    line_width: float = 1.4
    marker_size: float = 4.0
    grid_alpha: float = 0.18
    grid_line_style: str = "--"
    color_cycle: Tuple[str, ...] = (
        "#1F77B4",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#0072B2",
        "#6B6B6B",
    )

    # ---- DPI / export ----
    dpi_screen: int = 150
    dpi_export: int = 400

    # ---- Size presets (inch) ----
    width_single: float = 8.0 / 2.54
    width_double: float = 16.0 / 2.54
    height_small: float = 7.0 / 2.54
    height_medium: float = 7.0 / 2.54
    height_large: float = 14.0 / 2.54
    height_tall: float = 14.0 / 2.54

    # ---- Subplot spacing ----
    subplot_wspace: float = 0.34
    subplot_hspace: float = 0.42
    subplot_top: float = 0.92
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
            "font.serif": [style.font_family, "Times", "Liberation Serif"],
            "font.size": style.font_size,
            "axes.labelsize": style.axes_label_size,
            "axes.titlesize": style.axes_title_size,
            "xtick.labelsize": style.tick_label_size,
            "ytick.labelsize": style.tick_label_size,
            "legend.fontsize": style.legend_font_size,
            "figure.dpi": style.dpi_screen,
            "savefig.dpi": style.dpi_export,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": style.grid_alpha,
            "grid.linestyle": style.grid_line_style,
            "grid.color": "#B8BEC8",
            "axes.edgecolor": "#2F3437",
            "axes.linewidth": 0.8,
            "axes.prop_cycle": cycler(color=style.color_cycle),
            "lines.linewidth": style.line_width,
            "lines.markersize": style.marker_size,
            "mathtext.fontset": "custom",
            "mathtext.rm": style.font_family,
            "mathtext.it": f"{style.font_family}:italic",
            "mathtext.bf": f"{style.font_family}:bold",
            "mathtext.default": "it",
            "axes.formatter.use_mathtext": True,
            "text.usetex": False,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
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
        - "single"       : 8 cm x 7 cm single-plot figure
        - "wide"         : 16 cm x 14 cm subplot figure
        - "quad"         : 16 cm x 14 cm subplot figure
        - "tall"         : 16 cm x 14 cm subplot figure
        - "custom"       : use default fallback
    """
    style = style or DEFAULT_FIG_STYLE

    has_subplots = nrows > 1 or ncols > 1
    if kind == "single" and not has_subplots:
        figsize = (style.width_single, style.height_medium)
    elif kind in {"wide", "quad", "tall"} or has_subplots:
        figsize = (style.width_double, style.height_tall)
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
# 3. Figure data export helpers
# =============================================================================


def columns_to_rows(columns: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Convert column-oriented figure data into Excel-friendly row dictionaries."""
    max_len = max((len(values) for values in columns.values()), default=0)
    rows: List[Dict[str, Any]] = []
    for i in range(max_len):
        row: Dict[str, Any] = {}
        for name, values in columns.items():
            row[name] = values[i] if i < len(values) else ""
        rows.append(row)
    return rows


def save_excel_workbook(path: str | Path, sheets: Mapping[str, Sequence[Mapping[str, Any]]]) -> Path:
    """Save simple tabular data to an XLSX workbook without extra dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    prepared: List[Tuple[str, List[Mapping[str, Any]]]] = []
    used_names: set[str] = set()
    for raw_name, rows in sheets.items():
        sheet_name = _sanitize_sheet_name(raw_name, used_names)
        prepared.append((sheet_name, list(rows)))

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _xlsx_content_types(len(prepared)))
        zf.writestr("_rels/.rels", _xlsx_root_rels())
        zf.writestr("xl/workbook.xml", _xlsx_workbook_xml([name for name, _ in prepared]))
        zf.writestr("xl/_rels/workbook.xml.rels", _xlsx_workbook_rels(len(prepared)))
        zf.writestr("xl/styles.xml", _xlsx_styles_xml())
        for idx, (_, rows) in enumerate(prepared, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _xlsx_sheet_xml(rows))
    return path


def _sanitize_sheet_name(name: str, used_names: set[str]) -> str:
    base = re.sub(r"[\[\]:*?/\\]", "_", str(name)).strip() or "Sheet"
    base = base[:31]
    candidate = base
    n = 1
    while candidate in used_names:
        suffix = f"_{n}"
        candidate = f"{base[:31 - len(suffix)]}{suffix}"
        n += 1
    used_names.add(candidate)
    return candidate


def _xlsx_col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def _xlsx_cell(ref: str, value: Any) -> str:
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return f'<c r="{ref}"><v>{number:.15g}</v></c>'
        return f'<c r="{ref}"/>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _xlsx_sheet_xml(rows: Sequence[Mapping[str, Any]]) -> str:
    headers: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(str(key))

    xml_rows: List[str] = []
    header_cells = [_xlsx_cell(f"{_xlsx_col_name(i)}1", header) for i, header in enumerate(headers, start=1)]
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')
    for row_idx, row in enumerate(rows, start=2):
        cells = []
        for col_idx, header in enumerate(headers, start=1):
            ref = f"{_xlsx_col_name(col_idx)}{row_idx}"
            cells.append(_xlsx_cell(ref, row.get(header, "")))
        xml_rows.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(xml_rows)}</sheetData>'
        "</worksheet>"
    )


def _xlsx_content_types(n_sheets: int) -> str:
    sheet_overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for i in range(1, n_sheets + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        f"{sheet_overrides}</Types>"
    )


def _xlsx_root_rels() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )


def _xlsx_workbook_xml(sheet_names: Sequence[str]) -> str:
    sheets_xml = "".join(
        f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
        for i, name in enumerate(sheet_names, start=1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{sheets_xml}</sheets></workbook>"
    )


def _xlsx_workbook_rels(n_sheets: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{i}.xml"/>'
        for i in range(1, n_sheets + 1)
    )
    rels += (
        f'<Relationship Id="rId{n_sheets + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{rels}</Relationships>"
    )


def _xlsx_styles_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="12"/><name val="DejaVu Serif"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border/></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        "</styleSheet>"
    )


# =============================================================================
# 4. Common axis formatting helpers
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
