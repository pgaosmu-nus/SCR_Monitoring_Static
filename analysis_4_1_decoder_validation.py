#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_4_1_decoder_validation.py

Section 4.1: Forward surrogate (Decoder) validation.

Main tasks
----------
1. Generate an in-domain exact-solver test set;
2. Evaluate the trained Decoder on the random test set;
3. Evaluate several representative typical cases for plotting;
4. Save metrics, raw data and figures in a paper-friendly folder structure.

Expected companion files
------------------------
- Decoder_DD.py
- paper_plot_style.py
- para_config.json
- physics_config.json
- trained Decoder checkpoint (.pth)
- exact solver dependency used by Decoder_DD.py

Notes
-----
- This script is designed for *interpolation-domain* validation only;
- The user settings section near the top is the main place to edit parameters;
- Figure export style is unified through paper_plot_style.py.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Decoder_DD import (
    BMNConfig,
    PARAM_NAMES,
    compute_exact_case,
    decode_fullfield_np,
    load_config,
    load_decoder_model,
    sample_one_case,
)
from paper_plot_style import (
    DEFAULT_EXPORT_CONFIG,
    DEFAULT_FIG_STYLE,
    apply_paper_style,
    create_subplots,
    finalize_axes,
    save_figure,
)


# =============================================================================
# 0. USER SETTINGS (main place to edit)
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parent

# ---- Paths ----
CONFIG_PATH = REPO_ROOT / "para_config.json"
DECODER_CKPT_PATH = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "Decoder_DD_model.pth"
OUTPUT_ROOT = REPO_ROOT / "paper_outputs" / "4_1_decoder_validation"

# ---- Test-set settings ----
RANDOM_TEST_N_CASES = 500
RANDOM_TEST_SEED = 20260502
SAVE_RANDOM_CASE_RAW_DATA = True

# ---- Device for Decoder inference ----
DEVICE = "cpu"  # use "cuda" if available in your environment

# ---- Representative typical cases (all inside training domain; interpolation only) ----
# Order of parameters: [Dx, ht, Us, Ub, p]
TYPICAL_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "F1_low_flow",
        "description": "Low-current interpolation case",
        "Dx": 1800.0,
        "ht": 0.0,
        "Us": 0.80,
        "Ub": 0.15,
        "p": 0.170,
    },
    {
        "case_id": "F2_baseline",
        "description": "Baseline interpolation case",
        "Dx": 1800.0,
        "ht": 0.0,
        "Us": 1.50,
        "Ub": 0.40,
        "p": 0.240,
    },
    {
        "case_id": "F3_high_flow",
        "description": "High-current interpolation case",
        "Dx": 1800.0,
        "ht": 0.0,
        "Us": 2.20,
        "Ub": 0.70,
        "p": 0.300,
    },
    {
        "case_id": "F4_short_span",
        "description": "Small horizontal span interpolation case",
        "Dx": 1720.0,
        "ht": 0.0,
        "Us": 1.50,
        "Ub": 0.40,
        "p": 0.240,
    },
    {
        "case_id": "F5_long_span",
        "description": "Large horizontal span interpolation case",
        "Dx": 1880.0,
        "ht": 0.0,
        "Us": 1.50,
        "Ub": 0.40,
        "p": 0.240,
    },
    {
        "case_id": "F6_top_offset",
        "description": "Top-height-offset interpolation case",
        "Dx": 1800.0,
        "ht": 8.0,
        "Us": 1.50,
        "Ub": 0.40,
        "p": 0.240,
    },
]

# ---- Plot export configuration ----
FIG_STYLE = DEFAULT_FIG_STYLE
EXPORT_STYLE = DEFAULT_EXPORT_CONFIG


# =============================================================================
# 1. Utilities
# =============================================================================


OUTPUT_VARS = ["x", "z", "theta", "T", "M"]
EXACT_SOLVER_COLOR = "tab:blue"
DECODER_COLOR = "y"
DECODER_LINESTYLE = "--"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def case_dict_to_array(case: Dict[str, Any]) -> np.ndarray:
    return np.asarray([case[name] for name in PARAM_NAMES], dtype=np.float32)


def compute_scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(names):
        y_true_j = y_true[:, :, j]
        y_pred_j = y_pred[:, :, j]
        diff = y_pred_j - y_true_j
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        maxae = float(np.max(np.abs(diff)))
        value_range = float(np.max(y_true_j) - np.min(y_true_j))
        nrmse = float(rmse / value_range) if value_range > 1.0e-12 else float("nan")
        metrics[name] = {"rmse": rmse, "mae": mae, "maxae": maxae, "nrmse": nrmse}
    return metrics


def compute_feature_metrics(s: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, float]:
    idx_T = names.index("T")
    idx_M = names.index("M")

    T_top_true = y_true[:, -1, idx_T]
    T_top_pred = y_pred[:, -1, idx_T]
    M_abs_true = np.abs(y_true[:, :, idx_M])
    M_abs_pred = np.abs(y_pred[:, :, idx_M])
    M_max_true = np.max(M_abs_true, axis=1)
    M_max_pred = np.max(M_abs_pred, axis=1)
    s_Mmax_true = s[np.argmax(M_abs_true, axis=1)]
    s_Mmax_pred = s[np.argmax(M_abs_pred, axis=1)]

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    return {
        "T_top_rmse": rmse(T_top_true, T_top_pred),
        "T_top_mae": mae(T_top_true, T_top_pred),
        "M_max_rmse": rmse(M_max_true, M_max_pred),
        "M_max_mae": mae(M_max_true, M_max_pred),
        "s_Mmax_rmse": rmse(s_Mmax_true, s_Mmax_pred),
        "s_Mmax_mae": mae(s_Mmax_true, s_Mmax_pred),
    }


def build_random_exact_test_set(cfg: BMNConfig, n_cases: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases: List[Dict[str, Any]] = []
    while len(cases) < n_cases:
        sampled = sample_one_case(rng, cfg.ranges, cfg.physical)
        exact = compute_exact_case(sampled, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            continue
        cases.append({"params": sampled, "exact": exact})
        if len(cases) % max(1, n_cases // 10) == 0:
            print(f"  exact random test cases collected: {len(cases)}/{n_cases}")
    return cases


def evaluate_decoder_on_cases(
    decoder_model,
    decoder_ckpt: Dict[str, Any],
    cases: List[Dict[str, Any]],
    device: str,
) -> Dict[str, Any]:
    s = np.asarray(cases[0]["exact"]["s"], dtype=np.float32)
    n_cases = len(cases)
    n_nodes = len(s)
    y_true = np.zeros((n_cases, n_nodes, len(OUTPUT_VARS)), dtype=np.float32)
    y_pred = np.zeros_like(y_true)
    params = np.zeros((n_cases, len(PARAM_NAMES)), dtype=np.float32)

    for i, item in enumerate(cases):
        par = item["params"]
        params[i, :] = case_dict_to_array(par)
        c = np.asarray([par["Dx"], par["ht"]], dtype=np.float32)
        mu = np.asarray([par["Us"], par["Ub"], par["p"]], dtype=np.float32)
        pred = decode_fullfield_np(decoder_model, decoder_ckpt, s, c, mu, device=device)
        for j, name in enumerate(OUTPUT_VARS):
            y_true[i, :, j] = np.asarray(item["exact"][name], dtype=np.float32)
            y_pred[i, :, j] = pred[:, j]

    scalar_metrics = compute_scalar_metrics(y_true, y_pred, OUTPUT_VARS)
    feature_metrics = compute_feature_metrics(s, y_true, y_pred, OUTPUT_VARS)
    mae_profile = np.mean(np.abs(y_pred - y_true), axis=0)  # [n_nodes, n_vars]

    return {
        "s": s,
        "params": params,
        "y_true": y_true,
        "y_pred": y_pred,
        "scalar_metrics": scalar_metrics,
        "feature_metrics": feature_metrics,
        "mae_profile": mae_profile,
    }


def load_decoder_history_if_available(cfg: BMNConfig) -> Dict[str, Any] | None:
    history_path = Path(cfg.dataset.output_dir) / cfg.decoder_training.history_filename
    if not history_path.exists():
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 2. Plotting functions
# =============================================================================


def plot_training_history(history: Dict[str, Any], fig_dir: Path) -> None:
    fig, ax = create_subplots(kind="single", style=FIG_STYLE)
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Decoder training history")
    ax.set_yscale("log")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_1_decoder_training_history", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


def plot_typical_case(case_id: str, desc: str, s: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, fig_dir: Path) -> None:
    idx = {name: i for i, name in enumerate(OUTPUT_VARS)}
    fig, axes = create_subplots(2, 2, kind="quad", style=FIG_STYLE)
    ax1, ax2, ax3, ax4 = axes.flat

    ax1.plot(y_true[:, idx["x"]], y_true[:, idx["z"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax1.plot(y_pred[:, idx["x"]], y_pred[:, idx["z"]], label="Decoder", color=DECODER_COLOR, linestyle=DECODER_LINESTYLE)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_title("(a) SCR configuration")

    ax2.plot(s, y_true[:, idx["theta"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax2.plot(s, y_pred[:, idx["theta"]], label="Decoder", color=DECODER_COLOR, linestyle=DECODER_LINESTYLE)
    ax2.set_xlabel("s (m)")
    ax2.set_ylabel(r"$\theta$ (rad)")
    ax2.set_title("(b) Tangent angle")

    ax3.plot(s, y_true[:, idx["T"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax3.plot(s, y_pred[:, idx["T"]], label="Decoder", color=DECODER_COLOR, linestyle=DECODER_LINESTYLE)
    ax3.set_xlabel("s (m)")
    ax3.set_ylabel("T (N)")
    ax3.set_title("(c) Effective tension")

    ax4.plot(s, y_true[:, idx["M"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax4.plot(s, y_pred[:, idx["M"]], label="Decoder", color=DECODER_COLOR, linestyle=DECODER_LINESTYLE)
    ax4.set_xlabel("s (m)")
    ax4.set_ylabel("M (N·m)")
    ax4.set_title("(d) Bending moment")

    fig.suptitle(f"{case_id}: {desc}")
    finalize_axes(axes)
    save_figure(fig, fig_dir, f"fig_4_1_{case_id}", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


def plot_error_profile(s: np.ndarray, mae_profile: np.ndarray, fig_dir: Path) -> None:
    fig, ax = create_subplots(kind="single", style=FIG_STYLE)
    for j, name in enumerate(OUTPUT_VARS):
        ax.plot(s, mae_profile[:, j], label=name)
    ax.set_xlabel("s (m)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Decoder mean absolute error profile along arc length")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_1_decoder_error_profile", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


def plot_metric_bar(metrics: Dict[str, Dict[str, float]], fig_dir: Path) -> None:
    vars_ = list(metrics.keys())
    rmse = [metrics[v]["rmse"] for v in vars_]
    nrmse = [metrics[v]["nrmse"] for v in vars_]

    fig, axes = create_subplots(1, 2, kind="wide", style=FIG_STYLE)
    axes[0].bar(vars_, rmse)
    axes[0].set_title("(a) RMSE by response variable")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xlabel("Response variable")

    axes[1].bar(vars_, nrmse)
    axes[1].set_title("(b) NRMSE by response variable")
    axes[1].set_ylabel("NRMSE")
    axes[1].set_xlabel("Response variable")

    finalize_axes(axes, legend=False)
    save_figure(fig, fig_dir, "fig_4_1_decoder_metrics_bar", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


# =============================================================================
# 3. Main workflow
# =============================================================================


def main() -> None:
    t0 = time.time()
    apply_paper_style(FIG_STYLE)

    config_path = require_existing_file(CONFIG_PATH, "Config file")
    decoder_ckpt_path = require_existing_file(DECODER_CKPT_PATH, "Decoder checkpoint")

    # ---- Folder structure ----
    out_root = ensure_dir(OUTPUT_ROOT)
    data_dir = ensure_dir(out_root / "data")
    fig_dir = ensure_dir(out_root / "figures")
    table_dir = ensure_dir(out_root / "tables")

    # ---- Load config and decoder ----
    cfg = load_config(config_path)
    decoder_model, decoder_ckpt = load_decoder_model(decoder_ckpt_path, map_location=DEVICE)

    # ---- Save run metadata ----
    run_info = {
        "config_path": str(config_path),
        "decoder_ckpt_path": str(decoder_ckpt_path),
        "random_test_n_cases": RANDOM_TEST_N_CASES,
        "random_test_seed": RANDOM_TEST_SEED,
        "device": DEVICE,
        "typical_cases": TYPICAL_CASES,
    }
    save_json(run_info, data_dir / "run_info_4_1.json")

    # ---- 4.1.1 Random in-domain exact test set ----
    print("=" * 88)
    print("[4.1] Building random in-domain exact-solver test set...")
    random_cases = build_random_exact_test_set(cfg, RANDOM_TEST_N_CASES, RANDOM_TEST_SEED)

    # ---- Evaluate decoder on random test set ----
    print("=" * 88)
    print("[4.1] Evaluating Decoder on random test set...")
    random_result = evaluate_decoder_on_cases(decoder_model, decoder_ckpt, random_cases, DEVICE)

    # ---- Save random test raw data ----
    if SAVE_RANDOM_CASE_RAW_DATA:
        np.savez_compressed(
            data_dir / "decoder_random_test_predictions.npz",
            s=random_result["s"],
            params=random_result["params"],
            y_true=random_result["y_true"],
            y_pred=random_result["y_pred"],
            mae_profile=random_result["mae_profile"],
            output_vars=np.asarray(OUTPUT_VARS),
        )

    # ---- Save metrics (JSON + CSV) ----
    metrics_json = {
        "scalar_metrics": random_result["scalar_metrics"],
        "feature_metrics": random_result["feature_metrics"],
    }
    save_json(metrics_json, data_dir / "decoder_random_test_metrics.json")

    scalar_rows: List[Dict[str, Any]] = []
    for name in OUTPUT_VARS:
        row = {"variable": name}
        row.update(random_result["scalar_metrics"][name])
        scalar_rows.append(row)
    save_csv(scalar_rows, table_dir / "table_4_1_decoder_scalar_metrics.csv")

    feature_rows = [{"metric": k, "value": v} for k, v in random_result["feature_metrics"].items()]
    save_csv(feature_rows, table_dir / "table_4_1_decoder_feature_metrics.csv")

    # ---- Plot training history if available ----
    history = load_decoder_history_if_available(cfg)
    if history is not None:
        print("[4.1] Plotting Decoder training history...")
        plot_training_history(history, fig_dir)

    # ---- Plot random-test aggregate figures ----
    print("[4.1] Plotting aggregate figures...")
    plot_error_profile(random_result["s"], random_result["mae_profile"], fig_dir)
    plot_metric_bar(random_result["scalar_metrics"], fig_dir)

    # ---- 4.1.2 Typical cases ----
    print("=" * 88)
    print("[4.1] Evaluating typical cases...")
    typical_case_rows: List[Dict[str, Any]] = []
    typical_npz_data: Dict[str, Any] = {}

    for case in TYPICAL_CASES:
        exact = compute_exact_case(case, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            print(f"  WARNING: exact solver failed for typical case {case['case_id']}")
            continue
        wrapper = [{"params": case, "exact": exact}]
        result = evaluate_decoder_on_cases(decoder_model, decoder_ckpt, wrapper, DEVICE)
        s = result["s"]
        y_true = result["y_true"][0]
        y_pred = result["y_pred"][0]

        # Save figure
        plot_typical_case(case["case_id"], case["description"], s, y_true, y_pred, fig_dir)

        # Save data into a single npz bundle later
        typical_npz_data[f"{case['case_id']}_s"] = s
        typical_npz_data[f"{case['case_id']}_y_true"] = y_true
        typical_npz_data[f"{case['case_id']}_y_pred"] = y_pred
        typical_npz_data[f"{case['case_id']}_params"] = case_dict_to_array(case)

        idx_T = OUTPUT_VARS.index("T")
        idx_M = OUTPUT_VARS.index("M")
        row = {
            "case_id": case["case_id"],
            "description": case["description"],
            "Dx": case["Dx"],
            "ht": case["ht"],
            "Us": case["Us"],
            "Ub": case["Ub"],
            "p": case["p"],
            "T_top_true": float(y_true[-1, idx_T]),
            "T_top_pred": float(y_pred[-1, idx_T]),
            "T_top_abs_error": float(abs(y_pred[-1, idx_T] - y_true[-1, idx_T])),
            "M_max_true": float(np.max(np.abs(y_true[:, idx_M]))),
            "M_max_pred": float(np.max(np.abs(y_pred[:, idx_M]))),
            "M_max_abs_error": float(abs(np.max(np.abs(y_pred[:, idx_M])) - np.max(np.abs(y_true[:, idx_M])))),
        }
        typical_case_rows.append(row)

    if typical_npz_data:
        np.savez_compressed(data_dir / "decoder_typical_cases.npz", **typical_npz_data)
    save_csv(typical_case_rows, table_dir / "table_4_1_decoder_typical_cases.csv")

    summary = {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "output_root": str(out_root),
    }
    save_json(summary, out_root / "analysis_4_1_summary.json")

    print("=" * 88)
    print("Section 4.1 analysis finished.")
    print(f"Results saved to: {out_root}")
    print("=" * 88)


if __name__ == "__main__":
    main()
