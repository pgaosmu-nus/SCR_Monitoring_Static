#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_4_2_bmn_inversion_validation.py

Section 4.2: BMN inversion validation.

Main tasks
----------
1. Generate an in-domain exact-solver test set;
2. Extract sparse observations according to the trained BMN sensor setting;
3. Reconstruct global parameters and full-field responses using the trained BMN;
4. Save metrics, raw data and figures for Section 4.2.

Expected companion files
------------------------
- Decoder_DD.py
- BMN_DD.py
- paper_plot_style.py
- para_config.json
- trained Decoder and Encoder checkpoints (.pth)
- exact solver dependency used by Decoder_DD.py

Notes
-----
- This script evaluates *interpolation-domain* inversion only;
- Sensor configuration is loaded from the BMN encoder checkpoint by default;
- A dedicated user-settings block is placed near the top for easy modification.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from BMN_DD import load_encoder_model
from Decoder_DD import (
    PARAM_NAMES,
    StandardScaler,
    compute_exact_case,
    decode_fullfield_np,
    extract_observations_from_fields,
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
ENCODER_CKPT_PATH = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "BMN_DD_encoder.pth"
OUTPUT_ROOT = REPO_ROOT / "paper_outputs" / "4_2_bmn_inversion_validation"

# ---- Test-set settings ----
RANDOM_TEST_N_CASES = 500
RANDOM_TEST_SEED = 20260503
SAVE_RANDOM_CASE_RAW_DATA = True

# ---- Device for inference ----
DEVICE = "cpu"  # use "cuda" if available in your environment

# ---- Representative typical cases (all inside training domain; interpolation only) ----
# Order of parameters: [Dx, ht, Us, Ub, p]
TYPICAL_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "I1_low_flow",
        "description": "Low-current inversion case",
        "Dx": 1800.0,
        "ht": 0.0,
        "Us": 0.80,
        "Ub": 0.15,
        "p": 0.170,
    },
    {
        "case_id": "I2_baseline",
        "description": "Baseline inversion case",
        "Dx": 1800.0,
        "ht": 0.0,
        "Us": 1.50,
        "Ub": 0.40,
        "p": 0.240,
    },
    {
        "case_id": "I3_high_flow",
        "description": "High-current inversion case",
        "Dx": 1800.0,
        "ht": 0.0,
        "Us": 2.20,
        "Ub": 0.70,
        "p": 0.300,
    },
    {
        "case_id": "I4_long_span",
        "description": "Large horizontal span inversion case",
        "Dx": 1880.0,
        "ht": 0.0,
        "Us": 1.50,
        "Ub": 0.40,
        "p": 0.240,
    },
    {
        "case_id": "I5_top_offset",
        "description": "Top-height-offset inversion case",
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


EXACT_SOLVER_COLOR = "tab:blue"
MODEL_PRED_COLOR = "y"
MODEL_PRED_LINESTYLE = "--"


def case_dict_to_array(case: Dict[str, Any]) -> np.ndarray:
    return np.asarray([case[name] for name in PARAM_NAMES], dtype=np.float32)


def build_random_exact_test_set(cfg, n_cases: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases: List[Dict[str, Any]] = []
    while len(cases) < n_cases:
        sampled = sample_one_case(rng, cfg.ranges, cfg.physical)
        exact = compute_exact_case(sampled, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            continue
        cases.append({"params": sampled, "exact": exact})
        if len(cases) % max(1, n_cases // 10) == 0:
            print(f"  exact inversion test cases collected: {len(cases)}/{n_cases}")
    return cases


def prepare_inference_tools(encoder_ckpt_path: str | Path, decoder_ckpt_path: str | Path) -> Dict[str, Any]:
    encoder_model, encoder_ckpt = load_encoder_model(encoder_ckpt_path, map_location=DEVICE)
    decoder_model, decoder_ckpt = load_decoder_model(decoder_ckpt_path, map_location=DEVICE)
    encoder_model = encoder_model.to(DEVICE)
    encoder_model.eval()
    obs_scaler = StandardScaler.from_dict(encoder_ckpt["obs_scaler"])
    mu_scaler = StandardScaler.from_dict(encoder_ckpt["mu_scaler"])
    sensor_indices = np.asarray(encoder_ckpt["sensor_indices"], dtype=np.int64)
    observation_vars = [str(v) for v in encoder_ckpt["observation_vars"]]
    output_vars = [str(v) for v in encoder_ckpt["output_vars"]]
    obs_names = [str(v) for v in encoder_ckpt["obs_names"]]
    return {
        "encoder_model": encoder_model,
        "encoder_ckpt": encoder_ckpt,
        "decoder_model": decoder_model,
        "decoder_ckpt": decoder_ckpt,
        "obs_scaler": obs_scaler,
        "mu_scaler": mu_scaler,
        "sensor_indices": sensor_indices,
        "observation_vars": observation_vars,
        "output_vars": output_vars,
        "obs_names": obs_names,
    }


def predict_case_from_exact(
    tools: Dict[str, Any],
    s: np.ndarray,
    params: Dict[str, float],
    y_exact: np.ndarray,
) -> Dict[str, Any]:
    """Generate sparse observation from exact response, then reconstruct with BMN."""
    output_vars = tools["output_vars"]
    observation_vars = tools["observation_vars"]
    sensor_indices = tools["sensor_indices"]

    # Exact sparse observation
    obs_true = extract_observations_from_fields(
        y_exact[None, :, :],
        s,
        output_vars,
        observation_vars,
        sensor_indices,
    )[0]

    # Encoder prediction for mu
    obs_s = tools["obs_scaler"].transform(obs_true[None, :])
    with torch.no_grad():
        mu_s = tools["encoder_model"](torch.tensor(obs_s, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()
    mu_pred = tools["mu_scaler"].inverse_transform(mu_s)[0]

    # Decoder reconstruction
    c = np.asarray([params["Dx"], params["ht"]], dtype=np.float32)
    y_pred = decode_fullfield_np(tools["decoder_model"], tools["decoder_ckpt"], s, c, mu_pred, device=DEVICE)

    # Reconstructed sparse observation
    obs_pred = extract_observations_from_fields(
        y_pred[None, :, :],
        s,
        output_vars,
        observation_vars,
        sensor_indices,
    )[0]

    mu_true = np.asarray([params["Us"], params["Ub"], params["p"]], dtype=np.float32)

    return {
        "obs_true": obs_true.astype(np.float32),
        "obs_pred": obs_pred.astype(np.float32),
        "mu_true": mu_true.astype(np.float32),
        "mu_pred": mu_pred.astype(np.float32),
        "y_true": y_exact.astype(np.float32),
        "y_pred": y_pred.astype(np.float32),
    }


def evaluate_bmn_on_cases(tools: Dict[str, Any], cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = np.asarray(cases[0]["exact"]["s"], dtype=np.float32)
    output_vars = tools["output_vars"]

    n_cases = len(cases)
    y_true = np.zeros((n_cases, len(s), len(output_vars)), dtype=np.float32)
    y_pred = np.zeros_like(y_true)
    obs_true_list = []
    obs_pred_list = []
    mu_true = np.zeros((n_cases, 3), dtype=np.float32)
    mu_pred = np.zeros((n_cases, 3), dtype=np.float32)
    params_all = np.zeros((n_cases, len(PARAM_NAMES)), dtype=np.float32)

    for i, item in enumerate(cases):
        params = item["params"]
        params_all[i, :] = case_dict_to_array(params)
        y_exact = np.stack([np.asarray(item["exact"][name], dtype=np.float32) for name in output_vars], axis=-1)
        pred = predict_case_from_exact(tools, s, params, y_exact)
        y_true[i] = pred["y_true"]
        y_pred[i] = pred["y_pred"]
        mu_true[i] = pred["mu_true"]
        mu_pred[i] = pred["mu_pred"]
        obs_true_list.append(pred["obs_true"])
        obs_pred_list.append(pred["obs_pred"])

    obs_true = np.asarray(obs_true_list, dtype=np.float32)
    obs_pred = np.asarray(obs_pred_list, dtype=np.float32)

    # Parameter metrics
    mu_names = ["Us", "Ub", "p"]
    param_metrics: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(mu_names):
        diff = mu_pred[:, i] - mu_true[:, i]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(mu_true[:, i]), 1.0e-8)))
        param_metrics[name] = {"rmse": rmse, "mae": mae, "mape": mape}

    # Response metrics
    response_metrics: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(output_vars):
        diff = y_pred[:, :, j] - y_true[:, :, j]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        maxae = float(np.max(np.abs(diff)))
        value_range = float(np.max(y_true[:, :, j]) - np.min(y_true[:, :, j]))
        nrmse = float(rmse / value_range) if value_range > 1.0e-12 else float("nan")
        response_metrics[name] = {"rmse": rmse, "mae": mae, "maxae": maxae, "nrmse": nrmse}

    # Observation metrics
    obs_diff = obs_pred - obs_true
    observation_metrics = {
        "rmse": float(np.sqrt(np.mean(obs_diff**2))),
        "mae": float(np.mean(np.abs(obs_diff))),
        "maxae": float(np.max(np.abs(obs_diff))),
    }

    # Feature metrics
    idx_T = output_vars.index("T")
    idx_M = output_vars.index("M")
    T_top_true = y_true[:, -1, idx_T]
    T_top_pred = y_pred[:, -1, idx_T]
    M_abs_true = np.abs(y_true[:, :, idx_M])
    M_abs_pred = np.abs(y_pred[:, :, idx_M])
    M_max_true = np.max(M_abs_true, axis=1)
    M_max_pred = np.max(M_abs_pred, axis=1)
    s_Mmax_true = s[np.argmax(M_abs_true, axis=1)]
    s_Mmax_pred = s[np.argmax(M_abs_pred, axis=1)]
    feature_metrics = {
        "T_top_rmse": float(np.sqrt(np.mean((T_top_pred - T_top_true) ** 2))),
        "T_top_mae": float(np.mean(np.abs(T_top_pred - T_top_true))),
        "M_max_rmse": float(np.sqrt(np.mean((M_max_pred - M_max_true) ** 2))),
        "M_max_mae": float(np.mean(np.abs(M_max_pred - M_max_true))),
        "s_Mmax_rmse": float(np.sqrt(np.mean((s_Mmax_pred - s_Mmax_true) ** 2))),
        "s_Mmax_mae": float(np.mean(np.abs(s_Mmax_pred - s_Mmax_true))),
    }

    mae_profile = np.mean(np.abs(y_pred - y_true), axis=0)

    return {
        "s": s,
        "params_all": params_all,
        "mu_true": mu_true,
        "mu_pred": mu_pred,
        "obs_true": obs_true,
        "obs_pred": obs_pred,
        "y_true": y_true,
        "y_pred": y_pred,
        "param_metrics": param_metrics,
        "response_metrics": response_metrics,
        "observation_metrics": observation_metrics,
        "feature_metrics": feature_metrics,
        "mae_profile": mae_profile,
    }


def load_encoder_history_if_available(cfg) -> Dict[str, Any] | None:
    history_path = Path(cfg.dataset.output_dir) / cfg.encoder_training.history_filename
    if not history_path.exists():
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# 2. Plotting functions
# =============================================================================


def plot_training_history(history: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = create_subplots(1, 2, kind="wide", style=FIG_STYLE)
    epochs = np.arange(1, len(history.get("train_total", [])) + 1)

    axes[0].plot(epochs, history.get("train_total", []), label="Train total")
    axes[0].plot(epochs, history.get("val_total", []), label="Validation total")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("(a) Total loss history")
    axes[0].set_yscale("log")

    axes[1].plot(epochs, history.get("train_mu", []), label=r"Train $L_\mu$")
    axes[1].plot(epochs, history.get("val_mu", []), label=r"Validation $L_\mu$")
    if np.any(np.asarray(history.get("train_observation", [])) > 0.0):
        axes[1].plot(epochs, history.get("train_observation", []), label=r"Train $L_{obs}$")
        axes[1].plot(epochs, history.get("val_observation", []), label=r"Validation $L_{obs}$")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("(b) Loss components")
    axes[1].set_yscale("log")

    finalize_axes(axes)
    save_figure(fig, fig_dir, "fig_4_2_encoder_training_history", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


def plot_param_scatter(mu_true: np.ndarray, mu_pred: np.ndarray, fig_dir: Path) -> None:
    names = ["Us", "Ub", "p"]
    fig, axes = create_subplots(1, 3, kind="wide", style=FIG_STYLE)
    for i, ax in enumerate(axes):
        t = mu_true[:, i]
        p = mu_pred[:, i]
        ax.scatter(t, p, s=12, alpha=0.7, label="Test cases")
        lo = min(np.min(t), np.min(p))
        hi = max(np.max(t), np.max(p))
        ax.plot([lo, hi], [lo, hi], linestyle="--", label="y=x")
        ax.set_xlabel(f"True {names[i]}")
        ax.set_ylabel(f"Predicted {names[i]}")
        ax.set_title(f"({chr(97+i)}) {names[i]}")
    finalize_axes(axes)
    save_figure(fig, fig_dir, "fig_4_2_parameter_scatter", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


def plot_error_profile(s: np.ndarray, mae_profile: np.ndarray, output_vars: List[str], fig_dir: Path) -> None:
    fig, ax = create_subplots(kind="single", style=FIG_STYLE)
    for j, name in enumerate(output_vars):
        ax.plot(s, mae_profile[:, j], label=name)
    ax.set_xlabel("s (m)")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("BMN mean absolute response error along arc length")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_2_bmn_error_profile", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


def plot_typical_case(
    case_id: str,
    desc: str,
    s: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_indices: np.ndarray,
    output_vars: List[str],
    fig_dir: Path,
) -> None:
    idx = {name: i for i, name in enumerate(output_vars)}
    fig, axes = create_subplots(2, 2, kind="quad", style=FIG_STYLE)
    ax1, ax2, ax3, ax4 = axes.flat

    # Configuration
    ax1.plot(y_true[:, idx["x"]], y_true[:, idx["z"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax1.plot(y_pred[:, idx["x"]], y_pred[:, idx["z"]], label="BMN reconstruction", color=MODEL_PRED_COLOR, linestyle=MODEL_PRED_LINESTYLE)
    ax1.scatter(y_true[-1, idx["x"]], y_true[-1, idx["z"]], marker="o", label="Top point")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_title("(a) SCR configuration")

    # Theta with sparse observations
    ax2.plot(s, y_true[:, idx["theta"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax2.plot(s, y_pred[:, idx["theta"]], label="BMN reconstruction", color=MODEL_PRED_COLOR, linestyle=MODEL_PRED_LINESTYLE)
    ax2.scatter(s[sensor_indices], y_true[sensor_indices, idx["theta"]], marker="o", label="Sparse obs")
    ax2.set_xlabel("s (m)")
    ax2.set_ylabel(r"$\theta$ (rad)")
    ax2.set_title("(b) Tangent angle")

    # T with sparse observations
    ax3.plot(s, y_true[:, idx["T"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax3.plot(s, y_pred[:, idx["T"]], label="BMN reconstruction", color=MODEL_PRED_COLOR, linestyle=MODEL_PRED_LINESTYLE)
    ax3.scatter(s[sensor_indices], y_true[sensor_indices, idx["T"]], marker="o", label="Sparse obs")
    ax3.set_xlabel("s (m)")
    ax3.set_ylabel("T (N)")
    ax3.set_title("(c) Effective tension")

    # M with sparse observations
    ax4.plot(s, y_true[:, idx["M"]], label="Exact solver", color=EXACT_SOLVER_COLOR)
    ax4.plot(s, y_pred[:, idx["M"]], label="BMN reconstruction", color=MODEL_PRED_COLOR, linestyle=MODEL_PRED_LINESTYLE)
    ax4.scatter(s[sensor_indices], y_true[sensor_indices, idx["M"]], marker="o", label="Sparse obs")
    ax4.set_xlabel("s (m)")
    ax4.set_ylabel("M (N·m)")
    ax4.set_title("(d) Bending moment")

    fig.suptitle(f"{case_id}: {desc}")
    finalize_axes(axes)
    save_figure(fig, fig_dir, f"fig_4_2_{case_id}", style=FIG_STYLE, export=EXPORT_STYLE)
    plt.close(fig)


# =============================================================================
# 3. Main workflow
# =============================================================================


def main() -> None:
    t0 = time.time()
    apply_paper_style(FIG_STYLE)

    config_path = require_existing_file(CONFIG_PATH, "Config file")
    decoder_ckpt_path = require_existing_file(DECODER_CKPT_PATH, "Decoder checkpoint")
    encoder_ckpt_path = require_existing_file(ENCODER_CKPT_PATH, "Encoder checkpoint")

    out_root = ensure_dir(OUTPUT_ROOT)
    data_dir = ensure_dir(out_root / "data")
    fig_dir = ensure_dir(out_root / "figures")
    table_dir = ensure_dir(out_root / "tables")

    cfg = load_config(config_path)
    tools = prepare_inference_tools(encoder_ckpt_path, decoder_ckpt_path)

    run_info = {
        "config_path": str(config_path),
        "decoder_ckpt_path": str(decoder_ckpt_path),
        "encoder_ckpt_path": str(encoder_ckpt_path),
        "random_test_n_cases": RANDOM_TEST_N_CASES,
        "random_test_seed": RANDOM_TEST_SEED,
        "device": DEVICE,
        "sensor_indices": tools["sensor_indices"].tolist(),
        "observation_vars": tools["observation_vars"],
        "obs_names": tools["obs_names"],
        "typical_cases": TYPICAL_CASES,
    }
    save_json(run_info, data_dir / "run_info_4_2.json")

    print("=" * 88)
    print("[4.2] Building random in-domain exact-solver test set...")
    random_cases = build_random_exact_test_set(cfg, RANDOM_TEST_N_CASES, RANDOM_TEST_SEED)

    print("=" * 88)
    print("[4.2] Evaluating BMN on random inversion test set...")
    random_result = evaluate_bmn_on_cases(tools, random_cases)

    if SAVE_RANDOM_CASE_RAW_DATA:
        np.savez_compressed(
            data_dir / "bmn_random_test_predictions.npz",
            s=random_result["s"],
            params_all=random_result["params_all"],
            mu_true=random_result["mu_true"],
            mu_pred=random_result["mu_pred"],
            obs_true=random_result["obs_true"],
            obs_pred=random_result["obs_pred"],
            y_true=random_result["y_true"],
            y_pred=random_result["y_pred"],
            mae_profile=random_result["mae_profile"],
            sensor_indices=tools["sensor_indices"],
            output_vars=np.asarray(tools["output_vars"]),
            observation_vars=np.asarray(tools["observation_vars"]),
        )

    # Save metrics
    metrics_json = {
        "parameter_metrics": random_result["param_metrics"],
        "response_metrics": random_result["response_metrics"],
        "observation_metrics": random_result["observation_metrics"],
        "feature_metrics": random_result["feature_metrics"],
    }
    save_json(metrics_json, data_dir / "bmn_random_test_metrics.json")

    param_rows = []
    for name, values in random_result["param_metrics"].items():
        row = {"parameter": name}
        row.update(values)
        param_rows.append(row)
    save_csv(param_rows, table_dir / "table_4_2_parameter_metrics.csv")

    response_rows = []
    for name, values in random_result["response_metrics"].items():
        row = {"response": name}
        row.update(values)
        response_rows.append(row)
    save_csv(response_rows, table_dir / "table_4_2_response_metrics.csv")

    feature_rows = [{"metric": k, "value": v} for k, v in random_result["feature_metrics"].items()]
    save_csv(feature_rows, table_dir / "table_4_2_feature_metrics.csv")

    obs_rows = [{"metric": k, "value": v} for k, v in random_result["observation_metrics"].items()]
    save_csv(obs_rows, table_dir / "table_4_2_observation_metrics.csv")

    # Plot aggregate results
    history = load_encoder_history_if_available(cfg)
    if history is not None:
        print("[4.2] Plotting Encoder training history...")
        plot_training_history(history, fig_dir)

    print("[4.2] Plotting aggregate figures...")
    plot_param_scatter(random_result["mu_true"], random_result["mu_pred"], fig_dir)
    plot_error_profile(random_result["s"], random_result["mae_profile"], tools["output_vars"], fig_dir)

    # Typical cases
    print("=" * 88)
    print("[4.2] Evaluating typical inversion cases...")
    typical_case_rows: List[Dict[str, Any]] = []
    typical_npz_data: Dict[str, Any] = {}

    for case in TYPICAL_CASES:
        exact = compute_exact_case(case, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            print(f"  WARNING: exact solver failed for typical case {case['case_id']}")
            continue
        s = np.asarray(exact["s"], dtype=np.float32)
        y_exact = np.stack([np.asarray(exact[name], dtype=np.float32) for name in tools["output_vars"]], axis=-1)
        pred = predict_case_from_exact(tools, s, case, y_exact)

        plot_typical_case(
            case_id=case["case_id"],
            desc=case["description"],
            s=s,
            y_true=pred["y_true"],
            y_pred=pred["y_pred"],
            sensor_indices=tools["sensor_indices"],
            output_vars=tools["output_vars"],
            fig_dir=fig_dir,
        )

        typical_npz_data[f"{case['case_id']}_s"] = s
        typical_npz_data[f"{case['case_id']}_mu_true"] = pred["mu_true"]
        typical_npz_data[f"{case['case_id']}_mu_pred"] = pred["mu_pred"]
        typical_npz_data[f"{case['case_id']}_obs_true"] = pred["obs_true"]
        typical_npz_data[f"{case['case_id']}_obs_pred"] = pred["obs_pred"]
        typical_npz_data[f"{case['case_id']}_y_true"] = pred["y_true"]
        typical_npz_data[f"{case['case_id']}_y_pred"] = pred["y_pred"]
        typical_npz_data[f"{case['case_id']}_params"] = case_dict_to_array(case)

        mu_names = ["Us", "Ub", "p"]
        row: Dict[str, Any] = {
            "case_id": case["case_id"],
            "description": case["description"],
            "Dx": case["Dx"],
            "ht": case["ht"],
        }
        for i, name in enumerate(mu_names):
            row[f"{name}_true"] = float(pred["mu_true"][i])
            row[f"{name}_pred"] = float(pred["mu_pred"][i])
            row[f"{name}_abs_error"] = float(abs(pred["mu_pred"][i] - pred["mu_true"][i]))

        idx_T = tools["output_vars"].index("T")
        idx_M = tools["output_vars"].index("M")
        row["T_top_true"] = float(pred["y_true"][-1, idx_T])
        row["T_top_pred"] = float(pred["y_pred"][-1, idx_T])
        row["T_top_abs_error"] = float(abs(pred["y_pred"][-1, idx_T] - pred["y_true"][-1, idx_T]))
        row["M_max_true"] = float(np.max(np.abs(pred["y_true"][:, idx_M])))
        row["M_max_pred"] = float(np.max(np.abs(pred["y_pred"][:, idx_M])))
        row["M_max_abs_error"] = float(abs(row["M_max_pred"] - row["M_max_true"]))
        typical_case_rows.append(row)

    if typical_npz_data:
        np.savez_compressed(data_dir / "bmn_typical_cases.npz", **typical_npz_data)
    save_csv(typical_case_rows, table_dir / "table_4_2_typical_cases.csv")

    summary = {
        "status": "completed",
        "elapsed_seconds": time.time() - t0,
        "output_root": str(out_root),
    }
    save_json(summary, out_root / "analysis_4_2_summary.json")

    print("=" * 88)
    print("Section 4.2 analysis finished.")
    print(f"Results saved to: {out_root}")
    print("=" * 88)


if __name__ == "__main__":
    main()
