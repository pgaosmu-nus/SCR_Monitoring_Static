#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from BMN_DD import FrozenDecoderAdapter, build_decoder_generated_encoder_dataset, load_encoder_model, train_encoder
from Decoder_DD import (
    BMNConfig,
    PARAM_NAMES,
    StandardScaler,
    compute_exact_case,
    extract_observations_from_fields,
    generate_decoder_dataset,
    load_config,
    train_decoder,
)
from paper_plot_style import (
    DEFAULT_EXPORT_CONFIG,
    DEFAULT_FIG_STYLE,
    apply_paper_style,
    columns_to_rows,
    create_subplots,
    finalize_axes,
    save_excel_workbook,
    save_figure,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "para_config.json"
DEFAULT_DECODER_CKPT = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "Decoder_DD_model.pth"
DEFAULT_SHARED_EVAL_DATASET = REPO_ROOT / "paper_outputs" / "paper_testset_4_2_4_4_in_domain_exact.npz"
DEFAULT_MODEL_ROOT = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs"

TYPICAL_CASES: List[Dict[str, Any]] = [
    {"case_id": "I1_low_flow", "description": "Low-current inversion case", "Dx": 1800.0, "ht": 0.0, "Us": 0.80, "Ub": 0.15, "p": 0.170},
    {"case_id": "I2_baseline", "description": "Baseline inversion case", "Dx": 1800.0, "ht": 0.0, "Us": 1.50, "Ub": 0.40, "p": 0.240},
    {"case_id": "I3_high_flow", "description": "High-current inversion case", "Dx": 1800.0, "ht": 0.0, "Us": 2.20, "Ub": 0.70, "p": 0.300},
    {"case_id": "I4_long_span", "description": "Large horizontal span inversion case", "Dx": 1880.0, "ht": 0.0, "Us": 1.50, "Ub": 0.40, "p": 0.240},
    {"case_id": "I5_top_offset", "description": "Top-height-offset inversion case", "Dx": 1800.0, "ht": 8.0, "Us": 1.50, "Ub": 0.40, "p": 0.240},
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def require_existing_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p


def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_device(requested: str) -> str:
    return "cpu" if requested == "cuda" and not torch.cuda.is_available() else requested


def case_dict_to_array(case: Dict[str, Any]) -> np.ndarray:
    return np.asarray([case[name] for name in PARAM_NAMES], dtype=np.float32)


def load_eval_dataset(path: str | Path) -> Dict[str, Any]:
    return load_npz(require_existing_file(path, "Evaluation dataset"))


def compute_metrics(
    s: np.ndarray,
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_vars: Sequence[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"parameter": {}, "response": {}, "features": {}}
    for i, name in enumerate(["Us", "Ub", "p"]):
        diff = mu_pred[:, i] - mu_true[:, i]
        value_range = float(np.max(mu_true[:, i]) - np.min(mu_true[:, i]))
        rmse = float(np.sqrt(np.mean(diff**2)))
        nrmse = float(rmse / value_range) if value_range > 1.0e-12 else None
        metrics["parameter"][name] = {
            "rmse": rmse,
            "mae": float(np.mean(np.abs(diff))),
            "mape": float(np.mean(np.abs(diff) / np.maximum(np.abs(mu_true[:, i]), 1.0e-8))),
            "nrmse": nrmse,
        }
    for j, name in enumerate(output_vars):
        diff = y_pred[:, :, j] - y_true[:, :, j]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        maxae = float(np.max(np.abs(diff)))
        value_range = float(np.max(y_true[:, :, j]) - np.min(y_true[:, :, j]))
        nrmse = float(rmse / value_range) if value_range > 1.0e-12 else None
        metrics["response"][name] = {"rmse": rmse, "mae": mae, "maxae": maxae, "nrmse": nrmse}
    if "T" in output_vars and "M" in output_vars:
        idx_T = list(output_vars).index("T")
        idx_M = list(output_vars).index("M")
        T_top_true = y_true[:, -1, idx_T]
        T_top_pred = y_pred[:, -1, idx_T]
        M_abs_true = np.abs(y_true[:, :, idx_M])
        M_abs_pred = np.abs(y_pred[:, :, idx_M])
        M_max_true = np.max(M_abs_true, axis=1)
        M_max_pred = np.max(M_abs_pred, axis=1)
        s_Mmax_true = s[np.argmax(M_abs_true, axis=1)]
        s_Mmax_pred = s[np.argmax(M_abs_pred, axis=1)]
        metrics["features"] = {
            "T_top_rmse": float(np.sqrt(np.mean((T_top_pred - T_top_true) ** 2))),
            "T_top_mae": float(np.mean(np.abs(T_top_pred - T_top_true))),
            "M_max_rmse": float(np.sqrt(np.mean((M_max_pred - M_max_true) ** 2))),
            "M_max_mae": float(np.mean(np.abs(M_max_pred - M_max_true))),
            "s_Mmax_rmse": float(np.sqrt(np.mean((s_Mmax_pred - s_Mmax_true) ** 2))),
            "s_Mmax_mae": float(np.mean(np.abs(s_Mmax_pred - s_Mmax_true))),
        }
    return metrics


def mean_response_nrmse(metrics: Dict[str, Any]) -> float:
    vals = [v["nrmse"] for v in metrics.get("response", {}).values() if isinstance(v, dict) and v.get("nrmse") is not None]
    return float(np.mean(vals)) if vals else float("nan")


def make_scenario_cfg(base_cfg: BMNConfig, scenario_output_dir: str | Path, scenario_tag: str) -> BMNConfig:
    cfg = copy.deepcopy(base_cfg)
    scenario_output_dir = str(Path(scenario_output_dir))
    cfg.dataset.output_dir = scenario_output_dir
    cfg.dataset.encoder_dataset_filename = f"bmn_encoder_dataset_{scenario_tag}.npz"
    cfg.encoder_training.model_filename = f"BMN_DD_encoder_{scenario_tag}.pth"
    cfg.encoder_training.history_filename = f"BMN_DD_history_{scenario_tag}.json"
    return cfg


def make_decoder_bmn_scenario_cfg(base_cfg: BMNConfig, scenario_output_dir: str | Path, scenario_tag: str) -> BMNConfig:
    cfg = copy.deepcopy(base_cfg)
    scenario_output_dir = str(Path(scenario_output_dir))
    cfg.dataset.output_dir = scenario_output_dir
    cfg.dataset.full_dataset_filename = f"decoder_fullfield_dataset_{scenario_tag}.npz"
    cfg.decoder_training.model_filename = f"Decoder_DD_model_{scenario_tag}.pth"
    cfg.decoder_training.history_filename = f"Decoder_DD_history_{scenario_tag}.json"
    cfg.dataset.encoder_dataset_filename = f"bmn_encoder_dataset_{scenario_tag}.npz"
    cfg.encoder_training.model_filename = f"BMN_DD_encoder_{scenario_tag}.pth"
    cfg.encoder_training.history_filename = f"BMN_DD_history_{scenario_tag}.json"
    return cfg


def train_or_reuse_bmn_scenario(
    cfg: BMNConfig,
    decoder_ckpt_path: str | Path,
    scenario_output_dir: str | Path,
    scenario_tag: str,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    scenario_output_dir = ensure_dir(scenario_output_dir)
    cfg = make_scenario_cfg(cfg, scenario_output_dir, scenario_tag)
    encoder_dataset_path = scenario_output_dir / cfg.dataset.encoder_dataset_filename
    encoder_ckpt_path = scenario_output_dir / cfg.encoder_training.model_filename
    history_path = scenario_output_dir / cfg.encoder_training.history_filename
    config_used_path = scenario_output_dir / f"scenario_config_{scenario_tag}.json"

    save_json(asdict(cfg), config_used_path)

    train_started = time.time()
    if force_retrain or not encoder_ckpt_path.exists():
        build_decoder_generated_encoder_dataset(cfg, decoder_ckpt_path, output_dir=scenario_output_dir)
        train_encoder(cfg, encoder_dataset_path, decoder_ckpt_path, output_dir=scenario_output_dir)
    train_elapsed = time.time() - train_started

    return {
        "scenario_output_dir": scenario_output_dir,
        "encoder_dataset_path": encoder_dataset_path,
        "encoder_ckpt_path": encoder_ckpt_path,
        "history_path": history_path,
        "config_used_path": config_used_path,
        "train_elapsed_seconds": float(train_elapsed),
    }


def train_or_reuse_decoder_then_bmn_scenario(
    cfg: BMNConfig,
    scenario_output_dir: str | Path,
    scenario_tag: str,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    scenario_output_dir = ensure_dir(scenario_output_dir)
    cfg = make_decoder_bmn_scenario_cfg(cfg, scenario_output_dir, scenario_tag)
    decoder_dataset_path = scenario_output_dir / cfg.dataset.full_dataset_filename
    decoder_ckpt_path = scenario_output_dir / cfg.decoder_training.model_filename
    decoder_history_path = scenario_output_dir / cfg.decoder_training.history_filename
    encoder_dataset_path = scenario_output_dir / cfg.dataset.encoder_dataset_filename
    encoder_ckpt_path = scenario_output_dir / cfg.encoder_training.model_filename
    encoder_history_path = scenario_output_dir / cfg.encoder_training.history_filename
    config_used_path = scenario_output_dir / f"scenario_config_{scenario_tag}.json"

    save_json(asdict(cfg), config_used_path)

    train_started = time.time()
    if force_retrain or not decoder_ckpt_path.exists():
        generate_decoder_dataset(cfg)
        train_decoder(cfg, decoder_dataset_path)
    if force_retrain or not encoder_ckpt_path.exists():
        build_decoder_generated_encoder_dataset(cfg, decoder_ckpt_path, output_dir=scenario_output_dir)
        train_encoder(cfg, encoder_dataset_path, decoder_ckpt_path, output_dir=scenario_output_dir)
    train_elapsed = time.time() - train_started

    return {
        "scenario_output_dir": scenario_output_dir,
        "decoder_dataset_path": decoder_dataset_path,
        "decoder_ckpt_path": decoder_ckpt_path,
        "decoder_history_path": decoder_history_path,
        "encoder_dataset_path": encoder_dataset_path,
        "encoder_ckpt_path": encoder_ckpt_path,
        "encoder_history_path": encoder_history_path,
        "config_used_path": config_used_path,
        "train_elapsed_seconds": float(train_elapsed),
    }


def load_bmn_bundle(encoder_ckpt_path: str | Path, decoder_ckpt_path: str | Path, device: str) -> Dict[str, Any]:
    encoder_model, encoder_ckpt = load_encoder_model(encoder_ckpt_path, map_location=device)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    obs_scaler = StandardScaler.from_dict(encoder_ckpt["obs_scaler"])
    mu_scaler = StandardScaler.from_dict(encoder_ckpt["mu_scaler"])
    sensor_indices = np.asarray(encoder_ckpt["sensor_indices"], dtype=np.int64)
    observation_vars = [str(v) for v in encoder_ckpt["observation_vars"]]
    output_vars = [str(v) for v in encoder_ckpt["output_vars"]]
    decoder_adapter = FrozenDecoderAdapter(decoder_ckpt_path, device=device)
    return {
        "encoder_model": encoder_model,
        "encoder_ckpt": encoder_ckpt,
        "obs_scaler": obs_scaler,
        "mu_scaler": mu_scaler,
        "sensor_indices": sensor_indices,
        "observation_vars": observation_vars,
        "output_vars": output_vars,
        "decoder_adapter": decoder_adapter,
        "device": device,
    }


def evaluate_bmn_checkpoint(
    encoder_ckpt_path: str | Path,
    decoder_ckpt_path: str | Path,
    dataset: Dict[str, Any],
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = load_bmn_bundle(encoder_ckpt_path, decoder_ckpt_path, device)
    s = np.asarray(dataset["s"], dtype=np.float32)
    c = np.asarray(dataset["c"], dtype=np.float32)
    y_true = np.asarray(dataset["y"], dtype=np.float32)
    mu_true = np.asarray(dataset["mu"], dtype=np.float32)
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    observations = extract_observations_from_fields(
        y=y_true,
        s=s,
        output_vars=output_vars,
        observation_vars=bundle["observation_vars"],
        sensor_indices=bundle["sensor_indices"],
    ).astype(np.float32)

    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    obs_s = torch.tensor(bundle["obs_scaler"].transform(observations), dtype=torch.float32, device=device)
    t0 = time.time()
    with torch.no_grad():
        mu_pred_s = bundle["encoder_model"](obs_s).detach().cpu().numpy()
        mu_pred = bundle["mu_scaler"].inverse_transform(mu_pred_s)
        y_pred = bundle["decoder_adapter"](
            s_t,
            torch.tensor(c, dtype=torch.float32, device=device),
            torch.tensor(mu_pred, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    elapsed = time.time() - t0

    metrics = compute_metrics(s, mu_true, mu_pred, y_true, y_pred, output_vars)
    metrics["timing"] = {
        "n_cases": int(len(c)),
        "total_seconds": float(elapsed),
        "mean_seconds_per_case": float(elapsed / max(len(c), 1)),
    }
    predictions = {
        "s": s,
        "mu_true": mu_true,
        "mu_pred": mu_pred.astype(np.float32),
        "y_true": y_true,
        "y_pred": y_pred.astype(np.float32),
        "observations": observations,
        "c": c,
    }
    return metrics, predictions


def predict_bmn_typical_case(bundle: Dict[str, Any], cfg: BMNConfig, case: Dict[str, Any]) -> Dict[str, Any]:
    exact = compute_exact_case(case, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
    if exact is None:
        raise RuntimeError(f"Exact solver failed for case {case['case_id']}")
    s = np.asarray(exact["s"], dtype=np.float32)
    y_exact = np.stack([np.asarray(exact[name], dtype=np.float32) for name in bundle["output_vars"]], axis=-1)
    obs_true = extract_observations_from_fields(
        y_exact[None, :, :],
        s,
        bundle["output_vars"],
        bundle["observation_vars"],
        bundle["sensor_indices"],
    )[0].astype(np.float32)
    obs_s = bundle["obs_scaler"].transform(obs_true[None, :])
    with torch.no_grad():
        mu_pred_s = bundle["encoder_model"](torch.tensor(obs_s, dtype=torch.float32, device=bundle["device"])).detach().cpu().numpy()
        mu_pred = bundle["mu_scaler"].inverse_transform(mu_pred_s)[0]
    c = np.asarray([case["Dx"], case["ht"]], dtype=np.float32)
    y_pred = bundle["decoder_adapter"](
        torch.tensor(s, dtype=torch.float32, device=bundle["device"]),
        torch.tensor(c[None, :], dtype=torch.float32, device=bundle["device"]),
        torch.tensor(mu_pred[None, :], dtype=torch.float32, device=bundle["device"]),
    ).detach().cpu().numpy()[0]
    return {
        "s": s,
        "y_true": y_exact.astype(np.float32),
        "y_pred": y_pred.astype(np.float32),
        "mu_true": np.asarray([case["Us"], case["Ub"], case["p"]], dtype=np.float32),
        "mu_pred": mu_pred.astype(np.float32),
    }


def save_metrics_bundle(
    metrics_by_label: Dict[str, Dict[str, Any]],
    out_dir: str | Path,
    filename_prefix: str,
    label_key: str,
) -> None:
    out_dir = ensure_dir(out_dir)
    save_json(metrics_by_label, Path(out_dir) / f"{filename_prefix}_metrics.json")

    param_rows: List[Dict[str, Any]] = []
    response_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for label, metrics in metrics_by_label.items():
        for param_name, vals in metrics["parameter"].items():
            row = {label_key: label, "parameter": param_name}
            row.update(vals)
            param_rows.append(row)
        for resp_name, vals in metrics["response"].items():
            row = {label_key: label, "response": resp_name}
            row.update(vals)
            response_rows.append(row)
        for feat_name, val in metrics["features"].items():
            feature_rows.append({label_key: label, "metric": feat_name, "value": val})
        summary_rows.append(
            {
                label_key: label,
                "Us_rmse": metrics["parameter"]["Us"]["rmse"],
                "Ub_rmse": metrics["parameter"]["Ub"]["rmse"],
                "p_rmse": metrics["parameter"]["p"]["rmse"],
                "mean_response_nrmse": mean_response_nrmse(metrics),
                "mean_time_per_case_s": metrics.get("timing", {}).get("mean_seconds_per_case"),
                "total_time_s": metrics.get("timing", {}).get("total_seconds"),
            }
        )
    save_csv(param_rows, Path(out_dir) / f"{filename_prefix}_parameter_metrics.csv")
    save_csv(response_rows, Path(out_dir) / f"{filename_prefix}_response_metrics.csv")
    save_csv(feature_rows, Path(out_dir) / f"{filename_prefix}_feature_metrics.csv")
    save_csv(summary_rows, Path(out_dir) / f"{filename_prefix}_summary.csv")


def plot_metric_curves(
    x_values: Sequence[float],
    series: Dict[str, Sequence[float]],
    x_label: str,
    y_label: str,
    title: str,
    save_name: str,
    fig_dir: str | Path,
    x_tick_labels: Sequence[str] | None = None,
) -> None:
    fig, ax = create_subplots(kind="single", style=DEFAULT_FIG_STYLE)
    for name, vals in series.items():
        ax.plot(x_values, vals, marker="o", label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if x_tick_labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_tick_labels)
    finalize_axes(ax)
    save_figure(fig, ensure_dir(fig_dir), save_name, style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        Path(fig_dir) / f"{save_name}.xlsx",
        {"data": columns_to_rows({x_label: x_values, **{name: vals for name, vals in series.items()}})},
    )
    plt.close(fig)


def plot_grouped_bars(
    categories: Sequence[str],
    series: Dict[str, Sequence[float]],
    x_label: str,
    y_label: str,
    title: str,
    save_name: str,
    fig_dir: str | Path,
) -> None:
    fig, ax = create_subplots(kind="single", style=DEFAULT_FIG_STYLE)
    categories = list(categories)
    labels = list(series.keys())
    x = np.arange(len(categories), dtype=float)
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        vals = np.asarray(series[label], dtype=float)
        offset = (i - 0.5 * (len(labels) - 1)) * width
        ax.bar(x + offset, vals, width=width, label=label)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=False)
    finalize_axes(ax)
    save_figure(fig, ensure_dir(fig_dir), save_name, style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        Path(fig_dir) / f"{save_name}.xlsx",
        {
            "data": [
                {"category": category, **{label: series[label][i] for label in labels}}
                for i, category in enumerate(categories)
            ]
        },
    )
    plt.close(fig)


def plot_typical_case_family(
    case_id: str,
    desc: str,
    s: np.ndarray,
    y_exact: np.ndarray,
    series_by_label: Dict[str, np.ndarray],
    output_vars: Sequence[str],
    fig_dir: str | Path,
    save_name: str,
    sensor_indices: np.ndarray | None = None,
    show_sensor_positions: bool = False,
) -> None:
    fig, axes = create_subplots(2, 2, kind="quad", style=DEFAULT_FIG_STYLE)
    idx = {name: i for i, name in enumerate(output_vars)}
    ax1, ax2, ax3, ax4 = axes.flat
    ax1.plot(y_exact[:, idx["x"]], y_exact[:, idx["z"]], color="tab:blue", linestyle="-", label="Exact")
    ax2.plot(s, y_exact[:, idx["theta"]], color="tab:blue", linestyle="-", label="Exact")
    ax3.plot(s, y_exact[:, idx["T"]], color="tab:blue", linestyle="-", label="Exact")
    ax4.plot(s, y_exact[:, idx["M"]], color="tab:blue", linestyle="-", label="Exact")

    cmap = plt.get_cmap("viridis")
    labels = list(series_by_label.keys())
    for i, label in enumerate(labels):
        color = cmap(i / max(len(labels) - 1, 1))
        y = series_by_label[label]
        ax1.plot(y[:, idx["x"]], y[:, idx["z"]], linestyle="--", color=color, label=label)
        ax2.plot(s, y[:, idx["theta"]], linestyle="--", color=color, label=label)
        ax3.plot(s, y[:, idx["T"]], linestyle="--", color=color, label=label)
        ax4.plot(s, y[:, idx["M"]], linestyle="--", color=color, label=label)

    if show_sensor_positions and sensor_indices is not None:
        sensor_indices = np.asarray(sensor_indices, dtype=int)
        sensor_indices = sensor_indices[(sensor_indices >= 0) & (sensor_indices < len(s))]
        if sensor_indices.size > 0:
            ax1.scatter(y_exact[sensor_indices, idx["x"]], y_exact[sensor_indices, idx["z"]], color="black", s=12, zorder=5)
            ax2.scatter(s[sensor_indices], y_exact[sensor_indices, idx["theta"]], color="black", s=12, zorder=5)
            ax3.scatter(s[sensor_indices], y_exact[sensor_indices, idx["T"]], color="black", s=12, zorder=5)
            ax4.scatter(s[sensor_indices], y_exact[sensor_indices, idx["M"]], color="black", s=12, zorder=5)
            for ax in (ax2, ax3, ax4):
                for sensor_s in s[sensor_indices]:
                    ax.axvline(float(sensor_s), color="0.55", linestyle=":", linewidth=0.9, zorder=0)

    ax1.set_xlabel(r"$x$ ($\mathrm{m}$)")
    ax1.set_ylabel(r"$z$ ($\mathrm{m}$)")
    ax1.set_title("(a) SCR configuration")
    ax2.set_xlabel(r"$s$ ($\mathrm{m}$)")
    ax2.set_ylabel(r"$\theta$ ($\mathrm{rad}$)")
    ax2.set_title("(b) Tangent angle")
    ax3.set_xlabel(r"$s$ ($\mathrm{m}$)")
    ax3.set_ylabel(r"$T$ ($\mathrm{N}$)")
    ax3.set_title("(c) Effective tension")
    ax4.set_xlabel(r"$s$ ($\mathrm{m}$)")
    ax4.set_ylabel(r"$M$ ($\mathrm{N\,m}$)")
    ax4.set_title("(d) Bending moment")
    fig.suptitle(f"{case_id}: {desc}")
    finalize_axes(axes)
    save_figure(fig, ensure_dir(fig_dir), save_name, style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        Path(fig_dir) / f"{save_name}.xlsx",
        {case_id: _typical_case_response_rows(s, y_exact, series_by_label, idx)},
    )
    plt.close(fig)


def plot_typical_case_family_collection(
    cases: Sequence[Dict[str, Any]],
    case_results: Dict[str, Dict[str, Any]],
    output_vars: Sequence[str],
    fig_dir: str | Path,
    save_name: str,
    sensor_indices: np.ndarray | None = None,
    show_sensor_positions: bool = False,
    title: str | None = None,
) -> None:
    fig, axes = create_subplots(2, 2, kind="quad", style=DEFAULT_FIG_STYLE, sharex=False, sharey=False)
    idx = {name: i for i, name in enumerate(output_vars)}
    ax1, ax2, ax3, ax4 = axes.flat
    case_cmap = plt.get_cmap("tab10")

    sensor_indices_arr = None if sensor_indices is None else np.asarray(sensor_indices, dtype=int)

    for row, case in enumerate(cases):
        result = case_results[case["case_id"]]
        s = result["s"]
        y_exact = result["y_true"]
        series_by_label = result["series_by_label"]
        case_color = case_cmap(row % 10)

        exact_label = "Exact" if row == 0 else "_nolegend_"
        ax1.plot(y_exact[:, idx["x"]], y_exact[:, idx["z"]], color="tab:blue", linestyle="-", label=exact_label)
        ax2.plot(s, y_exact[:, idx["theta"]], color="tab:blue", linestyle="-", label=exact_label)
        ax3.plot(s, y_exact[:, idx["T"]], color="tab:blue", linestyle="-", label=exact_label)
        ax4.plot(s, y_exact[:, idx["M"]], color="tab:blue", linestyle="-", label=exact_label)

        cmap = plt.get_cmap("viridis")
        labels = list(series_by_label.keys())
        for i, label in enumerate(labels):
            color = cmap(i / max(len(labels) - 1, 1))
            y = series_by_label[label]
            curve_label = label if row == 0 else "_nolegend_"
            ax1.plot(y[:, idx["x"]], y[:, idx["z"]], linestyle="--", color=color, label=curve_label)
            ax2.plot(s, y[:, idx["theta"]], linestyle="--", color=color, label=curve_label)
            ax3.plot(s, y[:, idx["T"]], linestyle="--", color=color, label=curve_label)
            ax4.plot(s, y[:, idx["M"]], linestyle="--", color=color, label=curve_label)

        if show_sensor_positions and sensor_indices_arr is not None:
            sensor_indices_safe = sensor_indices_arr[(sensor_indices_arr >= 0) & (sensor_indices_arr < len(s))]
            if sensor_indices_safe.size > 0:
                ax1.scatter(
                    y_exact[sensor_indices_safe, idx["x"]],
                    y_exact[sensor_indices_safe, idx["z"]],
                    color=[case_color],
                    s=12,
                    zorder=5,
                )
                ax2.scatter(s[sensor_indices_safe], y_exact[sensor_indices_safe, idx["theta"]], color=[case_color], s=12, zorder=5)
                ax3.scatter(s[sensor_indices_safe], y_exact[sensor_indices_safe, idx["T"]], color=[case_color], s=12, zorder=5)
                ax4.scatter(s[sensor_indices_safe], y_exact[sensor_indices_safe, idx["M"]], color=[case_color], s=12, zorder=5)

    ax1.set_xlabel(r"$x$ ($\mathrm{m}$)")
    ax1.set_ylabel(r"$z$ ($\mathrm{m}$)")
    ax1.set_title("(a) SCR configuration")
    ax2.set_xlabel(r"$s$ ($\mathrm{m}$)")
    ax2.set_ylabel(r"$\theta$ ($\mathrm{rad}$)")
    ax2.set_title("(b) Tangent angle")
    ax3.set_xlabel(r"$s$ ($\mathrm{m}$)")
    ax3.set_ylabel(r"$T$ ($\mathrm{N}$)")
    ax3.set_title("(c) Effective tension")
    ax4.set_xlabel(r"$s$ ($\mathrm{m}$)")
    ax4.set_ylabel(r"$M$ ($\mathrm{N\,m}$)")
    ax4.set_title("(d) Bending moment")

    finalize_axes(axes, legend=False)
    ax1.legend(frameon=False, loc="best")
    if title:
        fig.suptitle(title, y=0.995)
    save_figure(fig, ensure_dir(fig_dir), save_name, style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        Path(fig_dir) / f"{save_name}.xlsx",
        {
            case["case_id"]: _typical_case_response_rows(
                case_results[case["case_id"]]["s"],
                case_results[case["case_id"]]["y_true"],
                case_results[case["case_id"]]["series_by_label"],
                idx,
            )
            for case in cases
            if case["case_id"] in case_results
        },
    )
    plt.close(fig)


def apply_style() -> None:
    apply_paper_style(DEFAULT_FIG_STYLE)


def _typical_case_response_rows(
    s: np.ndarray,
    y_exact: np.ndarray,
    series_by_label: Dict[str, np.ndarray],
    idx: Dict[str, int],
) -> List[Dict[str, Any]]:
    columns: Dict[str, Sequence[Any]] = {
        "s_m": s,
        "x_exact_m": y_exact[:, idx["x"]],
        "z_exact_m": y_exact[:, idx["z"]],
        "theta_exact_rad": y_exact[:, idx["theta"]],
        "T_exact_N": y_exact[:, idx["T"]],
        "M_exact_N_m": y_exact[:, idx["M"]],
    }
    for label, y in series_by_label.items():
        clean = str(label).replace(" ", "_")
        columns[f"x_{clean}_m"] = y[:, idx["x"]]
        columns[f"z_{clean}_m"] = y[:, idx["z"]]
        columns[f"theta_{clean}_rad"] = y[:, idx["theta"]]
        columns[f"T_{clean}_N"] = y[:, idx["T"]]
        columns[f"M_{clean}_N_m"] = y[:, idx["M"]]
    return columns_to_rows(columns)
