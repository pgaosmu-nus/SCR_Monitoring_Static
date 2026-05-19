#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_4_4_model_comparison.py

Section 4.4: unified comparison of BMN, DirectMapping, and
DecoderOptimization baselines on the same evaluation cases.

Default behavior
----------------
1. Load the BMN encoder checkpoint.
2. Use its saved encoder dataset path and test split as the common evaluation set.
3. Re-run BMN, DirectMapping, and DecoderOptimization on those same cases.
4. Save unified metrics, raw predictions, and comparison figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from BMN_DD import FrozenDecoderAdapter, extract_observation_torch, load_encoder_model
from Decoder_DD import (
    BMNConfig,
    MU_NAMES,
    PARAM_NAMES,
    StandardScaler,
    compute_exact_case,
    extract_observations_from_fields,
    load_config,
    sample_one_case,
)
from DecoderOptimizationInversion import (
    get_mu_bounds,
    load_obs_scaler_from_encoder,
    optimize_single_case,
)
from DirectMapping_DD import load_direct_model, predict_direct_fullfield_np
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
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_outputs" / "4_4_model_comparison"
DEFAULT_BMN_CKPT = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "BMN_DD_encoder.pth"
DEFAULT_DECODER_CKPT = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "Decoder_DD_model.pth"
DEFAULT_DIRECT_CKPT = REPO_ROOT / "outputs" / "BMN_SCR_DD_outputs" / "DirectMapping_baseline" / "DirectMapping_DD_model.pth"
DEFAULT_EVAL_DATASET = REPO_ROOT / "paper_outputs" / "paper_testset_4_2_4_4_in_domain_exact.npz"

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

METHOD_STYLES: Dict[str, Dict[str, str]] = {
    "Exact": {"color": "tab:blue", "linestyle": "-"},
    "BMN": {"color": "tab:orange", "linestyle": "--"},
    "DirectMapping": {"color": "tab:green", "linestyle": "-."},
    "DecoderOptimization": {"color": "tab:red", "linestyle": ":"},
}


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
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def case_dict_to_array(case: Dict[str, Any]) -> np.ndarray:
    return np.asarray([case[name] for name in PARAM_NAMES], dtype=np.float32)


def build_random_exact_test_set(cfg: BMNConfig, n_cases: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases: List[Dict[str, Any]] = []
    while len(cases) < n_cases:
        sampled = sample_one_case(rng, cfg.ranges, cfg.physical)
        exact = compute_exact_case(sampled, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            continue
        cases.append({"params": sampled, "exact": exact})
    return cases


def random_cases_to_dataset(cfg: BMNConfig, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cases:
        raise ValueError("No exact test cases were generated.")
    output_vars = list(cfg.dataset.output_vars)
    s = np.asarray(cases[0]["exact"]["s"], dtype=np.float32)
    y = np.zeros((len(cases), len(s), len(output_vars)), dtype=np.float32)
    params = np.zeros((len(cases), len(PARAM_NAMES)), dtype=np.float32)
    c = np.zeros((len(cases), 2), dtype=np.float32)
    mu = np.zeros((len(cases), len(MU_NAMES)), dtype=np.float32)
    for i, item in enumerate(cases):
        case = item["params"]
        exact = item["exact"]
        params[i] = case_dict_to_array(case)
        c[i] = np.asarray([case["Dx"], case["ht"]], dtype=np.float32)
        mu[i] = np.asarray([case["Us"], case["Ub"], case["p"]], dtype=np.float32)
        y[i] = np.stack([np.asarray(exact[name], dtype=np.float32) for name in output_vars], axis=-1)
    return {
        "s": s,
        "c": c,
        "mu": mu,
        "params": params,
        "y": y,
        "output_vars": np.asarray(output_vars),
        "source": np.asarray("random_exact_test_set"),
    }


def compute_metrics(
    s: np.ndarray,
    mu_true: np.ndarray,
    mu_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_vars: Sequence[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"parameter": {}, "response": {}, "features": {}}
    for i, name in enumerate(MU_NAMES):
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


def extract_method_observations(
    y_all: np.ndarray,
    s: np.ndarray,
    output_vars: Sequence[str],
    observation_vars: Sequence[str],
    sensor_indices: np.ndarray,
) -> np.ndarray:
    return extract_observations_from_fields(
        y=y_all,
        s=s,
        output_vars=output_vars,
        observation_vars=observation_vars,
        sensor_indices=sensor_indices,
    ).astype(np.float32)


def evaluate_bmn_method(
    bmn_ckpt_path: Path,
    decoder_ckpt_path: Path,
    dataset: Dict[str, Any],
    case_indices: np.ndarray,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    encoder, eckpt = load_encoder_model(bmn_ckpt_path, map_location=device)
    encoder = encoder.to(device)
    encoder.eval()
    obs_scaler = StandardScaler.from_dict(eckpt["obs_scaler"])
    mu_scaler = StandardScaler.from_dict(eckpt["mu_scaler"])
    s = np.asarray(dataset["s"], dtype=np.float32)
    c = np.asarray(dataset["c"], dtype=np.float32)[case_indices]
    y_true = np.asarray(dataset["y"], dtype=np.float32)[case_indices]
    mu_true = np.asarray(dataset["mu"], dtype=np.float32)[case_indices]
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    if "observations" in dataset:
        observations = np.asarray(dataset["observations"], dtype=np.float32)[case_indices]
    else:
        sensor_indices = np.asarray(eckpt["sensor_indices"], dtype=np.int64)
        observation_vars = [str(v) for v in eckpt["observation_vars"]]
        observations = extract_method_observations(
            np.asarray(dataset["y"], dtype=np.float32),
            s,
            output_vars,
            observation_vars,
            sensor_indices,
        )[case_indices]

    decoder_adapter = FrozenDecoderAdapter(decoder_ckpt_path, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    obs_s = torch.tensor(obs_scaler.transform(observations), dtype=torch.float32, device=device)

    t0 = time.time()
    with torch.no_grad():
        mu_pred_s = encoder(obs_s).detach().cpu().numpy()
        mu_pred = mu_scaler.inverse_transform(mu_pred_s)
        y_pred = decoder_adapter(
            s_t,
            torch.tensor(c, dtype=torch.float32, device=device),
            torch.tensor(mu_pred, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    elapsed = time.time() - t0

    metrics = compute_metrics(s, mu_true, mu_pred, y_true, y_pred, output_vars)
    metrics["timing"] = {
        "n_cases": int(len(case_indices)),
        "total_seconds": float(elapsed),
        "mean_seconds_per_case": float(elapsed / max(len(case_indices), 1)),
    }
    predictions = {
        "mu_true": mu_true,
        "mu_pred": mu_pred.astype(np.float32),
        "y_true": y_true,
        "y_pred": y_pred.astype(np.float32),
        "case_indices": case_indices,
    }
    return metrics, predictions


def evaluate_direct_method(
    direct_ckpt_path: Path,
    dataset: Dict[str, Any],
    case_indices: np.ndarray,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model, ckpt = load_direct_model(direct_ckpt_path, map_location=device)
    s = np.asarray(dataset["s"], dtype=np.float32)
    c_all = np.asarray(dataset["c"], dtype=np.float32)
    y_all = np.asarray(dataset["y"], dtype=np.float32)
    mu_true = np.asarray(dataset["mu"], dtype=np.float32)[case_indices]
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    observation_vars = [str(v) for v in ckpt["observation_vars"]]
    sensor_indices = np.asarray(ckpt["sensor_indices"], dtype=np.int64)
    observations = extract_method_observations(y_all, s, output_vars, observation_vars, sensor_indices)
    y_true = y_all[case_indices]
    y_pred = np.zeros_like(y_true)

    t0 = time.time()
    for k, case_id in enumerate(case_indices):
        y_pred[k] = predict_direct_fullfield_np(
            model=model,
            checkpoint=ckpt,
            s=s,
            c=c_all[case_id],
            observation=observations[case_id],
            device=device,
        )
    elapsed = time.time() - t0

    metrics = compute_metrics(s, mu_true, np.full_like(mu_true, np.nan, dtype=np.float32), y_true, y_pred, output_vars)
    metrics["parameter"] = "not_available_for_direct_mapping"
    metrics["timing"] = {
        "n_cases": int(len(case_indices)),
        "total_seconds": float(elapsed),
        "mean_seconds_per_case": float(elapsed / max(len(case_indices), 1)),
    }
    predictions = {
        "mu_true": mu_true,
        "mu_pred": None,
        "y_true": y_true,
        "y_pred": y_pred.astype(np.float32),
        "case_indices": case_indices,
    }
    return metrics, predictions


def evaluate_decoderopt_method(
    cfg: BMNConfig,
    decoder_ckpt_path: Path,
    bmn_ckpt_path: Path,
    dataset: Dict[str, Any],
    case_indices: np.ndarray,
    n_steps: int,
    lr: float,
    num_restarts: int,
    lambda_order: float,
    lbfgs_steps: int,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    s = np.asarray(dataset["s"], dtype=np.float32)
    c_all = np.asarray(dataset["c"], dtype=np.float32)
    y_all = np.asarray(dataset["y"], dtype=np.float32)
    mu_true = np.asarray(dataset["mu"], dtype=np.float32)[case_indices]
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    observation_vars = list(cfg.dataset.observation_vars)
    sensor_indices = np.asarray(dataset["sensor_indices"], dtype=np.int64) if "sensor_indices" in dataset else np.asarray([], dtype=np.int64)
    if sensor_indices.size == 0:
        eckpt = torch.load(bmn_ckpt_path, map_location="cpu")
        sensor_indices = np.asarray(eckpt["sensor_indices"], dtype=np.int64)
        observation_vars = [str(v) for v in eckpt["observation_vars"]]
    obs_all = extract_method_observations(y_all, s, output_vars, observation_vars, sensor_indices)
    obs_scaler = load_obs_scaler_from_encoder(bmn_ckpt_path)

    decoder_adapter = FrozenDecoderAdapter(decoder_ckpt_path, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    obs_mean_t = torch.tensor(obs_scaler.mean, dtype=torch.float32, device=device)
    obs_std_t = torch.tensor(obs_scaler.std, dtype=torch.float32, device=device)
    mu_lower, mu_upper = get_mu_bounds(cfg)
    mu_lower_t = torch.tensor(mu_lower, dtype=torch.float32, device=device)
    mu_upper_t = torch.tensor(mu_upper, dtype=torch.float32, device=device)

    y_true = y_all[case_indices]
    y_pred = np.zeros_like(y_true)
    mu_pred = np.zeros_like(mu_true)

    t0 = time.time()
    for k, case_id in enumerate(case_indices):
        result = optimize_single_case(
            decoder_adapter=decoder_adapter,
            s_t=s_t,
            c_one_t=torch.tensor(c_all[case_id : case_id + 1], dtype=torch.float32, device=device),
            obs_true_one_t=torch.tensor(obs_all[case_id : case_id + 1], dtype=torch.float32, device=device),
            output_vars=output_vars,
            observation_vars=observation_vars,
            sensor_indices=sensor_indices,
            obs_mean_t=obs_mean_t,
            obs_std_t=obs_std_t,
            mu_lower_t=mu_lower_t,
            mu_upper_t=mu_upper_t,
            n_steps=n_steps,
            lr=lr,
            num_restarts=num_restarts,
            lambda_order=lambda_order,
            lbfgs_steps=lbfgs_steps,
            seed=int(cfg.encoder_training.seed) + 7919 * int(k),
            device=device,
        )
        mu_pred[k] = result["mu_pred"]
        y_pred[k] = result["y_pred"]
    elapsed = time.time() - t0

    metrics = compute_metrics(s, mu_true, mu_pred, y_true, y_pred, output_vars)
    metrics["timing"] = {
        "n_cases": int(len(case_indices)),
        "total_seconds": float(elapsed),
        "mean_seconds_per_case": float(elapsed / max(len(case_indices), 1)),
    }
    predictions = {
        "mu_true": mu_true,
        "mu_pred": mu_pred.astype(np.float32),
        "y_true": y_true,
        "y_pred": y_pred.astype(np.float32),
        "case_indices": case_indices,
    }
    return metrics, predictions


def load_decoderopt_predictions_if_compatible(
    predictions_path: Path,
    dataset: Dict[str, Any],
    case_indices: np.ndarray,
    n_steps: int,
    lr: float,
    num_restarts: int,
    lambda_order: float,
    lbfgs_steps: int,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if not predictions_path.exists():
        return None
    data = load_npz(predictions_path)
    required = ["case_indices", "params", "mu_true", "mu_pred", "y_true", "y_pred", "c", "s", "output_vars"]
    if any(key not in data for key in required):
        return None

    saved_case_indices = np.asarray(data["case_indices"], dtype=np.int64)
    if saved_case_indices.shape != case_indices.shape or not np.array_equal(saved_case_indices, case_indices):
        return None

    saved_s = np.asarray(data["s"], dtype=np.float32)
    expected_s = np.asarray(dataset["s"], dtype=np.float32)
    if saved_s.shape != expected_s.shape or not np.allclose(saved_s, expected_s):
        return None

    saved_output_vars = [str(v) for v in np.asarray(data["output_vars"]).tolist()]
    expected_output_vars = [str(v) for v in np.asarray(dataset["output_vars"]).tolist()]
    if saved_output_vars != expected_output_vars:
        return None

    if "params" not in dataset or "mu" not in dataset:
        return None
    expected_params = np.asarray(dataset["params"], dtype=np.float32)[case_indices]
    expected_mu_true = np.asarray(dataset["mu"], dtype=np.float32)[case_indices]
    expected_c = np.asarray(dataset["c"], dtype=np.float32)[case_indices]
    if not np.allclose(np.asarray(data["params"], dtype=np.float32), expected_params):
        return None
    if not np.allclose(np.asarray(data["mu_true"], dtype=np.float32), expected_mu_true):
        return None
    if not np.allclose(np.asarray(data["c"], dtype=np.float32), expected_c):
        return None

    saved_steps = int(np.asarray(data.get("optimization_n_steps", np.asarray(-1))).item())
    saved_restarts = int(np.asarray(data.get("optimization_num_restarts", np.asarray(-1))).item())
    saved_lbfgs_steps = int(np.asarray(data.get("optimization_lbfgs_steps", np.asarray(-1))).item())
    saved_lr = float(np.asarray(data.get("optimization_lr", np.asarray(np.nan, dtype=np.float32))).item())
    saved_lambda_order = float(np.asarray(data.get("optimization_lambda_order", np.asarray(np.nan, dtype=np.float32))).item())
    if (
        saved_steps != int(n_steps)
        or saved_restarts != int(num_restarts)
        or saved_lbfgs_steps != int(lbfgs_steps)
        or not np.isclose(saved_lr, float(lr))
        or not np.isclose(saved_lambda_order, float(lambda_order))
    ):
        return None

    y_true = np.asarray(data["y_true"], dtype=np.float32)
    y_pred = np.asarray(data["y_pred"], dtype=np.float32)
    mu_true = np.asarray(data["mu_true"], dtype=np.float32)
    mu_pred = np.asarray(data["mu_pred"], dtype=np.float32)
    metrics = compute_metrics(saved_s, mu_true, mu_pred, y_true, y_pred, expected_output_vars)
    if "time_per_case" in data:
        time_per_case = np.asarray(data["time_per_case"], dtype=np.float32)
        metrics["timing"] = {
            "n_cases": int(len(case_indices)),
            "total_seconds": float(np.sum(time_per_case)),
            "mean_seconds_per_case": float(np.mean(time_per_case)),
        }
    else:
        metrics["timing"] = {
            "n_cases": int(len(case_indices)),
            "total_seconds": None,
            "mean_seconds_per_case": None,
        }
    metrics["source"] = f"loaded_from:{predictions_path}"
    predictions = {
        "mu_true": mu_true,
        "mu_pred": mu_pred,
        "y_true": y_true,
        "y_pred": y_pred,
        "case_indices": saved_case_indices,
    }
    return metrics, predictions


def build_parameter_rows(all_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in all_metrics.items():
        param_metrics = metrics.get("parameter")
        if not isinstance(param_metrics, dict):
            continue
        for name, vals in param_metrics.items():
            row = {"method": method, "parameter": name}
            row.update(vals)
            rows.append(row)
    return rows


def build_response_rows(all_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in all_metrics.items():
        for name, vals in metrics.get("response", {}).items():
            row = {"method": method, "response": name}
            row.update(vals)
            rows.append(row)
    return rows


def build_feature_rows(all_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in all_metrics.items():
        for name, val in metrics.get("features", {}).items():
            rows.append({"method": method, "metric": name, "value": val})
    return rows


def build_tradeoff_rows(all_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, metrics in all_metrics.items():
        param_metrics = metrics.get("parameter")
        response_metrics = metrics.get("response", {})
        timing = metrics.get("timing", {})
        response_nrmse_vals = [
            vals["nrmse"] for vals in response_metrics.values()
            if isinstance(vals, dict) and vals.get("nrmse") is not None
        ]
        rows.append(
            {
                "method": method,
                "parameter_estimation_available": "Yes" if isinstance(param_metrics, dict) else "No",
                "response_reconstruction_available": "Yes",
                "one_shot_inference": "Yes" if method in {"BMN", "DirectMapping"} else "No",
                "mean_time_per_case_s": timing.get("mean_seconds_per_case"),
                "total_time_s": timing.get("total_seconds"),
                "Us_rmse": param_metrics["Us"]["rmse"] if isinstance(param_metrics, dict) else None,
                "Ub_rmse": param_metrics["Ub"]["rmse"] if isinstance(param_metrics, dict) else None,
                "p_rmse": param_metrics["p"]["rmse"] if isinstance(param_metrics, dict) else None,
                "mean_response_nrmse": float(np.mean(response_nrmse_vals)) if response_nrmse_vals else None,
            }
        )
    return rows


def plot_parameter_nrmse(all_metrics: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    methods = [m for m, metrics in all_metrics.items() if isinstance(metrics.get("parameter"), dict)]
    params = list(MU_NAMES)
    fig, ax = create_subplots(kind="single", style=DEFAULT_FIG_STYLE)
    x = np.arange(len(params))
    width = 0.24
    for i, method in enumerate(methods):
        vals = [all_metrics[method]["parameter"][p]["nrmse"] for p in params]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_ylabel("NRMSE")
    ax.set_title("Parameter NRMSE comparison")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_4_parameter_nrmse_comparison", style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        fig_dir / "fig_4_4_parameter_nrmse_comparison.xlsx",
        {
            "parameter_nrmse": [
                {"parameter": param, **{method: all_metrics[method]["parameter"][param]["nrmse"] for method in methods}}
                for param in params
            ]
        },
    )
    plt.close(fig)


def plot_response_nrmse(all_metrics: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    methods = list(all_metrics.keys())
    responses = list(next(iter(all_metrics.values()))["response"].keys())
    fig, ax = create_subplots(kind="single", style=DEFAULT_FIG_STYLE)
    x = np.arange(len(responses))
    width = 0.24
    for i, method in enumerate(methods):
        vals = [all_metrics[method]["response"][r]["nrmse"] for r in responses]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(responses)
    ax.set_ylabel("NRMSE")
    ax.set_title("Response NRMSE comparison")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_4_response_nrmse_comparison", style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        fig_dir / "fig_4_4_response_nrmse_comparison.xlsx",
        {
            "response_nrmse": [
                {"response": response, **{method: all_metrics[method]["response"][response]["nrmse"] for method in methods}}
                for response in responses
            ]
        },
    )
    plt.close(fig)


def plot_mean_time_per_case(all_metrics: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    methods = list(all_metrics.keys())
    vals = [all_metrics[m].get("timing", {}).get("mean_seconds_per_case") for m in methods]
    fig, ax = create_subplots(kind="single", style=DEFAULT_FIG_STYLE)
    x = np.arange(len(methods))
    ax.bar(x, vals, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean time per case (s)")
    ax.set_title("Inference cost comparison")
    if all(v is not None and v > 0 for v in vals):
        ax.set_yscale("log")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_4_mean_time_per_case", style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        fig_dir / "fig_4_4_mean_time_per_case.xlsx",
        {"mean_time": [{"method": method, "mean_time_per_case_s": vals[i]} for i, method in enumerate(methods)]},
    )
    plt.close(fig)


def plot_accuracy_vs_time(all_metrics: Dict[str, Dict[str, Any]], fig_dir: Path) -> None:
    rows = build_tradeoff_rows(all_metrics)
    fig, ax = create_subplots(kind="single", style=DEFAULT_FIG_STYLE)
    for row in rows:
        x = row["mean_time_per_case_s"]
        y = row["mean_response_nrmse"]
        if x is None or y is None:
            continue
        ax.scatter([x], [y], s=64, label=row["method"])
        ax.annotate(row["method"], (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean time per case (s)")
    ax.set_ylabel("Mean response NRMSE")
    ax.set_title("Accuracy-efficiency trade-off")
    if any((row["mean_time_per_case_s"] or 0) > 0 for row in rows):
        ax.set_xscale("log")
    finalize_axes(ax)
    save_figure(fig, fig_dir, "fig_4_4_accuracy_vs_time", style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(fig_dir / "fig_4_4_accuracy_vs_time.xlsx", {"accuracy_vs_time": rows})
    plt.close(fig)


def build_observation_from_exact(
    y_exact: np.ndarray,
    s: np.ndarray,
    output_vars: Sequence[str],
    observation_vars: Sequence[str],
    sensor_indices: np.ndarray,
) -> np.ndarray:
    return extract_observations_from_fields(
        y=y_exact[None, :, :],
        s=s,
        output_vars=output_vars,
        observation_vars=observation_vars,
        sensor_indices=sensor_indices,
    )[0].astype(np.float32)


def predict_bmn_single_case(
    encoder: torch.nn.Module,
    encoder_ckpt: Dict[str, Any],
    decoder_adapter: FrozenDecoderAdapter,
    observation: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    device: str,
) -> np.ndarray:
    obs_scaler = StandardScaler.from_dict(encoder_ckpt["obs_scaler"])
    mu_scaler = StandardScaler.from_dict(encoder_ckpt["mu_scaler"])
    obs_s = obs_scaler.transform(observation[None, :])
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu_pred_s = encoder(torch.tensor(obs_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
        mu_pred = mu_scaler.inverse_transform(mu_pred_s)
        y_pred = decoder_adapter(
            s_t,
            torch.tensor(c[None, :], dtype=torch.float32, device=device),
            torch.tensor(mu_pred, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()[0]
    return y_pred.astype(np.float32)


def predict_decoderopt_single_case(
    cfg: BMNConfig,
    decoder_ckpt_path: Path,
    bmn_ckpt_path: Path,
    observation: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    output_vars: Sequence[str],
    observation_vars: Sequence[str],
    sensor_indices: np.ndarray,
    n_steps: int,
    lr: float,
    num_restarts: int,
    lambda_order: float,
    lbfgs_steps: int,
    device: str,
) -> np.ndarray:
    obs_scaler = load_obs_scaler_from_encoder(bmn_ckpt_path)
    decoder_adapter = FrozenDecoderAdapter(decoder_ckpt_path, device=device)
    mu_lower, mu_upper = get_mu_bounds(cfg)
    result = optimize_single_case(
        decoder_adapter=decoder_adapter,
        s_t=torch.tensor(s, dtype=torch.float32, device=device),
        c_one_t=torch.tensor(c[None, :], dtype=torch.float32, device=device),
        obs_true_one_t=torch.tensor(observation[None, :], dtype=torch.float32, device=device),
        output_vars=list(output_vars),
        observation_vars=list(observation_vars),
        sensor_indices=np.asarray(sensor_indices, dtype=np.int64),
        obs_mean_t=torch.tensor(obs_scaler.mean, dtype=torch.float32, device=device),
        obs_std_t=torch.tensor(obs_scaler.std, dtype=torch.float32, device=device),
        mu_lower_t=torch.tensor(mu_lower, dtype=torch.float32, device=device),
        mu_upper_t=torch.tensor(mu_upper, dtype=torch.float32, device=device),
        n_steps=n_steps,
        lr=lr,
        num_restarts=num_restarts,
        lambda_order=lambda_order,
        lbfgs_steps=lbfgs_steps,
        seed=int(cfg.encoder_training.seed),
        device=device,
    )
    return result["y_pred"].astype(np.float32)


def plot_typical_case_collection(
    cases: Sequence[Dict[str, Any]],
    case_results: Dict[str, Dict[str, Any]],
    output_vars: Sequence[str],
    fig_dir: Path,
    sensor_indices: np.ndarray | None = None,
) -> None:
    fig, axes = create_subplots(2, 2, kind="quad", style=DEFAULT_FIG_STYLE)
    idx = {name: i for i, name in enumerate(output_vars)}
    ax1, ax2, ax3, ax4 = axes.flat
    case_cmap = plt.get_cmap("tab10")

    sensor_indices_arr = None if sensor_indices is None else np.asarray(sensor_indices, dtype=int)

    for row, case in enumerate(cases):
        result = case_results[case["case_id"]]
        s = result["s"]
        y_series = result["y_series"]
        y_exact = result["y_exact"]
        case_color = case_cmap(row % 10)

        for method, y in y_series.items():
            style = METHOD_STYLES[method]
            label = method if row == 0 else "_nolegend_"
            ax1.plot(y[:, idx["x"]], y[:, idx["z"]], label=label, color=style["color"], linestyle=style["linestyle"])
            ax2.plot(s, y[:, idx["theta"]], label=label, color=style["color"], linestyle=style["linestyle"])
            ax3.plot(s, y[:, idx["T"]], label=label, color=style["color"], linestyle=style["linestyle"])
            ax4.plot(s, y[:, idx["M"]], label=label, color=style["color"], linestyle=style["linestyle"])

        if sensor_indices_arr is not None:
            sensor_indices_safe = sensor_indices_arr[(sensor_indices_arr >= 0) & (sensor_indices_arr < len(s))]
            if sensor_indices_safe.size > 0:
                ax1.scatter(y_exact[sensor_indices_safe, idx["x"]], y_exact[sensor_indices_safe, idx["z"]], color=[case_color], s=12, zorder=5)
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
    fig.suptitle("Typical-case response comparison across methods", y=0.995)
    save_figure(fig, fig_dir, "fig_4_4_typical_case_comparison_all", style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        fig_dir / "fig_4_4_typical_case_comparison_all.xlsx",
        {
            case["case_id"]: _method_response_rows(
                case_results[case["case_id"]]["s"],
                case_results[case["case_id"]]["y_exact"],
                case_results[case["case_id"]]["y_series"],
                idx,
            )
            for case in cases
            if case["case_id"] in case_results
        },
    )
    plt.close(fig)


def plot_typical_case_comparison(
    case_id: str,
    desc: str,
    s: np.ndarray,
    y_exact: np.ndarray,
    y_series: Dict[str, np.ndarray],
    output_vars: Sequence[str],
    fig_dir: Path,
    sensor_indices: np.ndarray | None = None,
) -> None:
    idx = {name: i for i, name in enumerate(output_vars)}
    fig, axes = create_subplots(2, 2, kind="quad", style=DEFAULT_FIG_STYLE)
    ax1, ax2, ax3, ax4 = axes.flat

    for method, y in y_series.items():
        style = METHOD_STYLES[method]
        ax1.plot(y[:, idx["x"]], y[:, idx["z"]], label=method, color=style["color"], linestyle=style["linestyle"])
        ax2.plot(s, y[:, idx["theta"]], label=method, color=style["color"], linestyle=style["linestyle"])
        ax3.plot(s, y[:, idx["T"]], label=method, color=style["color"], linestyle=style["linestyle"])
        ax4.plot(s, y[:, idx["M"]], label=method, color=style["color"], linestyle=style["linestyle"])

    if sensor_indices is not None:
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
    save_figure(fig, fig_dir, f"fig_4_4_{case_id}_comparison", style=DEFAULT_FIG_STYLE, export=DEFAULT_EXPORT_CONFIG)
    save_excel_workbook(
        fig_dir / f"fig_4_4_{case_id}_comparison.xlsx",
        {case_id: _method_response_rows(s, y_exact, y_series, idx)},
    )
    plt.close(fig)


def _method_response_rows(
    s: np.ndarray,
    y_exact: np.ndarray,
    y_series: Dict[str, np.ndarray],
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
    for method, y in y_series.items():
        clean = str(method).replace(" ", "_")
        columns[f"x_{clean}_m"] = y[:, idx["x"]]
        columns[f"z_{clean}_m"] = y[:, idx["z"]]
        columns[f"theta_{clean}_rad"] = y[:, idx["theta"]]
        columns[f"T_{clean}_N"] = y[:, idx["T"]]
        columns[f"M_{clean}_N_m"] = y[:, idx["M"]]
    return columns_to_rows(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Section 4.4 model comparison on a common evaluation set.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--bmn_ckpt", type=str, default=str(DEFAULT_BMN_CKPT))
    parser.add_argument("--decoder_ckpt", type=str, default=str(DEFAULT_DECODER_CKPT))
    parser.add_argument("--direct_ckpt", type=str, default=str(DEFAULT_DIRECT_CKPT))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--eval_dataset", type=str, default=str(DEFAULT_EVAL_DATASET), help="Shared exact-solver evaluation dataset. Override when needed.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--decoderopt_steps", type=int, default=300)
    parser.add_argument("--decoderopt_lr", type=float, default=5.0e-2)
    parser.add_argument("--decoderopt_restarts", type=int, default=1)
    parser.add_argument("--decoderopt_lambda_order", type=float, default=1.0)
    parser.add_argument("--decoderopt_lbfgs_steps", type=int, default=0)
    parser.add_argument("--decoderopt_predictions", type=str, default=None, help="Optional existing DecoderOptimization_predictions.npz to reuse.")
    args = parser.parse_args()

    raw_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    bmn_ckpt_path = require_existing_file(args.bmn_ckpt, "BMN checkpoint")
    decoder_ckpt_path = require_existing_file(args.decoder_ckpt, "Decoder checkpoint")
    direct_ckpt_path = require_existing_file(args.direct_ckpt, "DirectMapping checkpoint")
    output_dir = ensure_dir(args.output_dir)
    data_dir = ensure_dir(output_dir / "data")
    table_dir = ensure_dir(output_dir / "tables")
    fig_dir = ensure_dir(output_dir / "figures")
    device = resolve_device(args.device)
    apply_paper_style(DEFAULT_FIG_STYLE)

    _ = load_encoder_model(bmn_ckpt_path, map_location="cpu")
    dataset_path = require_existing_file(args.eval_dataset, "Evaluation dataset")
    dataset = load_npz(dataset_path)
    case_indices = np.arange(int(dataset["y"].shape[0]), dtype=np.int64)
    eval_label = str(dataset_path)

    output_vars = [str(v) for v in dataset["output_vars"].tolist()]

    run_info = {
        "config_path": str(Path(args.config).resolve()),
        "bmn_ckpt": str(bmn_ckpt_path),
        "decoder_ckpt": str(decoder_ckpt_path),
        "direct_ckpt": str(direct_ckpt_path),
        "eval_dataset": eval_label,
        "n_cases": int(len(case_indices)),
        "device": device,
    }
    save_json(run_info, data_dir / "run_info_4_4.json")

    all_metrics: Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, Dict[str, Any]] = {}

    bmn_metrics, bmn_predictions = evaluate_bmn_method(bmn_ckpt_path, decoder_ckpt_path, dataset, case_indices, device)
    all_metrics["BMN"] = bmn_metrics
    all_predictions["BMN"] = bmn_predictions

    direct_metrics, direct_predictions = evaluate_direct_method(direct_ckpt_path, dataset, case_indices, device)
    all_metrics["DirectMapping"] = direct_metrics
    all_predictions["DirectMapping"] = direct_predictions

    decoderopt_block = raw_cfg.get("decoder_optimization", {}) if isinstance(raw_cfg.get("decoder_optimization"), dict) else {}
    default_decoderopt_predictions = Path(decoderopt_block.get("output_dir", Path(cfg.dataset.output_dir) / "DecoderOptimization_baseline")) / "DecoderOptimization_predictions.npz"
    decoderopt_predictions_path = Path(args.decoderopt_predictions) if args.decoderopt_predictions is not None else default_decoderopt_predictions
    decoderopt_loaded = load_decoderopt_predictions_if_compatible(
        predictions_path=decoderopt_predictions_path,
        dataset=dataset,
        case_indices=case_indices,
        n_steps=args.decoderopt_steps,
        lr=args.decoderopt_lr,
        num_restarts=args.decoderopt_restarts,
        lambda_order=args.decoderopt_lambda_order,
        lbfgs_steps=args.decoderopt_lbfgs_steps,
    )
    if decoderopt_loaded is not None:
        decoderopt_metrics, decoderopt_predictions = decoderopt_loaded
    else:
        decoderopt_metrics, decoderopt_predictions = evaluate_decoderopt_method(
            cfg=cfg,
            decoder_ckpt_path=decoder_ckpt_path,
            bmn_ckpt_path=bmn_ckpt_path,
            dataset=dataset,
            case_indices=case_indices,
            n_steps=args.decoderopt_steps,
            lr=args.decoderopt_lr,
            num_restarts=args.decoderopt_restarts,
            lambda_order=args.decoderopt_lambda_order,
            lbfgs_steps=args.decoderopt_lbfgs_steps,
            device=device,
        )
    all_metrics["DecoderOptimization"] = decoderopt_metrics
    all_predictions["DecoderOptimization"] = decoderopt_predictions

    save_json(all_metrics, data_dir / "model_comparison_metrics.json")
    save_csv(build_parameter_rows(all_metrics), table_dir / "table_4_4_parameter_metrics.csv")
    save_csv(build_response_rows(all_metrics), table_dir / "table_4_4_response_metrics.csv")
    save_csv(build_feature_rows(all_metrics), table_dir / "table_4_4_feature_metrics.csv")
    save_csv(build_tradeoff_rows(all_metrics), table_dir / "table_4_4_tradeoff_summary.csv")

    np.savez_compressed(
        data_dir / "model_comparison_predictions.npz",
        s=np.asarray(dataset["s"], dtype=np.float32),
        case_indices=case_indices,
        output_vars=np.asarray(output_vars),
        mu_true=all_predictions["BMN"]["mu_true"],
        bmn_mu_pred=all_predictions["BMN"]["mu_pred"],
        decoderopt_mu_pred=all_predictions["DecoderOptimization"]["mu_pred"],
        y_true=all_predictions["BMN"]["y_true"],
        bmn_y_pred=all_predictions["BMN"]["y_pred"],
        direct_y_pred=all_predictions["DirectMapping"]["y_pred"],
        decoderopt_y_pred=all_predictions["DecoderOptimization"]["y_pred"],
    )

    plot_parameter_nrmse(all_metrics, fig_dir)
    plot_response_nrmse(all_metrics, fig_dir)
    plot_mean_time_per_case(all_metrics, fig_dir)
    plot_accuracy_vs_time(all_metrics, fig_dir)

    summary = {
        "status": "completed",
        "output_dir": str(output_dir),
        "eval_dataset": eval_label,
        "methods": list(all_metrics.keys()),
    }
    save_json(summary, output_dir / "analysis_4_4_summary.json")

    encoder_model, encoder_ckpt = load_encoder_model(bmn_ckpt_path, map_location=device)
    encoder_model = encoder_model.to(device)
    encoder_model.eval()
    direct_model, direct_ckpt = load_direct_model(direct_ckpt_path, map_location=device)
    bmn_sensor_indices = np.asarray(encoder_ckpt["sensor_indices"], dtype=np.int64)
    bmn_observation_vars = [str(v) for v in encoder_ckpt["observation_vars"]]
    decoder_adapter = FrozenDecoderAdapter(decoder_ckpt_path, device=device)

    case_results: Dict[str, Dict[str, Any]] = {}
    for case in TYPICAL_CASES:
        exact = compute_exact_case(case, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            continue
        s = np.asarray(exact["s"], dtype=np.float32)
        y_exact = np.stack([np.asarray(exact[name], dtype=np.float32) for name in output_vars], axis=-1)
        c = np.asarray([case["Dx"], case["ht"]], dtype=np.float32)
        observation = build_observation_from_exact(
            y_exact=y_exact,
            s=s,
            output_vars=output_vars,
            observation_vars=bmn_observation_vars,
            sensor_indices=bmn_sensor_indices,
        )
        y_bmn = predict_bmn_single_case(
            encoder=encoder_model,
            encoder_ckpt=encoder_ckpt,
            decoder_adapter=decoder_adapter,
            observation=observation,
            c=c,
            s=s,
            device=device,
        )
        y_direct = predict_direct_fullfield_np(
            model=direct_model,
            checkpoint=direct_ckpt,
            s=s,
            c=c,
            observation=observation,
            device=device,
        ).astype(np.float32)
        y_decoderopt = predict_decoderopt_single_case(
            cfg=cfg,
            decoder_ckpt_path=decoder_ckpt_path,
            bmn_ckpt_path=bmn_ckpt_path,
            observation=observation,
            c=c,
            s=s,
            output_vars=output_vars,
            observation_vars=bmn_observation_vars,
            sensor_indices=bmn_sensor_indices,
            n_steps=args.decoderopt_steps,
            lr=args.decoderopt_lr,
            num_restarts=args.decoderopt_restarts,
            lambda_order=args.decoderopt_lambda_order,
            lbfgs_steps=args.decoderopt_lbfgs_steps,
            device=device,
        )
        case_results[case["case_id"]] = {
            "s": s,
            "y_exact": y_exact,
            "y_series": {
                "Exact": y_exact,
                "BMN": y_bmn,
                "DirectMapping": y_direct,
                "DecoderOptimization": y_decoderopt,
            },
        }
        plot_typical_case_comparison(
            case_id=case["case_id"],
            desc=case["description"],
            s=s,
            y_exact=y_exact,
            y_series=case_results[case["case_id"]]["y_series"],
            output_vars=output_vars,
            fig_dir=fig_dir,
            sensor_indices=bmn_sensor_indices,
        )
    if case_results:
        plot_typical_case_collection(
            cases=TYPICAL_CASES,
            case_results=case_results,
            output_vars=output_vars,
            fig_dir=fig_dir,
            sensor_indices=bmn_sensor_indices,
        )


if __name__ == "__main__":
    main()
