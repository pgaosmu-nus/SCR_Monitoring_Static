#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DecoderOptimizationInversion.py

Section 4.4 baseline 2: Decoder-optimization inversion for sparse-monitoring
SCR static-state reconstruction.

Baseline definition
-------------------
This baseline does not train an Encoder. For each test case, it directly solves
an optimization problem in the latent/environmental parameter space:

    mu_hat = arg min_mu || S[Decoder(c, mu)] - observation ||^2,

where
    c  = [Dx, ht],
    mu = [Us, Ub, p],
    S[.] is the sparse observation extraction operator.

The trained Decoder is frozen. Gradients are propagated through the frozen
Decoder to optimize mu for each test sample.

Recommended workflow
--------------------
1. Train the original Decoder first:
       python Decoder_DD.py --config para_config.json --mode all

2. Run optimization inversion on the default dataset/split:
       python DecoderOptimizationInversion.py --config para_config.json --mode evaluate

3. Run on a fixed common exact-solver test set:
       python DecoderOptimizationInversion.py --config para_config.json \
           --eval_dataset paper_testset_4_4_exact.npz \
           --decoder_ckpt outputs/BMN_SCR_DD_outputs/Decoder_DD_model.pth

Outputs
-------
- DecoderOptimization_predictions.npz
- DecoderOptimization_metrics.json
- DecoderOptimization_parameter_metrics.csv
- DecoderOptimization_response_metrics.csv
- DecoderOptimization_feature_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim

from Decoder_DD import (
    BMNConfig,
    MU_NAMES,
    PARAM_NAMES,
    StandardScaler,
    compute_exact_case,
    extract_observations_from_fields,
    load_config,
    resolve_sensor_indices,
    sample_one_case,
)
from BMN_DD import FrozenDecoderAdapter, extract_observation_torch


# =============================================================================
# 0. USER-EDITABLE DEFAULTS
# =============================================================================

DEFAULT_CONFIG_PATH = "para_config.json"
DEFAULT_OUTPUT_SUBDIR = "DecoderOptimization_baseline"
RANDOM_TEST_N_CASES = 500
RANDOM_TEST_SEED = 20260503


# =============================================================================
# 1. Utilities
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_json_dict(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def get_mu_bounds(cfg: BMNConfig) -> Tuple[np.ndarray, np.ndarray]:
    lower = np.asarray([cfg.ranges.Us_min, cfg.ranges.Ub_min, cfg.ranges.p_min], dtype=np.float32)
    upper = np.asarray([cfg.ranges.Us_max, cfg.ranges.Ub_max, cfg.ranges.p_max], dtype=np.float32)
    return lower, upper


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


def raw_to_mu(raw: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Bounded parameterization: mu = lower + (upper-lower) sigmoid(raw)."""
    return lower[None, :] + (upper - lower)[None, :] * torch.sigmoid(raw)


def make_obs_names(observation_vars: Sequence[str], sensor_indices: np.ndarray) -> List[str]:
    obs_names = ["x_top", "z_top"]
    for idx in sensor_indices:
        for var in observation_vars:
            obs_names.append(f"{var}_idx{int(idx)}")
    return obs_names


def load_obs_scaler_from_encoder(encoder_ckpt_path: Optional[str | Path]) -> Optional[StandardScaler]:
    if encoder_ckpt_path is None:
        raise ValueError(
            "DecoderOptimizationInversion requires --encoder_ckpt_for_obs_scaler. "
            "Pass a BMN encoder checkpoint so observation normalization matches the BMN comparison baseline."
        )
    ckpt_path = Path(encoder_ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint for obs_scaler not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "obs_scaler" not in ckpt:
        raise KeyError(f"No obs_scaler found in encoder checkpoint: {ckpt_path}")
    return StandardScaler.from_dict(ckpt["obs_scaler"])


# =============================================================================
# 2. Core optimization
# =============================================================================


def make_initial_raw(num_restarts: int, device: str, seed: int, random_scale: float = 2.0) -> torch.Tensor:
    """Create multi-start raw variables.

    Restart 0 is the parameter-domain midpoint. Additional restarts are random.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    raw = torch.zeros((num_restarts, 3), dtype=torch.float32, device=device)
    if num_restarts > 1:
        raw[1:] = random_scale * torch.randn((num_restarts - 1, 3), dtype=torch.float32, device=device, generator=gen)
    return raw


def optimize_single_case(
    decoder_adapter: FrozenDecoderAdapter,
    s_t: torch.Tensor,
    c_one_t: torch.Tensor,
    obs_true_one_t: torch.Tensor,
    output_vars: List[str],
    observation_vars: List[str],
    sensor_indices: np.ndarray,
    obs_mean_t: torch.Tensor,
    obs_std_t: torch.Tensor,
    mu_lower_t: torch.Tensor,
    mu_upper_t: torch.Tensor,
    n_steps: int = 300,
    lr: float = 5.0e-2,
    num_restarts: int = 1,
    lambda_order: float = 1.0,
    lbfgs_steps: int = 0,
    seed: int = 42,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Optimize mu for one case using multi-start Adam and optional L-BFGS."""
    raw = make_initial_raw(num_restarts, device=device, seed=seed)
    raw.requires_grad_(True)

    c_batch = c_one_t.repeat(num_restarts, 1)
    obs_true_batch = obs_true_one_t.repeat(num_restarts, 1)
    obs_true_s = (obs_true_batch - obs_mean_t[None, :]) / obs_std_t[None, :]

    def loss_by_restart(current_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = raw_to_mu(current_raw, mu_lower_t, mu_upper_t)
        y_pred = decoder_adapter(s_t, c_batch, mu)
        obs_pred = extract_observation_torch(y_pred, output_vars, observation_vars, sensor_indices)
        obs_pred_s = (obs_pred - obs_mean_t[None, :]) / obs_std_t[None, :]
        loss_obs_each = torch.mean((obs_pred_s - obs_true_s) ** 2, dim=1)
        order_violation = torch.relu(mu[:, 1] - mu[:, 0])  # enforce Ub <= Us softly
        loss_order_each = order_violation ** 2
        loss_each = loss_obs_each + float(lambda_order) * loss_order_each
        return loss_each, mu, y_pred, obs_pred

    optimizer = optim.Adam([raw], lr=float(lr))
    history: List[float] = []
    best_loss = float("inf")
    best_raw = raw.detach().clone()

    for step in range(1, int(n_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_each, _, _, _ = loss_by_restart(raw)
        loss = torch.mean(loss_each)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            min_loss = float(torch.min(loss_each).detach().cpu())
            history.append(min_loss)
            if min_loss < best_loss:
                best_loss = min_loss
                best_raw = raw.detach().clone()

    if int(lbfgs_steps) > 0:
        with torch.no_grad():
            raw.copy_(best_raw)
        lbfgs = optim.LBFGS([raw], lr=1.0, max_iter=int(lbfgs_steps), line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss_each, _, _, _ = loss_by_restart(raw)
            loss = torch.mean(loss_each)
            loss.backward()
            return loss

        lbfgs.step(closure)
        with torch.no_grad():
            loss_each, _, _, _ = loss_by_restart(raw)
            min_loss = float(torch.min(loss_each).detach().cpu())
            history.append(min_loss)
            if min_loss < best_loss:
                best_loss = min_loss
                best_raw = raw.detach().clone()

    with torch.no_grad():
        loss_each, mu_all, y_all, obs_all = loss_by_restart(best_raw)
        best_idx = int(torch.argmin(loss_each).detach().cpu())
        mu_best = mu_all[best_idx : best_idx + 1]
        y_best = y_all[best_idx : best_idx + 1]
        obs_best = obs_all[best_idx : best_idx + 1]
        best_loss_final = float(loss_each[best_idx].detach().cpu())

    return {
        "mu_pred": mu_best.detach().cpu().numpy()[0],
        "y_pred": y_best.detach().cpu().numpy()[0],
        "obs_pred": obs_best.detach().cpu().numpy()[0],
        "best_loss": best_loss_final,
        "loss_history": np.asarray(history, dtype=np.float32),
    }


# =============================================================================
# 3. Metrics
# =============================================================================


def compute_inversion_metrics(
    s: np.ndarray,
    mu_true: Optional[np.ndarray],
    mu_pred: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    obs_true: np.ndarray,
    obs_pred: np.ndarray,
    output_vars: Sequence[str],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "parameter": {},
        "response": {},
        "observation": {},
        "features": {},
    }

    if mu_true is not None:
        for i, name in enumerate(MU_NAMES):
            diff = mu_pred[:, i] - mu_true[:, i]
            metrics["parameter"][name] = {
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "mae": float(np.mean(np.abs(diff))),
                "mape": float(np.mean(np.abs(diff) / np.maximum(np.abs(mu_true[:, i]), 1.0e-8))),
            }
    else:
        metrics["parameter"] = "not_available_no_mu_true_in_dataset"

    for j, name in enumerate(output_vars):
        diff = y_pred[:, :, j] - y_true[:, :, j]
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        maxae = float(np.max(np.abs(diff)))
        value_range = float(np.max(y_true[:, :, j]) - np.min(y_true[:, :, j]))
        nrmse = float(rmse / value_range) if value_range > 1.0e-12 else None
        metrics["response"][name] = {"rmse": rmse, "mae": mae, "maxae": maxae, "nrmse": nrmse}

    obs_diff = obs_pred - obs_true
    metrics["observation"] = {
        "rmse": float(np.sqrt(np.mean(obs_diff**2))),
        "mae": float(np.mean(np.abs(obs_diff))),
        "maxae": float(np.max(np.abs(obs_diff))),
    }

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


def save_metric_tables(metrics: Dict[str, Any], output_dir: Path) -> None:
    if isinstance(metrics.get("parameter"), dict):
        parameter_rows: List[Dict[str, Any]] = []
        for name, vals in metrics["parameter"].items():
            row = {"parameter": name}
            row.update(vals)
            parameter_rows.append(row)
        save_csv(parameter_rows, output_dir / "DecoderOptimization_parameter_metrics.csv")

    response_rows: List[Dict[str, Any]] = []
    for name, vals in metrics.get("response", {}).items():
        row = {"response": name}
        row.update(vals)
        response_rows.append(row)
    save_csv(response_rows, output_dir / "DecoderOptimization_response_metrics.csv")

    observation_rows = [{"metric": k, "value": v} for k, v in metrics.get("observation", {}).items()]
    save_csv(observation_rows, output_dir / "DecoderOptimization_observation_metrics.csv")

    feature_rows = [{"metric": k, "value": v} for k, v in metrics.get("features", {}).items()]
    save_csv(feature_rows, output_dir / "DecoderOptimization_feature_metrics.csv")


# =============================================================================
# 5. Evaluation workflow
# =============================================================================


def prepare_eval_dataset(
    cfg: BMNConfig,
    dataset_path: str | Path,
    eval_dataset_path: Optional[str | Path],
    split_seed: int,
) -> Tuple[Dict[str, Any], np.ndarray, str]:
    """Return dataset and case indices to evaluate.

    If eval_dataset_path is provided, all cases in that dataset are used.
    Otherwise, use the same random exact-solver test-set definition as analysis_4_2.
    """
    if eval_dataset_path is not None:
        dataset = load_npz(eval_dataset_path)
        indices = np.arange(int(dataset["y"].shape[0]), dtype=np.int64)
        label = str(eval_dataset_path)
    else:
        dataset = random_cases_to_dataset(cfg, build_random_exact_test_set(cfg, RANDOM_TEST_N_CASES, RANDOM_TEST_SEED))
        indices = np.arange(int(dataset["y"].shape[0]), dtype=np.int64)
        label = f"random exact-solver test set (same as 4.2): n={RANDOM_TEST_N_CASES}, seed={RANDOM_TEST_SEED}"
    return dataset, indices, label


def evaluate_decoder_optimization(
    cfg: BMNConfig,
    decoder_ckpt_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    eval_dataset_path: Optional[str | Path] = None,
    encoder_ckpt_for_obs_scaler: Optional[str | Path] = None,
    n_steps: int = 300,
    lr: float = 5.0e-2,
    num_restarts: int = 1,
    lambda_order: float = 1.0,
    lbfgs_steps: int = 0,
    max_cases: Optional[int] = None,
    split_seed: int = 42,
    device: str = "cpu",
    save_loss_histories: bool = True,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    decoder_ckpt_path = Path(decoder_ckpt_path)
    if not decoder_ckpt_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt_path}")

    dataset, case_indices, eval_label = prepare_eval_dataset(cfg, dataset_path, eval_dataset_path, split_seed=split_seed)
    if max_cases is not None and int(max_cases) > 0:
        case_indices = case_indices[: int(max_cases)]

    s_np = np.asarray(dataset["s"], dtype=np.float32)
    c_all = np.asarray(dataset["c"], dtype=np.float32)
    y_all = np.asarray(dataset["y"], dtype=np.float32)
    mu_all = np.asarray(dataset["mu"], dtype=np.float32) if "mu" in dataset else None
    params_all = np.asarray(dataset["params"], dtype=np.float32) if "params" in dataset else None
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    observation_vars = list(cfg.dataset.observation_vars)
    sensor_indices = resolve_sensor_indices(s_np, cfg.dataset)
    obs_names = make_obs_names(observation_vars, sensor_indices)

    obs_all = extract_observations_from_fields(
        y=y_all,
        s=s_np,
        output_vars=output_vars,
        observation_vars=observation_vars,
        sensor_indices=sensor_indices,
    )

    obs_scaler = load_obs_scaler_from_encoder(encoder_ckpt_for_obs_scaler)

    device = resolve_device(device)
    decoder_adapter = FrozenDecoderAdapter(decoder_ckpt_path, device=device)
    decoder_output_vars = list(decoder_adapter.output_vars)
    if decoder_output_vars != output_vars:
        raise ValueError(
            "Decoder checkpoint output_vars and evaluation dataset output_vars are inconsistent:\n"
            f"  decoder: {decoder_output_vars}\n  dataset: {output_vars}"
        )

    s_t = torch.tensor(s_np, dtype=torch.float32, device=device)
    obs_mean_t = torch.tensor(obs_scaler.mean, dtype=torch.float32, device=device)
    obs_std_t = torch.tensor(obs_scaler.std, dtype=torch.float32, device=device)
    mu_lower, mu_upper = get_mu_bounds(cfg)
    mu_lower_t = torch.tensor(mu_lower, dtype=torch.float32, device=device)
    mu_upper_t = torch.tensor(mu_upper, dtype=torch.float32, device=device)

    n_eval = len(case_indices)
    y_true = y_all[case_indices]
    y_pred = np.zeros_like(y_true)
    obs_true = obs_all[case_indices]
    obs_pred = np.zeros_like(obs_true)
    mu_true = mu_all[case_indices] if mu_all is not None else None
    mu_pred = np.zeros((n_eval, len(MU_NAMES)), dtype=np.float32)
    c_eval = c_all[case_indices]
    params_eval = params_all[case_indices] if params_all is not None else None

    time_per_case: List[float] = []
    best_losses: List[float] = []
    loss_histories: List[np.ndarray] = []

    print("=" * 88)
    print("Evaluating Section 4.4 Decoder-optimization inversion baseline")
    print(f"Decoder ckpt : {decoder_ckpt_path}")
    print(f"Eval set     : {eval_label}")
    print(f"Cases        : {n_eval}")
    print(f"Obs vars     : {observation_vars}")
    print(f"Sensors      : {sensor_indices.tolist()}")
    print(f"n_steps/lr   : {n_steps}/{lr}")
    print(f"restarts     : {num_restarts}")
    print(f"LBFGS steps  : {lbfgs_steps}")
    print(f"Device       : {device}")
    print("=" * 88)

    t_all = time.time()
    for k, case_id in enumerate(case_indices):
        t0 = time.time()
        c_one_t = torch.tensor(c_all[case_id : case_id + 1], dtype=torch.float32, device=device)
        obs_one_t = torch.tensor(obs_all[case_id : case_id + 1], dtype=torch.float32, device=device)

        result = optimize_single_case(
            decoder_adapter=decoder_adapter,
            s_t=s_t,
            c_one_t=c_one_t,
            obs_true_one_t=obs_one_t,
            output_vars=output_vars,
            observation_vars=observation_vars,
            sensor_indices=sensor_indices,
            obs_mean_t=obs_mean_t,
            obs_std_t=obs_std_t,
            mu_lower_t=mu_lower_t,
            mu_upper_t=mu_upper_t,
            n_steps=int(n_steps),
            lr=float(lr),
            num_restarts=int(num_restarts),
            lambda_order=float(lambda_order),
            lbfgs_steps=int(lbfgs_steps),
            seed=int(split_seed) + 7919 * int(k),
            device=device,
        )

        mu_pred[k] = result["mu_pred"]
        y_pred[k] = result["y_pred"]
        obs_pred[k] = result["obs_pred"]
        best_losses.append(float(result["best_loss"]))
        if save_loss_histories:
            loss_histories.append(result["loss_history"])
        time_per_case.append(time.time() - t0)

        if k == 0 or (k + 1) % max(1, n_eval // 10) == 0:
            print(
                f"[DecoderOpt] case={k+1:5d}/{n_eval} | "
                f"best_loss={best_losses[-1]:.3e} | time={time_per_case[-1]:.2f}s"
            )

    metrics = compute_inversion_metrics(
        s=s_np,
        mu_true=mu_true,
        mu_pred=mu_pred,
        y_true=y_true,
        y_pred=y_pred,
        obs_true=obs_true,
        obs_pred=obs_pred,
        output_vars=output_vars,
    )
    metrics["method"] = "DecoderOptimizationInversion"
    metrics["eval_dataset"] = eval_label
    metrics["optimization"] = {
        "n_steps": int(n_steps),
        "lr": float(lr),
        "num_restarts": int(num_restarts),
        "lambda_order": float(lambda_order),
        "lbfgs_steps": int(lbfgs_steps),
        "best_loss_mean": float(np.mean(best_losses)),
        "best_loss_median": float(np.median(best_losses)),
    }
    metrics["timing"] = {
        "n_test": int(n_eval),
        "total_seconds": float(time.time() - t_all),
        "sum_case_seconds": float(np.sum(time_per_case)),
        "mean_seconds_per_case": float(np.mean(time_per_case)),
        "median_seconds_per_case": float(np.median(time_per_case)),
    }
    metrics["sensor_indices"] = sensor_indices.tolist()
    metrics["sensor_s"] = s_np[sensor_indices].tolist()
    metrics["observation_vars"] = observation_vars
    metrics["obs_names"] = obs_names

    if save_loss_histories and loss_histories:
        max_len = max(len(h) for h in loss_histories)
        loss_hist_arr = np.full((len(loss_histories), max_len), np.nan, dtype=np.float32)
        for i, hist in enumerate(loss_histories):
            loss_hist_arr[i, : len(hist)] = hist
    else:
        loss_hist_arr = np.empty((0, 0), dtype=np.float32)

    np.savez_compressed(
        output_dir / "DecoderOptimization_predictions.npz",
        s=s_np,
        case_indices=case_indices,
        c=c_eval,
        params=params_eval,
        mu_true=mu_true,
        mu_pred=mu_pred,
        y_true=y_true,
        y_pred=y_pred,
        obs_true=obs_true,
        obs_pred=obs_pred,
        sensor_indices=sensor_indices,
        sensor_s=s_np[sensor_indices],
        output_vars=np.asarray(output_vars),
        observation_vars=np.asarray(observation_vars),
        obs_names=np.asarray(obs_names),
        time_per_case=np.asarray(time_per_case, dtype=np.float32),
        best_losses=np.asarray(best_losses, dtype=np.float32),
        loss_histories=loss_hist_arr,
        eval_dataset_label=np.asarray(eval_label),
        random_test_n_cases=np.asarray(RANDOM_TEST_N_CASES, dtype=np.int64),
        random_test_seed=np.asarray(RANDOM_TEST_SEED, dtype=np.int64),
        optimization_n_steps=np.asarray(int(n_steps), dtype=np.int64),
        optimization_num_restarts=np.asarray(int(num_restarts), dtype=np.int64),
        optimization_lbfgs_steps=np.asarray(int(lbfgs_steps), dtype=np.int64),
        optimization_lr=np.asarray(float(lr), dtype=np.float32),
        optimization_lambda_order=np.asarray(float(lambda_order), dtype=np.float32),
    )
    save_json(metrics, output_dir / "DecoderOptimization_metrics.json")
    save_metric_tables(metrics, output_dir)
    print("=" * 88)
    print(f"Decoder-optimization inversion saved to: {output_dir}")
    print("=" * 88)
    return metrics


# =============================================================================
# 6. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoder optimization inversion baseline for Section 4.4.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to para_config.json.")
    parser.add_argument("--dataset", type=str, default=None, help="Default full-field dataset. Used when --eval_dataset is omitted.")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Optional fixed common exact-solver testset.")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Trained Decoder checkpoint path.")
    parser.add_argument("--encoder_ckpt_for_obs_scaler", type=str, default=None, help="Optional BMN encoder checkpoint; uses its obs_scaler for normalized observation loss.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--mode", type=str, default="evaluate", choices=["evaluate"], help="Only evaluate mode is needed for this optimization baseline.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="cpu or cuda.")
    parser.add_argument("--n_steps", type=int, default=300, help="Adam optimization steps per case.")
    parser.add_argument("--lr", type=float, default=5.0e-2, help="Adam learning rate for raw mu variables.")
    parser.add_argument("--num_restarts", type=int, default=1, help="Number of multi-start initializations.")
    parser.add_argument("--lambda_order", type=float, default=1.0, help="Soft penalty for Ub > Us.")
    parser.add_argument("--lbfgs_steps", type=int, default=0, help="Optional L-BFGS max_iter after Adam. 0 disables it.")
    parser.add_argument("--max_cases", type=int, default=None, help="Optional limit for quick debugging.")
    parser.add_argument("--split_seed", type=int, default=42, help="Split seed when --eval_dataset is omitted.")
    parser.add_argument("--no_loss_histories", action="store_true", help="Do not save per-case optimization loss histories.")
    args = parser.parse_args()

    raw_cfg = load_json_dict(args.config)
    cfg = load_config(args.config)
    opt_block = raw_cfg.get("decoder_optimization", {}) if isinstance(raw_cfg.get("decoder_optimization"), dict) else {}

    dataset_override = opt_block.get("dataset_path")
    decoder_ckpt_override = opt_block.get("decoder_ckpt")
    encoder_ckpt_override = opt_block.get("encoder_ckpt_for_obs_scaler")
    output_dir_override = opt_block.get("output_dir")

    dataset_path = (
        Path(args.dataset)
        if args.dataset is not None
        else Path(dataset_override)
        if dataset_override is not None
        else Path(cfg.dataset.output_dir) / cfg.dataset.full_dataset_filename
    )
    decoder_ckpt = (
        Path(args.decoder_ckpt)
        if args.decoder_ckpt is not None
        else Path(decoder_ckpt_override)
        if decoder_ckpt_override is not None
        else Path(cfg.dataset.output_dir) / cfg.decoder_training.model_filename
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(output_dir_override)
        if output_dir_override is not None
        else Path(cfg.dataset.output_dir) / DEFAULT_OUTPUT_SUBDIR
    )
    encoder_ckpt_for_obs_scaler = args.encoder_ckpt_for_obs_scaler or encoder_ckpt_override
    device = str(opt_block.get("device", args.device))
    n_steps = int(opt_block.get("n_steps", args.n_steps))
    lr = float(opt_block.get("lr", args.lr))
    num_restarts = int(opt_block.get("num_restarts", args.num_restarts))
    lambda_order = float(opt_block.get("lambda_order", args.lambda_order))
    lbfgs_steps = int(opt_block.get("lbfgs_steps", args.lbfgs_steps))
    max_cases = int(opt_block["max_cases"]) if opt_block.get("max_cases") is not None else args.max_cases
    split_seed = int(opt_block.get("split_seed", args.split_seed))
    save_loss_histories = bool(opt_block.get("save_loss_histories", not args.no_loss_histories))

    if args.device != parser.get_default("device"):
        device = args.device
    if args.n_steps != parser.get_default("n_steps"):
        n_steps = args.n_steps
    if args.lr != parser.get_default("lr"):
        lr = args.lr
    if args.num_restarts != parser.get_default("num_restarts"):
        num_restarts = args.num_restarts
    if args.lambda_order != parser.get_default("lambda_order"):
        lambda_order = args.lambda_order
    if args.lbfgs_steps != parser.get_default("lbfgs_steps"):
        lbfgs_steps = args.lbfgs_steps
    if args.max_cases is not None:
        max_cases = args.max_cases
    if args.split_seed != parser.get_default("split_seed"):
        split_seed = args.split_seed
    if args.no_loss_histories:
        save_loss_histories = False

    evaluate_decoder_optimization(
        cfg=cfg,
        decoder_ckpt_path=decoder_ckpt,
        dataset_path=dataset_path,
        output_dir=output_dir,
        eval_dataset_path=args.eval_dataset,
        encoder_ckpt_for_obs_scaler=encoder_ckpt_for_obs_scaler,
        n_steps=n_steps,
        lr=lr,
        num_restarts=num_restarts,
        lambda_order=lambda_order,
        lbfgs_steps=lbfgs_steps,
        max_cases=max_cases,
        split_seed=split_seed,
        device=device,
        save_loss_histories=save_loss_histories,
    )


if __name__ == "__main__":
    main()
