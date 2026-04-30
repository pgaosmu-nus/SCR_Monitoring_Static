#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the full BMN inverse pipeline on randomly sampled exact-solver cases.

This script:
1. samples random SCR parameter cases within the configured ranges
2. solves each case with the exact solver to obtain full-field responses
3. extracts sparse observations consistent with the trained Encoder
4. runs BMN inverse inference: observation -> Encoder -> mu_hat -> Decoder -> y_hat
5. plots BMN-vs-Exact comparisons for each case
6. writes a JSON summary with parameter and response errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from BMN_DD import load_encoder_model
from Decoder_DD import (
    BMNConfig,
    PARAM_NAMES,
    StandardScaler,
    compute_exact_case,
    config_from_dict,
    decode_fullfield_np,
    load_config,
    load_decoder_model,
    sample_one_case,
)


DEFAULT_OUTPUT_DIRNAME = "bmn_random_exact_comparison"
REQUIRED_VARS = ["x", "z", "theta", "T", "M"]


def exact_case_to_output_dict(exact: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "s": np.asarray(exact["s"], dtype=np.float32),
        "x": np.asarray(exact["x"], dtype=np.float32),
        "z": np.asarray(exact["z"], dtype=np.float32),
        "theta": np.asarray(exact["theta"], dtype=np.float32),
        "T": np.asarray(exact["T"], dtype=np.float32),
        "M": np.asarray(exact["M"], dtype=np.float32),
    }


def prediction_array_to_output_dict(
    s: np.ndarray,
    y: np.ndarray,
    output_vars: List[str],
) -> Dict[str, np.ndarray]:
    var_to_col = {name: i for i, name in enumerate(output_vars)}
    missing = [name for name in REQUIRED_VARS if name not in var_to_col]
    if missing:
        raise ValueError(f"Decoder output vars are missing required fields: {missing}")
    return {
        "s": np.asarray(s, dtype=np.float32),
        "x": np.asarray(y[:, var_to_col["x"]], dtype=np.float32),
        "z": np.asarray(y[:, var_to_col["z"]], dtype=np.float32),
        "theta": np.asarray(y[:, var_to_col["theta"]], dtype=np.float32),
        "T": np.asarray(y[:, var_to_col["T"]], dtype=np.float32),
        "M": np.asarray(y[:, var_to_col["M"]], dtype=np.float32),
    }


def extract_observation_from_exact(
    exact: Dict[str, np.ndarray],
    observation_vars: List[str],
    sensor_indices: np.ndarray,
) -> np.ndarray:
    obs_terms: List[float] = [float(exact["x"][-1]), float(exact["z"][-1])]
    for idx in sensor_indices:
        i = int(idx)
        for var in observation_vars:
            obs_terms.append(float(exact[var][i]))
    return np.asarray(obs_terms, dtype=np.float32)


def interpolate_exact_to_prediction_grid(pred_s: np.ndarray, exact: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "x": np.interp(pred_s, exact["s"], exact["x"]),
        "z": np.interp(pred_s, exact["s"], exact["z"]),
        "theta": np.interp(pred_s, exact["s"], exact["theta"]),
        "T": np.interp(pred_s, exact["s"], exact["T"]),
        "M": np.interp(pred_s, exact["s"], exact["M"]),
    }


def compute_response_metrics(pred: Dict[str, np.ndarray], exact: Dict[str, np.ndarray]) -> Dict[str, float]:
    exact_i = interpolate_exact_to_prediction_grid(pred["s"], exact)
    metrics: Dict[str, float] = {}
    for key in REQUIRED_VARS:
        diff = pred[key] - exact_i[key]
        metrics[f"rmse_{key}"] = float(np.sqrt(np.mean(diff ** 2)))
        metrics[f"mae_{key}"] = float(np.mean(np.abs(diff)))
    return metrics


def compute_parameter_metrics(mu_pred: np.ndarray, mu_true: np.ndarray, mu_names: List[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for i, name in enumerate(mu_names):
        diff = float(mu_pred[i] - mu_true[i])
        metrics[f"err_{name}"] = diff
        metrics[f"abs_err_{name}"] = abs(diff)
    return metrics


def plot_case_comparison(
    pred: Dict[str, np.ndarray],
    exact: Dict[str, np.ndarray],
    case_params: Dict[str, float],
    mu_pred: np.ndarray,
    mu_names: List[str],
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(pred["x"], pred["z"], lw=2.0, label="BMN")
    ax1.plot(exact["x"], exact["z"], "--", lw=2.0, label="Exact")
    ax1.set_title("x-z geometry")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("z [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pred["s"], pred["theta"], lw=2.0, label="BMN")
    ax2.plot(exact["s"], exact["theta"], "--", lw=2.0, label="Exact")
    ax2.set_title("theta(s)")
    ax2.set_xlabel("s [m]")
    ax2.set_ylabel("theta [rad]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pred["s"], pred["T"] / 1.0e3, lw=2.0, label="BMN")
    ax3.plot(exact["s"], exact["T"] / 1.0e3, "--", lw=2.0, label="Exact")
    ax3.set_title("T(s)")
    ax3.set_xlabel("s [m]")
    ax3.set_ylabel("T [kN]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(pred["s"], pred["M"] / 1.0e3, lw=2.0, label="BMN")
    ax4.plot(exact["s"], exact["M"] / 1.0e3, "--", lw=2.0, label="Exact")
    ax4.set_title("M(s)")
    ax4.set_xlabel("s [m]")
    ax4.set_ylabel("M [kN m]")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    mu_true_str = ", ".join(f"{name}={case_params[name]:.3f}" for name in mu_names)
    mu_pred_str = ", ".join(f"{name}_hat={float(mu_pred[i]):.3f}" for i, name in enumerate(mu_names))
    fig.suptitle(
        "BMN vs Exact | "
        f"Dx={case_params['Dx']:.3f}, ht={case_params['ht']:.3f} | "
        f"True: {mu_true_str} | Pred: {mu_pred_str}"
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def infer_config_from_checkpoints(
    encoder_checkpoint: Dict[str, Any],
    decoder_checkpoint: Dict[str, Any],
    config_path: Optional[Path],
) -> BMNConfig:
    if config_path is not None and config_path.exists():
        return load_config(config_path)
    cfg_dict = encoder_checkpoint.get("config")
    if isinstance(cfg_dict, dict):
        return config_from_dict(cfg_dict)
    cfg_dict = decoder_checkpoint.get("config")
    if isinstance(cfg_dict, dict):
        return config_from_dict(cfg_dict)
    raise RuntimeError("Could not recover BMNConfig from config file or checkpoints.")


def run_bmn_inverse_random_exact_cases(
    encoder_checkpoint_path: Path,
    decoder_checkpoint_path: Path,
    output_dir: Path,
    device: str,
    n_cases: int,
    seed: int,
    config_path: Optional[Path] = None,
) -> None:
    encoder, encoder_ckpt = load_encoder_model(encoder_checkpoint_path, map_location=device)
    encoder = encoder.to(device)
    decoder_model, decoder_ckpt = load_decoder_model(decoder_checkpoint_path, map_location=device)
    decoder_model = decoder_model.to(device)

    cfg = infer_config_from_checkpoints(encoder_ckpt, decoder_ckpt, config_path)
    obs_scaler = StandardScaler.from_dict(encoder_ckpt["obs_scaler"])
    mu_scaler = StandardScaler.from_dict(encoder_ckpt["mu_scaler"])
    sensor_indices = np.asarray(encoder_ckpt["sensor_indices"], dtype=np.int64)
    observation_vars = [str(v) for v in encoder_ckpt["observation_vars"]]
    output_vars = [str(v) for v in encoder_ckpt["output_vars"]]
    mu_names = [str(v) for v in encoder_ckpt["mu_names"]]
    rng = np.random.default_rng(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "encoder_checkpoint_path": str(encoder_checkpoint_path),
        "decoder_checkpoint_path": str(decoder_checkpoint_path),
        "config_path": None if config_path is None else str(config_path),
        "device": device,
        "requested_cases": int(n_cases),
        "cases": [],
    }

    aggregate_metrics: Dict[str, List[float]] = {}
    successful_cases = 0
    attempts = 0

    print("=" * 88)
    print("Evaluating BMN inverse pipeline on random exact-solver cases")
    print(f"Encoder ckpt : {encoder_checkpoint_path}")
    print(f"Decoder ckpt : {decoder_checkpoint_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Target cases : {n_cases}")
    print(f"Device       : {device}")
    print("=" * 88)

    while successful_cases < n_cases:
        attempts += 1
        case_params = sample_one_case(rng, cfg.ranges, cfg.physical)
        exact = compute_exact_case(case_params, cfg.physical, cfg.exact_solver, n_nodes=int(cfg.dataset.n_nodes))
        if exact is None:
            print(f"[attempt {attempts:04d}] skipped: exact solver failed")
            continue

        exact_out = exact_case_to_output_dict(exact)
        observation = extract_observation_from_exact(exact_out, observation_vars, sensor_indices)
        obs_s = obs_scaler.transform(observation[None, :])
        with torch.no_grad():
            mu_pred_s = encoder(torch.tensor(obs_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
        mu_pred = mu_scaler.inverse_transform(mu_pred_s)[0]

        c = np.asarray([case_params["Dx"], case_params["ht"]], dtype=np.float32)
        pred_y = decode_fullfield_np(decoder_model, decoder_ckpt, exact_out["s"], c, mu_pred.astype(np.float32), device=device)
        pred_out = prediction_array_to_output_dict(exact_out["s"], pred_y, output_vars)

        mu_true = np.asarray([case_params[name] for name in mu_names], dtype=np.float32)
        param_metrics = compute_parameter_metrics(mu_pred, mu_true, mu_names)
        response_metrics = compute_response_metrics(pred_out, exact_out)
        all_metrics = {**param_metrics, **response_metrics}
        for key, value in all_metrics.items():
            aggregate_metrics.setdefault(key, []).append(float(value))

        successful_cases += 1
        fig_path = output_dir / f"random_case_{successful_cases:04d}.png"
        plot_case_comparison(pred_out, exact_out, case_params, mu_pred, mu_names, fig_path)

        summary["cases"].append(
            {
                "case_id": successful_cases,
                "attempt_id": attempts,
                "parameters_true": {name: float(case_params[name]) for name in PARAM_NAMES},
                "parameters_pred": {name: float(mu_pred[i]) for i, name in enumerate(mu_names)},
                "figure": fig_path.name,
                "metrics": all_metrics,
            }
        )
        print(f"[{successful_cases:04d}/{n_cases:04d}] saved {fig_path.name}")

    summary["n_successful_cases"] = successful_cases
    summary["n_attempts"] = attempts
    summary["aggregate_metrics"] = {
        f"mean_{key}": float(np.mean(values))
        for key, values in aggregate_metrics.items()
        if values
    }
    summary["aggregate_metrics"].update(
        {
            f"max_{key}": float(np.max(values))
            for key, values in aggregate_metrics.items()
            if values
        }
    )

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print(f"Summary saved to: {output_dir / 'evaluation_summary.json'}")
    print("=" * 88)


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    default_output_dir = repo_dir / DEFAULT_OUTPUT_DIRNAME
    default_ckpt_dir = repo_dir / "BMN_SCR_DD_outputs"

    parser = argparse.ArgumentParser(description="Evaluate BMN inverse pipeline on random exact-solver cases")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=str(default_ckpt_dir / "BMN_DD_encoder.pth"),
        help="Path to BMN Encoder checkpoint",
    )
    parser.add_argument(
        "--decoder_ckpt",
        type=str,
        default=str(default_ckpt_dir / "Decoder_DD_model.pth"),
        help="Path to Decoder checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(repo_dir / "para_config.json"),
        help="Optional path to para_config.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_dir),
        help="Directory for figures and summary JSON",
    )
    parser.add_argument("--n_cases", type=int, default=50, help="Number of successful random exact cases to evaluate")
    parser.add_argument("--seed", type=int, default=20260430, help="Random seed for case sampling")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for Encoder/Decoder inference")
    args = parser.parse_args()

    run_bmn_inverse_random_exact_cases(
        encoder_checkpoint_path=Path(args.encoder_ckpt),
        decoder_checkpoint_path=Path(args.decoder_ckpt),
        output_dir=Path(args.output_dir),
        device=args.device,
        n_cases=int(args.n_cases),
        seed=int(args.seed),
        config_path=Path(args.config) if args.config else None,
    )


if __name__ == "__main__":
    main()
