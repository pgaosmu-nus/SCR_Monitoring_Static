#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained Decoder_DD model on its held-out test set.

This script:
1. loads a trained Decoder checkpoint
2. reads the original decoder dataset and stored test split
3. re-solves each test case with the exact solver
4. predicts the same case with the Decoder
5. writes one comparison figure per test case
6. saves a JSON summary with per-case metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from Decoder_DD import (
    BMNConfig,
    PARAM_NAMES,
    compute_exact_case,
    config_from_dict,
    decode_fullfield_np,
    load_decoder_model,
)


DEFAULT_OUTPUT_DIRNAME = "decoder_test_exact_comparison"
REQUIRED_VARS = ["x", "z", "theta", "T", "M"]


def load_npz_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=True)
    return {key: data[key] for key in data.files}


def infer_config(checkpoint: Dict[str, Any], dataset: Dict[str, np.ndarray]) -> BMNConfig:
    cfg_dict = checkpoint.get("config")
    if isinstance(cfg_dict, dict):
        return config_from_dict(cfg_dict)

    dataset_cfg_json = dataset.get("config_json")
    if dataset_cfg_json is None:
        raise RuntimeError("Could not recover BMNConfig from checkpoint or dataset.")
    return config_from_dict(json.loads(str(dataset_cfg_json.tolist())))


def output_array_to_dict(
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


def interpolate_exact_to_prediction_grid(pred_s: np.ndarray, exact: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "x": np.interp(pred_s, exact["s"], exact["x"]),
        "z": np.interp(pred_s, exact["s"], exact["z"]),
        "theta": np.interp(pred_s, exact["s"], exact["theta"]),
        "T": np.interp(pred_s, exact["s"], exact["T"]),
        "M": np.interp(pred_s, exact["s"], exact["M"]),
    }


def compute_case_metrics(pred: Dict[str, np.ndarray], exact: Dict[str, np.ndarray]) -> Dict[str, float]:
    exact_i = interpolate_exact_to_prediction_grid(pred["s"], exact)
    metrics: Dict[str, float] = {}
    for key in REQUIRED_VARS:
        diff = pred[key] - exact_i[key]
        metrics[f"rmse_{key}"] = float(np.sqrt(np.mean(diff ** 2)))
        metrics[f"mae_{key}"] = float(np.mean(np.abs(diff)))
    return metrics


def plot_case_comparison(
    pred: Dict[str, np.ndarray],
    exact: Dict[str, np.ndarray],
    case_params: Dict[str, float],
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(13, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(pred["x"], pred["z"], lw=2.0, label="Decoder")
    ax1.plot(exact["x"], exact["z"], "--", lw=2.0, label="Exact")
    ax1.set_title("x-z geometry")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("z [m]")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pred["s"], pred["theta"], lw=2.0, label="Decoder")
    ax2.plot(exact["s"], exact["theta"], "--", lw=2.0, label="Exact")
    ax2.set_title("theta(s)")
    ax2.set_xlabel("s [m]")
    ax2.set_ylabel("theta [rad]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pred["s"], pred["T"] / 1.0e3, lw=2.0, label="Decoder")
    ax3.plot(exact["s"], exact["T"] / 1.0e3, "--", lw=2.0, label="Exact")
    ax3.set_title("T(s)")
    ax3.set_xlabel("s [m]")
    ax3.set_ylabel("T [kN]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(pred["s"], pred["M"] / 1.0e3, lw=2.0, label="Decoder")
    ax4.plot(exact["s"], exact["M"] / 1.0e3, "--", lw=2.0, label="Exact")
    ax4.set_title("M(s)")
    ax4.set_xlabel("s [m]")
    ax4.set_ylabel("M [kN m]")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    fig.suptitle(
        "Decoder vs Exact | "
        f"Dx={case_params['Dx']:.3f}, ht={case_params['ht']:.3f}, "
        f"Us={case_params['Us']:.3f}, Ub={case_params['Ub']:.3f}, p={case_params['p']:.3f}"
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.95])
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def build_case_params_from_dataset(dataset: Dict[str, np.ndarray], case_idx: int) -> Dict[str, float]:
    params = np.asarray(dataset["params"], dtype=np.float32)[case_idx]
    return {name: float(params[i]) for i, name in enumerate(PARAM_NAMES)}


def evaluate_decoder_on_test_set(
    checkpoint_path: Path,
    dataset_path: Optional[Path],
    output_dir: Path,
    device: str,
    max_cases: Optional[int] = None,
) -> None:
    model, checkpoint = load_decoder_model(checkpoint_path, map_location=device)
    model = model.to(device)

    ckpt_dataset_path = checkpoint.get("dataset_path")
    resolved_dataset_path = dataset_path if dataset_path is not None else (
        Path(ckpt_dataset_path) if ckpt_dataset_path is not None else None
    )
    if resolved_dataset_path is None:
        raise RuntimeError("Dataset path must be provided when checkpoint does not store it.")
    if not resolved_dataset_path.exists():
        raise FileNotFoundError(f"Decoder dataset not found: {resolved_dataset_path}")

    dataset = load_npz_dataset(resolved_dataset_path)
    cfg = infer_config(checkpoint, dataset)
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    splits = checkpoint.get("splits")
    if not isinstance(splits, dict) or "test" not in splits:
        raise RuntimeError("Checkpoint does not contain a saved test split.")

    test_indices = np.asarray(splits["test"], dtype=int)
    if max_cases is not None:
        test_indices = test_indices[:max_cases]
    if test_indices.size == 0:
        raise RuntimeError("Test split is empty; nothing to evaluate.")

    s = np.asarray(dataset["s"], dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_path": str(resolved_dataset_path),
        "device": device,
        "n_test_cases_evaluated": int(len(test_indices)),
        "output_vars": output_vars,
        "cases": [],
    }

    aggregate_metrics: Dict[str, List[float]] = {}
    n_nodes = int(len(s))

    print("=" * 88)
    print("Evaluating Decoder_DD on test set with exact-solver comparisons")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Dataset    : {resolved_dataset_path}")
    print(f"Output dir : {output_dir}")
    print(f"Test cases : {len(test_indices)}")
    print(f"Device     : {device}")
    print("=" * 88)

    for i, case_idx in enumerate(test_indices, start=1):
        case_params = build_case_params_from_dataset(dataset, int(case_idx))
        exact = compute_exact_case(case_params, cfg.physical, cfg.exact_solver, n_nodes=n_nodes)
        if exact is None:
            print(f"[{i:04d}/{len(test_indices):04d}] skipped case {case_idx}: exact solver failed")
            summary["cases"].append(
                {
                    "case_idx": int(case_idx),
                    "parameters": case_params,
                    "status": "exact_solver_failed",
                }
            )
            continue

        c = np.asarray([case_params["Dx"], case_params["ht"]], dtype=np.float32)
        mu = np.asarray([case_params["Us"], case_params["Ub"], case_params["p"]], dtype=np.float32)
        pred_y = decode_fullfield_np(model, checkpoint, s, c, mu, device=device)
        pred = output_array_to_dict(s, pred_y, output_vars)

        metrics = compute_case_metrics(pred, exact)
        for key, value in metrics.items():
            aggregate_metrics.setdefault(key, []).append(value)

        fig_path = output_dir / f"test_case_{i:04d}_datasetidx_{int(case_idx):04d}.png"
        plot_case_comparison(pred, exact, case_params, fig_path)

        summary["cases"].append(
            {
                "case_idx": int(case_idx),
                "parameters": case_params,
                "status": "ok",
                "figure": fig_path.name,
                "metrics": metrics,
            }
        )
        print(f"[{i:04d}/{len(test_indices):04d}] saved {fig_path.name}")

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
    print("Decoder test-set evaluation finished.")
    print(f"Summary saved to: {output_dir / 'evaluation_summary.json'}")
    print("=" * 88)


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate trained Decoder_DD on its held-out test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(repo_dir / "BMN_SCR_DD_outputs" / "Decoder_DD_model.pth"),
        help="Path to trained Decoder_DD checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional override path to decoder_fullfield_dataset.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(repo_dir / DEFAULT_OUTPUT_DIRNAME),
        help="Directory to save all test-case figures and summary JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit on the number of test cases to evaluate.",
    )
    args = parser.parse_args()

    evaluate_decoder_on_test_set(
        checkpoint_path=Path(args.checkpoint),
        dataset_path=None if args.dataset is None else Path(args.dataset),
        output_dir=Path(args.output_dir),
        device=args.device,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
