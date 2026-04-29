#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random-case evaluation for the standalone 3V hybrid parametric PINN.

This script:
1. loads `scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.pth`
2. samples 20 in-domain exact-solvable cases
3. samples 20 mildly out-of-domain exact-solvable cases
4. predicts each case with the PINN
5. compares each prediction against the exact solution
6. writes one comparison figure per case
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0 import (
    EXACT_SOLVER_AVAILABLE,
    EXACT_SOLVER_IMPORT_ERROR,
    NetworkConfig,
    ParameterRanges,
    PhysicalConfig,
    SCRStaticPINNMT,
    ScaleConfig,
    SingleCaseConfig,
    case_geometry_is_admissible,
    compute_exact_solution,
    plot_prediction_vs_exact,
    predict_single_case,
)


MODEL_FILENAME = "scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.pth"
OUTPUT_DIRNAME = "random_case_evaluation"
N_IN_DOMAIN = 20
N_OUT_DOMAIN = 20
EXTRAP_FRAC = 0.20
SEED = 42
NODES_PRED = 256
MAX_ATTEMPTS = 20000


def infer_network_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> NetworkConfig:
    hidden_weight_keys = sorted(
        key for key in state_dict.keys()
        if key.startswith("hidden.") and key.endswith(".weight")
    )
    if not hidden_weight_keys:
        raise RuntimeError("Could not infer network structure from state dict.")
    first_weight = state_dict[hidden_weight_keys[0]]
    hidden_dim = int(first_weight.shape[0])
    num_hidden_layers = len(hidden_weight_keys)
    return NetworkConfig(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)


def build_outer_ranges(base: ParameterRanges, frac: float) -> Dict[str, Tuple[float, float]]:
    expanded: Dict[str, Tuple[float, float]] = {}
    for name in ["Us", "Ub", "p", "Dx", "ht"]:
        vmin = getattr(base, f"{name}_min")
        vmax = getattr(base, f"{name}_max")
        span = vmax - vmin
        expanded[name] = (vmin - frac * span, vmax + frac * span)
    return expanded


def sample_in_domain_case(rng: np.random.Generator, ranges: ParameterRanges, phys: PhysicalConfig) -> SingleCaseConfig:
    while True:
        ub = float(rng.uniform(ranges.Ub_min, ranges.Ub_max))
        us = float(rng.uniform(max(ub, ranges.Us_min), ranges.Us_max))
        p = float(rng.uniform(ranges.p_min, ranges.p_max))
        dx = float(rng.uniform(ranges.Dx_min, ranges.Dx_max))
        ht = float(rng.uniform(ranges.ht_min, ranges.ht_max))
        if case_geometry_is_admissible(dx, ht, phys):
            return SingleCaseConfig(Us=us, Ub=ub, p=p, Dx=dx, ht=ht)


def sample_out_domain_case(
    rng: np.random.Generator,
    ranges: ParameterRanges,
    phys: PhysicalConfig,
    frac: float,
) -> SingleCaseConfig:
    expanded = build_outer_ranges(ranges, frac)
    while True:
        ub = float(rng.uniform(*expanded["Ub"]))
        us_low = max(ub, expanded["Us"][0])
        us = float(rng.uniform(us_low, expanded["Us"][1]))
        p = float(rng.uniform(*expanded["p"]))
        dx = float(rng.uniform(*expanded["Dx"]))
        ht = float(rng.uniform(*expanded["ht"]))

        outside = (
            us < ranges.Us_min or us > ranges.Us_max or
            ub < ranges.Ub_min or ub > ranges.Ub_max or
            p < ranges.p_min or p > ranges.p_max or
            dx < ranges.Dx_min or dx > ranges.Dx_max or
            ht < ranges.ht_min or ht > ranges.ht_max
        )
        if not outside:
            continue
        if case_geometry_is_admissible(dx, ht, phys):
            return SingleCaseConfig(Us=us, Ub=ub, p=p, Dx=dx, ht=ht)


def collect_exact_solvable_cases(
    n_cases: int,
    sampler_name: str,
    rng: np.random.Generator,
    ranges: ParameterRanges,
    phys: PhysicalConfig,
    frac: float = EXTRAP_FRAC,
) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    attempts = 0
    while len(cases) < n_cases:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise RuntimeError(f"Failed to collect {n_cases} exact-solvable {sampler_name} cases within {MAX_ATTEMPTS} attempts.")
        if sampler_name == "in_domain":
            case = sample_in_domain_case(rng, ranges, phys)
        elif sampler_name == "out_domain":
            case = sample_out_domain_case(rng, ranges, phys, frac)
        else:
            raise ValueError(f"Unknown sampler_name: {sampler_name}")

        exact = compute_exact_solution(case, phys)
        if exact is None:
            continue
        cases.append({"case": case, "exact": exact})
    return cases


def compute_case_metrics(pred: Dict[str, np.ndarray], exact: Dict[str, np.ndarray]) -> Dict[str, float]:
    s_pred = pred["s"]
    metrics: Dict[str, float] = {}
    exact_interp = {
        "theta": np.interp(s_pred, exact["s"], exact["theta"]),
        "T": np.interp(s_pred, exact["s"], exact["N"]),
        "Q": np.interp(s_pred, exact["s"], exact["Q"]),
        "M": np.interp(s_pred, exact["s"], exact["M"]),
        "x": np.interp(s_pred, exact["s"], exact["x"]),
        "z": np.interp(s_pred, exact["s"], exact["z"]),
    }
    for key_pred, key_exact in [("theta", "theta"), ("T", "T"), ("Q", "Q"), ("M", "M"), ("x", "x"), ("z", "z")]:
        diff = pred[key_pred] - exact_interp[key_exact]
        metrics[f"rmse_{key_pred}"] = float(np.sqrt(np.mean(diff ** 2)))
    return metrics


def evaluate_case_set(
    model: SCRStaticPINNMT,
    cases: List[Dict[str, object]],
    label: str,
    output_dir: Path,
    ranges: ParameterRanges,
    phys: PhysicalConfig,
    net_cfg: NetworkConfig,
    scales: ScaleConfig,
    device: str,
) -> List[Dict[str, object]]:
    case_dir = output_dir / label
    case_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    for idx, item in enumerate(cases, start=1):
        case = item["case"]
        exact = item["exact"]
        pred = predict_single_case(model, case, ranges, phys, net_cfg, scales, NODES_PRED, device)
        fig_path = case_dir / f"{label}_case_{idx:02d}.png"
        plot_prediction_vs_exact(pred, exact, fig_path)
        summary = {
            "case_index": idx,
            "case_type": label,
            "parameters": asdict(case),
            "figure": fig_path.name,
            "metrics": compute_case_metrics(pred, exact),
        }
        summaries.append(summary)
    return summaries


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    model_path = repo_dir / MODEL_FILENAME
    output_dir = repo_dir / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    if not EXACT_SOLVER_AVAILABLE:
        raise RuntimeError(
            "Exact solver is unavailable, so random-case evaluation cannot proceed.\n"
            f"Import detail: {EXACT_SOLVER_IMPORT_ERROR}"
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phys = PhysicalConfig()
    scales = ScaleConfig().build_from_physics(phys)
    ranges = ParameterRanges()

    state_dict = torch.load(model_path, map_location=device)
    net_cfg = infer_network_config_from_state_dict(state_dict)
    model = SCRStaticPINNMT(net_cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    rng = np.random.default_rng(SEED)
    in_cases = collect_exact_solvable_cases(N_IN_DOMAIN, "in_domain", rng, ranges, phys)
    out_cases = collect_exact_solvable_cases(N_OUT_DOMAIN, "out_domain", rng, ranges, phys, frac=EXTRAP_FRAC)

    in_summary = evaluate_case_set(model, in_cases, "in_domain", output_dir, ranges, phys, net_cfg, scales, device)
    out_summary = evaluate_case_set(model, out_cases, "out_domain", output_dir, ranges, phys, net_cfg, scales, device)

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "device": device,
                "seed": SEED,
                "n_in_domain": N_IN_DOMAIN,
                "n_out_domain": N_OUT_DOMAIN,
                "extrapolation_fraction": EXTRAP_FRAC,
                "network": asdict(net_cfg),
                "ranges": asdict(ranges),
                "in_domain_cases": in_summary,
                "out_domain_cases": out_summary,
            },
            f,
            indent=2,
        )

    print("=" * 88)
    print("Random-case evaluation finished.")
    print(f"Output directory: {output_dir}")
    print(f"In-domain cases : {len(in_summary)}")
    print(f"Out-domain cases: {len(out_summary)}")
    print("=" * 88)


if __name__ == "__main__":
    main()
