#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_4_4_out_of_domain_testset.py

Generate an out-of-domain exact-solver test dataset for Section 4.4.

This script samples cases outside the training parameter domain defined in
`para_config.json`, solves them with the exact SCR solver, and saves the
results under `paper_outputs/`.

The saved `.npz` is compatible with `analysis_4_4_model_comparison.py`
through `--eval_dataset`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from Decoder_DD import (
    C_NAMES,
    MU_NAMES,
    PARAM_NAMES,
    BMNConfig,
    case_geometry_is_admissible,
    compute_exact_case,
    load_config,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "para_config.json"
DEFAULT_OUTPUT_NPZ = REPO_ROOT / "paper_outputs" / "paper_testset_4_4_out_of_domain_exact.npz"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "paper_outputs" / "paper_testset_4_4_out_of_domain_exact_summary.json"


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_training_bounds(cfg: BMNConfig) -> Dict[str, Tuple[float, float]]:
    r = cfg.ranges
    return {
        "Dx": (float(r.Dx_min), float(r.Dx_max)),
        "ht": (float(r.ht_min), float(r.ht_max)),
        "Us": (float(r.Us_min), float(r.Us_max)),
        "Ub": (float(r.Ub_min), float(r.Ub_max)),
        "p": (float(r.p_min), float(r.p_max)),
    }


def get_expanded_bounds(
    bounds: Dict[str, Tuple[float, float]],
    c_margin_ratio: float,
    mu_margin_ratio: float,
) -> Dict[str, Tuple[float, float]]:
    expanded: Dict[str, Tuple[float, float]] = {}
    for name, (lo, hi) in bounds.items():
        span = hi - lo
        ratio = c_margin_ratio if name in C_NAMES else mu_margin_ratio
        pad = ratio * span
        new_lo = lo - pad
        new_hi = hi + pad
        if name in {"Us", "Ub", "p"}:
            new_lo = max(0.0, new_lo)
        expanded[name] = (float(new_lo), float(new_hi))
    return expanded


def is_out_of_domain(
    case: Dict[str, float],
    bounds: Dict[str, Tuple[float, float]],
    scope: str,
) -> bool:
    names = list(PARAM_NAMES) if scope == "all" else list(MU_NAMES)
    for name in names:
        lo, hi = bounds[name]
        val = float(case[name])
        if val < lo or val > hi:
            return True
    return False


def get_outside_flags(case: Dict[str, float], bounds: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    for name in PARAM_NAMES:
        lo, hi = bounds[name]
        val = float(case[name])
        flags[name] = bool(val < lo or val > hi)
    return flags


def sample_one_out_of_domain_case(
    rng: np.random.Generator,
    cfg: BMNConfig,
    train_bounds: Dict[str, Tuple[float, float]],
    expanded_bounds: Dict[str, Tuple[float, float]],
    outside_scope: str,
) -> Dict[str, float]:
    while True:
        dx = float(rng.uniform(*expanded_bounds["Dx"]))
        ht = float(rng.uniform(*expanded_bounds["ht"]))
        ub = float(rng.uniform(*expanded_bounds["Ub"]))
        us_lo = max(ub, expanded_bounds["Us"][0])
        us_hi = expanded_bounds["Us"][1]
        if us_lo >= us_hi:
            continue
        us = float(rng.uniform(us_lo, us_hi))
        p = float(rng.uniform(*expanded_bounds["p"]))
        case = {"Dx": dx, "ht": ht, "Us": us, "Ub": ub, "p": p}
        if not is_out_of_domain(case, train_bounds, outside_scope):
            continue
        if case_geometry_is_admissible(dx, ht, cfg.physical):
            return case


def generate_dataset(
    cfg: BMNConfig,
    n_cases: int,
    seed: int,
    outside_scope: str,
    c_margin_ratio: float,
    mu_margin_ratio: float,
    max_attempts: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    train_bounds = get_training_bounds(cfg)
    expanded_bounds = get_expanded_bounds(train_bounds, c_margin_ratio=c_margin_ratio, mu_margin_ratio=mu_margin_ratio)
    output_vars = list(cfg.dataset.output_vars)

    params_rows: List[List[float]] = []
    c_rows: List[List[float]] = []
    mu_rows: List[List[float]] = []
    y_rows: List[np.ndarray] = []
    outside_rows: List[List[int]] = []
    exact_cases: List[Dict[str, Any]] = []

    attempts = 0
    while len(exact_cases) < n_cases:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to collect {n_cases} out-of-domain exact cases within {max_attempts} attempts. "
                f"Collected {len(exact_cases)} cases."
            )

        case = sample_one_out_of_domain_case(
            rng=rng,
            cfg=cfg,
            train_bounds=train_bounds,
            expanded_bounds=expanded_bounds,
            outside_scope=outside_scope,
        )
        exact = compute_exact_case(case, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        if exact is None:
            continue

        outside_flags = get_outside_flags(case, train_bounds)
        exact_cases.append({"params": case, "exact": exact, "outside_flags": outside_flags})

        params_rows.append([case[name] for name in PARAM_NAMES])
        c_rows.append([case[name] for name in C_NAMES])
        mu_rows.append([case[name] for name in MU_NAMES])
        y_rows.append(np.stack([np.asarray(exact[name], dtype=np.float32) for name in output_vars], axis=-1))
        outside_rows.append([int(outside_flags[name]) for name in PARAM_NAMES])

        if len(exact_cases) % max(1, n_cases // 10) == 0:
            print(f"  collected out-of-domain exact cases: {len(exact_cases)}/{n_cases}")

    s = np.asarray(exact_cases[0]["exact"]["s"], dtype=np.float32)
    dataset = {
        "s": s,
        "params": np.asarray(params_rows, dtype=np.float32),
        "c": np.asarray(c_rows, dtype=np.float32),
        "mu": np.asarray(mu_rows, dtype=np.float32),
        "y": np.asarray(y_rows, dtype=np.float32),
        "output_vars": np.asarray(output_vars),
        "param_names": np.asarray(PARAM_NAMES),
        "c_names": np.asarray(C_NAMES),
        "mu_names": np.asarray(MU_NAMES),
        "outside_flags": np.asarray(outside_rows, dtype=np.int8),
        "outside_flag_names": np.asarray(PARAM_NAMES),
    }

    outside_counts = {name: int(np.sum(dataset["outside_flags"][:, i])) for i, name in enumerate(PARAM_NAMES)}
    summary = {
        "n_cases": int(n_cases),
        "seed": int(seed),
        "outside_scope": outside_scope,
        "c_margin_ratio": float(c_margin_ratio),
        "mu_margin_ratio": float(mu_margin_ratio),
        "train_bounds": {k: [float(v[0]), float(v[1])] for k, v in train_bounds.items()},
        "expanded_bounds": {k: [float(v[0]), float(v[1])] for k, v in expanded_bounds.items()},
        "outside_counts": outside_counts,
        "all_cases_outside_any_parameter": int(np.sum(np.any(dataset["outside_flags"] > 0, axis=1))),
        "cases_outside_mu_domain": int(np.sum(np.any(dataset["outside_flags"][:, 2:] > 0, axis=1))),
    }
    return dataset, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an out-of-domain exact-solver test dataset for Section 4.4.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to para_config.json.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_NPZ), help="Output .npz path.")
    parser.add_argument("--summary", type=str, default=str(DEFAULT_OUTPUT_JSON), help="Output summary .json path.")
    parser.add_argument("--n_cases", type=int, default=200, help="Number of exact cases to generate.")
    parser.add_argument("--seed", type=int, default=20260505, help="Random seed for out-of-domain sampling.")
    parser.add_argument(
        "--outside_scope",
        type=str,
        default="all",
        choices=["all", "mu_only"],
        help="'all': at least one of [Dx, ht, Us, Ub, p] is outside the training domain; "
             "'mu_only': at least one of [Us, Ub, p] is outside the training domain.",
    )
    parser.add_argument("--c_margin_ratio", type=float, default=0.10, help="Relative expansion ratio for Dx and ht.")
    parser.add_argument("--mu_margin_ratio", type=float, default=0.20, help="Relative expansion ratio for Us, Ub, and p.")
    parser.add_argument("--max_attempts", type=int, default=200000, help="Maximum total sampling attempts.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = ensure_parent(args.output)
    summary_path = ensure_parent(args.summary)

    print("=" * 88)
    print("Generating out-of-domain exact-solver test set for Section 4.4")
    print(f"Config        : {Path(args.config).resolve()}")
    print(f"Output npz    : {output_path}")
    print(f"Output summary: {summary_path}")
    print(f"n_cases       : {args.n_cases}")
    print(f"seed          : {args.seed}")
    print(f"outside_scope : {args.outside_scope}")
    print(f"c_margin_ratio: {args.c_margin_ratio}")
    print(f"mu_margin_ratio: {args.mu_margin_ratio}")
    print("=" * 88)

    dataset, summary = generate_dataset(
        cfg=cfg,
        n_cases=int(args.n_cases),
        seed=int(args.seed),
        outside_scope=str(args.outside_scope),
        c_margin_ratio=float(args.c_margin_ratio),
        mu_margin_ratio=float(args.mu_margin_ratio),
        max_attempts=int(args.max_attempts),
    )

    np.savez_compressed(output_path, **dataset)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print(f"Saved out-of-domain test set to: {output_path}")
    print(f"Saved summary to              : {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
