#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_4_2_4_4_common_eval_dataset.py

Generate the shared in-domain exact-solver evaluation dataset used by
Section 4.2 and Section 4.4.

The saved dataset is placed under `paper_outputs/` and can be consumed by:

    python analysis_4_2_bmn_inversion_validation.py
    python analysis_4_4_model_comparison.py

through their default settings or via `--eval_dataset`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from Decoder_DD import C_NAMES, MU_NAMES, PARAM_NAMES, compute_exact_case, load_config, sample_one_case


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "para_config.json"
DEFAULT_OUTPUT_NPZ = REPO_ROOT / "paper_outputs" / "paper_testset_4_2_4_4_in_domain_exact.npz"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "paper_outputs" / "paper_testset_4_2_4_4_in_domain_exact_summary.json"
DEFAULT_N_CASES = 500
DEFAULT_SEED = 20260503


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


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
            print(f"  exact evaluation cases collected: {len(cases)}/{n_cases}")
    return cases


def cases_to_dataset(cfg, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not cases:
        raise ValueError("No exact cases were generated.")
    output_vars = list(cfg.dataset.output_vars)
    s = np.asarray(cases[0]["exact"]["s"], dtype=np.float32)
    params = np.zeros((len(cases), len(PARAM_NAMES)), dtype=np.float32)
    c = np.zeros((len(cases), len(C_NAMES)), dtype=np.float32)
    mu = np.zeros((len(cases), len(MU_NAMES)), dtype=np.float32)
    y = np.zeros((len(cases), len(s), len(output_vars)), dtype=np.float32)

    for i, item in enumerate(cases):
        case = item["params"]
        exact = item["exact"]
        params[i] = case_dict_to_array(case)
        c[i] = np.asarray([case[name] for name in C_NAMES], dtype=np.float32)
        mu[i] = np.asarray([case[name] for name in MU_NAMES], dtype=np.float32)
        y[i] = np.stack([np.asarray(exact[name], dtype=np.float32) for name in output_vars], axis=-1)

    return {
        "s": s,
        "params": params,
        "c": c,
        "mu": mu,
        "y": y,
        "output_vars": np.asarray(output_vars),
        "param_names": np.asarray(PARAM_NAMES),
        "c_names": np.asarray(C_NAMES),
        "mu_names": np.asarray(MU_NAMES),
        "source": np.asarray("shared_in_domain_exact_eval_set"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the shared in-domain exact evaluation dataset for Section 4.2 and 4.4.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_NPZ))
    parser.add_argument("--summary", type=str, default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--n_cases", type=int, default=DEFAULT_N_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = ensure_parent(args.output)
    summary_path = ensure_parent(args.summary)

    print("=" * 88)
    print("Generating shared in-domain exact evaluation dataset for Section 4.2 and 4.4")
    print(f"Config        : {Path(args.config).resolve()}")
    print(f"Output npz    : {output_path}")
    print(f"Output summary: {summary_path}")
    print(f"n_cases       : {args.n_cases}")
    print(f"seed          : {args.seed}")
    print("=" * 88)

    cases = build_random_exact_test_set(cfg, int(args.n_cases), int(args.seed))
    dataset = cases_to_dataset(cfg, cases)
    np.savez_compressed(output_path, **dataset)

    summary = {
        "n_cases": int(args.n_cases),
        "seed": int(args.seed),
        "config_path": str(Path(args.config).resolve()),
        "output_path": str(output_path),
        "output_vars": list(cfg.dataset.output_vars),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 88)
    print(f"Saved shared evaluation dataset to: {output_path}")
    print(f"Saved summary to                 : {summary_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
