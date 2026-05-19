#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_4_3_bmn_dataset_size_sensitivity.py

Section 4.3: sensitivity of BMN performance to Encoder-training dataset size.

This script reuses BMN_DD training functions, retrains BMN under multiple
encoder dataset sizes, evaluates all trained models on the same shared exact
evaluation dataset, and exports comparison tables/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from Decoder_DD import load_config
from analysis_4_bmn_sensitivity_utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DECODER_CKPT,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SHARED_EVAL_DATASET,
    TYPICAL_CASES,
    apply_style,
    ensure_dir,
    evaluate_bmn_checkpoint,
    load_bmn_bundle,
    load_eval_dataset,
    mean_response_nrmse,
    plot_grouped_bars,
    plot_typical_case_family,
    plot_typical_case_family_collection,
    predict_bmn_typical_case,
    require_existing_file,
    save_json,
    save_metrics_bundle,
    train_or_reuse_bmn_scenario,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_outputs" / "4_3_bmn_dataset_size_sensitivity"
DEFAULT_SCENARIO_ROOT = DEFAULT_MODEL_ROOT / "4_3_bmn_dataset_size_sensitivity"
DEFAULT_CASE_COUNTS = [2000, 5000, 10000, 20000]


def parse_case_counts(text: str) -> List[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one encoder dataset size is required.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 4.3 BMN dataset-size sensitivity analysis.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--decoder_ckpt", type=str, default=str(DEFAULT_DECODER_CKPT))
    parser.add_argument("--eval_dataset", type=str, default=str(DEFAULT_SHARED_EVAL_DATASET))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenario_root", type=str, default=str(DEFAULT_SCENARIO_ROOT))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--case_counts", type=str, default=",".join(str(v) for v in DEFAULT_CASE_COUNTS))
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    apply_style()
    cfg = load_config(args.config)
    decoder_ckpt_path = require_existing_file(args.decoder_ckpt, "Decoder checkpoint")
    eval_dataset = load_eval_dataset(args.eval_dataset)
    output_root = ensure_dir(args.output_root)
    data_dir = ensure_dir(output_root / "data")
    fig_dir = ensure_dir(output_root / "figures")
    table_dir = ensure_dir(output_root / "tables")
    scenario_root = ensure_dir(args.scenario_root)
    case_counts = parse_case_counts(args.case_counts)

    metrics_by_label: Dict[str, Dict[str, Any]] = {}
    scenario_records: List[Dict[str, Any]] = []

    for n_cases in case_counts:
        label = f"N={n_cases}"
        tag = f"enc_{n_cases}"
        scenario_dir = scenario_root / tag
        scenario_cfg = load_config(args.config)
        scenario_cfg.dataset.encoder_n_cases = int(n_cases)
        scenario_info = train_or_reuse_bmn_scenario(
            cfg=scenario_cfg,
            decoder_ckpt_path=decoder_ckpt_path,
            scenario_output_dir=scenario_dir,
            scenario_tag=tag,
            force_retrain=args.force_retrain,
        )
        metrics, predictions = evaluate_bmn_checkpoint(
            encoder_ckpt_path=scenario_info["encoder_ckpt_path"],
            decoder_ckpt_path=decoder_ckpt_path,
            dataset=eval_dataset,
            device=args.device,
        )
        metrics_by_label[label] = metrics
        scenario_records.append(
            {
                "label": label,
                "encoder_n_cases": int(n_cases),
                "scenario_output_dir": str(scenario_dir),
                "encoder_dataset_path": str(scenario_info["encoder_dataset_path"]),
                "encoder_ckpt_path": str(scenario_info["encoder_ckpt_path"]),
                "history_path": str(scenario_info["history_path"]),
                "train_elapsed_seconds": scenario_info["train_elapsed_seconds"],
                "mean_response_nrmse": mean_response_nrmse(metrics),
            }
        )
        save_json(metrics, data_dir / f"metrics_{tag}.json")
        save_json(
            {
                "s": predictions["s"].tolist(),
                "mu_true": predictions["mu_true"].tolist(),
                "mu_pred": predictions["mu_pred"].tolist(),
            },
            data_dir / f"predictions_{tag}.json",
        )

    save_json(
        {
            "config": str(Path(args.config).resolve()),
            "decoder_ckpt": str(decoder_ckpt_path),
            "eval_dataset": str(Path(args.eval_dataset).resolve()),
            "case_counts": case_counts,
            "scenario_records": scenario_records,
        },
        data_dir / "run_info_4_3.json",
    )
    save_metrics_bundle(metrics_by_label, table_dir, "table_4_3_dataset_size", "dataset_size")

    categories = [str(v) for v in case_counts]
    plot_grouped_bars(
        categories=categories,
        series={
            r"$U_s$": [metrics_by_label[f"N={v}"]["parameter"]["Us"]["nrmse"] for v in case_counts],
            r"$U_b$": [metrics_by_label[f"N={v}"]["parameter"]["Ub"]["nrmse"] for v in case_counts],
            r"$p$": [metrics_by_label[f"N={v}"]["parameter"]["p"]["nrmse"] for v in case_counts],
        },
        x_label="Encoder training dataset size",
        y_label="NRMSE",
        title="Parameter NRMSE vs encoder dataset size",
        save_name="fig_4_3_parameter_nrmse_vs_dataset_size",
        fig_dir=fig_dir,
    )
    response_names = [str(v) for v in eval_dataset["output_vars"].tolist()]
    plot_grouped_bars(
        categories=categories,
        series={name: [metrics_by_label[f"N={v}"]["response"][name]["nrmse"] for v in case_counts] for name in response_names},
        x_label="Encoder training dataset size",
        y_label="NRMSE",
        title="Response NRMSE vs encoder dataset size",
        save_name="fig_4_3_response_nrmse_vs_dataset_size",
        fig_dir=fig_dir,
    )

    output_vars = [str(v) for v in eval_dataset["output_vars"].tolist()]
    case_results: Dict[str, Dict[str, Any]] = {}
    ref_bundle = load_bmn_bundle(scenario_root / f"enc_{case_counts[0]}" / f"BMN_DD_encoder_enc_{case_counts[0]}.pth", decoder_ckpt_path, args.device)
    for case in TYPICAL_CASES:
        series_by_label: Dict[str, Any] = {}
        y_exact = None
        s = None
        for n_cases in case_counts:
            tag = f"enc_{n_cases}"
            bundle = load_bmn_bundle(scenario_root / tag / f"BMN_DD_encoder_{tag}.pth", decoder_ckpt_path, args.device)
            pred = predict_bmn_typical_case(bundle, cfg, case)
            y_exact = pred["y_true"]
            s = pred["s"]
            series_by_label[f"N={n_cases}"] = pred["y_pred"]
        if y_exact is not None and s is not None:
            case_results[case["case_id"]] = {"s": s, "y_true": y_exact, "series_by_label": series_by_label}
            plot_typical_case_family(
                case_id=case["case_id"],
                desc=case["description"],
                s=s,
                y_exact=y_exact,
                series_by_label=series_by_label,
                output_vars=output_vars,
                fig_dir=fig_dir,
                save_name=f"fig_4_3_{case['case_id']}_dataset_size_family",
                sensor_indices=ref_bundle["sensor_indices"],
                show_sensor_positions=True,
            )
    if case_results:
        plot_typical_case_family_collection(
            cases=TYPICAL_CASES,
            case_results=case_results,
            output_vars=output_vars,
            fig_dir=fig_dir,
            save_name="fig_4_3_dataset_size_family_all_cases",
            sensor_indices=ref_bundle["sensor_indices"],
            show_sensor_positions=True,
            title="Typical-case response comparison under different encoder dataset sizes",
        )

    save_json({"status": "completed", "output_root": str(output_root)}, output_root / "analysis_4_3_summary.json")


if __name__ == "__main__":
    main()
