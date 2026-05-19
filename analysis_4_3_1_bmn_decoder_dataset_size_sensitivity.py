#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_4_3_1_bmn_decoder_dataset_size_sensitivity.py

Section 4.3.1: sensitivity of BMN performance to Decoder-training dataset size.

For each Decoder dataset size, this script:
1. regenerates the exact-solver Decoder full-field dataset;
2. retrains the Decoder model;
3. rebuilds the decoder-generated Encoder supervision dataset with fixed size;
4. retrains BMN;
5. evaluates BMN on the shared exact evaluation dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from Decoder_DD import load_config
from analysis_4_bmn_sensitivity_utils import (
    DEFAULT_CONFIG_PATH,
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
    train_or_reuse_decoder_then_bmn_scenario,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_outputs" / "4_3_1_bmn_decoder_dataset_size_sensitivity"
DEFAULT_SCENARIO_ROOT = DEFAULT_MODEL_ROOT / "4_3_1_bmn_decoder_dataset_size_sensitivity"
DEFAULT_DECODER_CASE_COUNTS = [50, 200, 500, 1000, 2000, 4000]
DEFAULT_FIXED_ENCODER_CASES = 10000


def parse_case_counts(text: str) -> List[int]:
    values = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one Decoder dataset size is required.")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 4.3.1 BMN sensitivity to Decoder-training dataset size.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--eval_dataset", type=str, default=str(DEFAULT_SHARED_EVAL_DATASET))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenario_root", type=str, default=str(DEFAULT_SCENARIO_ROOT))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--decoder_case_counts", type=str, default=",".join(str(v) for v in DEFAULT_DECODER_CASE_COUNTS))
    parser.add_argument("--encoder_n_cases", type=int, default=DEFAULT_FIXED_ENCODER_CASES)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    apply_style()
    base_cfg = load_config(args.config)
    eval_dataset = load_eval_dataset(args.eval_dataset)
    output_root = ensure_dir(args.output_root)
    data_dir = ensure_dir(output_root / "data")
    fig_dir = ensure_dir(output_root / "figures")
    table_dir = ensure_dir(output_root / "tables")
    scenario_root = ensure_dir(args.scenario_root)
    decoder_case_counts = parse_case_counts(args.decoder_case_counts)

    metrics_by_label: Dict[str, Dict[str, Any]] = {}
    scenario_records: List[Dict[str, Any]] = []

    for n_cases in decoder_case_counts:
        label = f"N={n_cases}"
        tag = f"dec_{n_cases}"
        scenario_dir = scenario_root / tag
        scenario_cfg = load_config(args.config)
        scenario_cfg.dataset.decoder_n_cases = int(n_cases)
        scenario_cfg.dataset.encoder_n_cases = int(args.encoder_n_cases)
        scenario_info = train_or_reuse_decoder_then_bmn_scenario(
            cfg=scenario_cfg,
            scenario_output_dir=scenario_dir,
            scenario_tag=tag,
            force_retrain=args.force_retrain,
        )
        metrics, predictions = evaluate_bmn_checkpoint(
            encoder_ckpt_path=scenario_info["encoder_ckpt_path"],
            decoder_ckpt_path=scenario_info["decoder_ckpt_path"],
            dataset=eval_dataset,
            device=args.device,
        )
        metrics_by_label[label] = metrics
        scenario_records.append(
            {
                "label": label,
                "decoder_n_cases": int(n_cases),
                "encoder_n_cases": int(args.encoder_n_cases),
                "scenario_output_dir": str(scenario_dir),
                "decoder_dataset_path": str(scenario_info["decoder_dataset_path"]),
                "decoder_ckpt_path": str(scenario_info["decoder_ckpt_path"]),
                "decoder_history_path": str(scenario_info["decoder_history_path"]),
                "encoder_dataset_path": str(scenario_info["encoder_dataset_path"]),
                "encoder_ckpt_path": str(scenario_info["encoder_ckpt_path"]),
                "encoder_history_path": str(scenario_info["encoder_history_path"]),
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
            "eval_dataset": str(Path(args.eval_dataset).resolve()),
            "decoder_case_counts": decoder_case_counts,
            "encoder_n_cases": int(args.encoder_n_cases),
            "scenario_records": scenario_records,
        },
        data_dir / "run_info_4_3_1.json",
    )
    save_metrics_bundle(metrics_by_label, table_dir, "table_4_3_1_decoder_dataset_size", "decoder_dataset_size")

    categories = [str(v) for v in decoder_case_counts]
    plot_grouped_bars(
        categories=categories,
        series={
            r"$U_s$": [metrics_by_label[f"N={v}"]["parameter"]["Us"]["nrmse"] for v in decoder_case_counts],
            r"$U_b$": [metrics_by_label[f"N={v}"]["parameter"]["Ub"]["nrmse"] for v in decoder_case_counts],
            r"$p$": [metrics_by_label[f"N={v}"]["parameter"]["p"]["nrmse"] for v in decoder_case_counts],
        },
        x_label="Decoder training dataset size",
        y_label="NRMSE",
        title="Parameter NRMSE vs Decoder dataset size",
        save_name="fig_4_3_1_parameter_nrmse_vs_decoder_dataset_size",
        fig_dir=fig_dir,
    )
    response_names = [str(v) for v in eval_dataset["output_vars"].tolist()]
    plot_grouped_bars(
        categories=categories,
        series={name: [metrics_by_label[f"N={v}"]["response"][name]["nrmse"] for v in decoder_case_counts] for name in response_names},
        x_label="Decoder training dataset size",
        y_label="NRMSE",
        title="Response NRMSE vs Decoder dataset size",
        save_name="fig_4_3_1_response_nrmse_vs_decoder_dataset_size",
        fig_dir=fig_dir,
    )

    output_vars = [str(v) for v in eval_dataset["output_vars"].tolist()]
    first_tag = f"dec_{decoder_case_counts[0]}"
    ref_bundle = load_bmn_bundle(
        require_existing_file(scenario_root / first_tag / f"BMN_DD_encoder_{first_tag}.pth", "Scenario BMN checkpoint"),
        require_existing_file(scenario_root / first_tag / f"Decoder_DD_model_{first_tag}.pth", "Scenario Decoder checkpoint"),
        args.device,
    )
    case_results: Dict[str, Dict[str, Any]] = {}
    for case in TYPICAL_CASES:
        series_by_label: Dict[str, Any] = {}
        y_exact = None
        s = None
        for n_cases in decoder_case_counts:
            tag = f"dec_{n_cases}"
            decoder_ckpt = require_existing_file(scenario_root / tag / f"Decoder_DD_model_{tag}.pth", "Scenario Decoder checkpoint")
            encoder_ckpt = require_existing_file(scenario_root / tag / f"BMN_DD_encoder_{tag}.pth", "Scenario BMN checkpoint")
            bundle = load_bmn_bundle(encoder_ckpt, decoder_ckpt, args.device)
            pred = predict_bmn_typical_case(bundle, base_cfg, case)
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
                save_name=f"fig_4_3_1_{case['case_id']}_decoder_dataset_size_family",
                sensor_indices=ref_bundle["sensor_indices"],
                show_sensor_positions=True,
            )
    if case_results:
        plot_typical_case_family_collection(
            cases=TYPICAL_CASES,
            case_results=case_results,
            output_vars=output_vars,
            fig_dir=fig_dir,
            save_name="fig_4_3_1_decoder_dataset_size_family_all_cases",
            sensor_indices=ref_bundle["sensor_indices"],
            show_sensor_positions=True,
            title="Typical-case response comparison under different Decoder dataset sizes",
        )

    save_json({"status": "completed", "output_root": str(output_root)}, output_root / "analysis_4_3_1_summary.json")


if __name__ == "__main__":
    main()
