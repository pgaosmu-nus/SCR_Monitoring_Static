#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis_4_5_bmn_sensor_layout_sensitivity.py

Section 4.5: sensitivity of BMN performance to monitoring-point number and layout.

This script reuses BMN_DD training functions, retrains BMN under multiple
sensor-count and sensor-layout scenarios, evaluates all trained models on the
same shared exact evaluation dataset, and exports comparison tables/figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "paper_outputs" / "4_5_bmn_sensor_layout_sensitivity"
DEFAULT_SCENARIO_ROOT = DEFAULT_MODEL_ROOT / "4_5_bmn_sensor_layout_sensitivity"
DEFAULT_SENSOR_COUNTS = [6, 8, 12, 16, 20, 24]


def resolve_current_sensor_s(cfg) -> List[float]:
    if cfg.dataset.sensor_s:
        return sorted(float(v) for v in cfg.dataset.sensor_s)
    if cfg.dataset.sensor_indices:
        s = np.linspace(0.0, float(cfg.physical.L), int(cfg.dataset.n_nodes), dtype=np.float32)
        return sorted(float(s[int(idx)]) for idx in cfg.dataset.sensor_indices)
    raise ValueError("Current config does not define sensor_s or sensor_indices explicitly.")


def extend_sensor_s(base_sensor_s: List[float], target_count: int) -> List[float]:
    base = sorted(base_sensor_s)
    base_count = len(base)
    if target_count < base_count:
        raise ValueError("Target count must be >= current base sensor count.")
    if target_count == base_count:
        return base
    spacing = float(np.median(np.diff(base)))
    extra = target_count - base_count
    start = base[0] - extra * spacing
    return [float(start + i * spacing) for i in range(target_count)]


def uniform_sensor_s(start: float, end: float, count: int) -> List[float]:
    vals = np.linspace(start, end, count + 2, dtype=np.float32)[1:-1]
    return [float(v) for v in vals]


def build_count_scenarios(cfg) -> Dict[str, List[float]]:
    current = resolve_current_sensor_s(cfg)
    return {f"C{count}": extend_sensor_s(current, int(count)) for count in DEFAULT_SENSOR_COUNTS}


def build_layout_scenarios(cfg) -> Dict[str, List[float]]:
    L = float(cfg.physical.L)
    current = resolve_current_sensor_s(cfg)
    n = len(current)
    return {
        "current6": current,
        "full_uniform6": uniform_sensor_s(0.0, L, n),
        "upper_half_uniform6": uniform_sensor_s(0.5 * L, L, n),
        "lower_half_uniform6": uniform_sensor_s(0.0, 0.5 * L, n),
    }


def run_scenarios(
    cfg_path: str | Path,
    decoder_ckpt_path: str | Path,
    eval_dataset_path: str | Path,
    device: str,
    scenario_root: Path,
    scenario_group: str,
    sensor_scenarios: Dict[str, List[float]],
    force_retrain: bool,
) -> Dict[str, Dict[str, Any]]:
    base_cfg = load_config(cfg_path)
    eval_dataset = load_eval_dataset(eval_dataset_path)
    metrics_by_label: Dict[str, Dict[str, Any]] = {}
    for label, sensor_s in sensor_scenarios.items():
        scenario_cfg = load_config(cfg_path)
        scenario_cfg.dataset.sensor_indices = []
        scenario_cfg.dataset.sensor_s = [float(v) for v in sensor_s]
        scenario_dir = scenario_root / scenario_group / label
        tag = f"{scenario_group}_{label}"
        scenario_info = train_or_reuse_bmn_scenario(
            cfg=scenario_cfg,
            decoder_ckpt_path=decoder_ckpt_path,
            scenario_output_dir=scenario_dir,
            scenario_tag=tag,
            force_retrain=force_retrain,
        )
        metrics, _ = evaluate_bmn_checkpoint(
            encoder_ckpt_path=scenario_info["encoder_ckpt_path"],
            decoder_ckpt_path=decoder_ckpt_path,
            dataset=eval_dataset,
            device=device,
        )
        metrics_by_label[label] = metrics
    return metrics_by_label


def main() -> None:
    parser = argparse.ArgumentParser(description="Section 4.5 BMN sensor-count and sensor-layout sensitivity analysis.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--decoder_ckpt", type=str, default=str(DEFAULT_DECODER_CKPT))
    parser.add_argument("--eval_dataset", type=str, default=str(DEFAULT_SHARED_EVAL_DATASET))
    parser.add_argument("--output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--scenario_root", type=str, default=str(DEFAULT_SCENARIO_ROOT))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
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

    count_scenarios = build_count_scenarios(cfg)
    layout_scenarios = build_layout_scenarios(cfg)

    count_metrics = run_scenarios(
        cfg_path=args.config,
        decoder_ckpt_path=decoder_ckpt_path,
        eval_dataset_path=args.eval_dataset,
        device=args.device,
        scenario_root=scenario_root,
        scenario_group="count",
        sensor_scenarios=count_scenarios,
        force_retrain=args.force_retrain,
    )
    layout_metrics = run_scenarios(
        cfg_path=args.config,
        decoder_ckpt_path=decoder_ckpt_path,
        eval_dataset_path=args.eval_dataset,
        device=args.device,
        scenario_root=scenario_root,
        scenario_group="layout",
        sensor_scenarios=layout_scenarios,
        force_retrain=args.force_retrain,
    )

    save_json(
        {
            "config": str(Path(args.config).resolve()),
            "decoder_ckpt": str(decoder_ckpt_path),
            "eval_dataset": str(Path(args.eval_dataset).resolve()),
            "count_scenarios": count_scenarios,
            "layout_scenarios": layout_scenarios,
        },
        data_dir / "run_info_4_5.json",
    )

    save_metrics_bundle(count_metrics, table_dir, "table_4_5_sensor_count", "sensor_count")
    save_metrics_bundle(layout_metrics, table_dir, "table_4_5_sensor_layout", "sensor_layout")

    count_labels = list(count_scenarios.keys())
    plot_grouped_bars(
        categories=count_labels,
        series={
            r"$U_s$": [count_metrics[label]["parameter"]["Us"]["nrmse"] for label in count_labels],
            r"$U_b$": [count_metrics[label]["parameter"]["Ub"]["nrmse"] for label in count_labels],
            r"$p$": [count_metrics[label]["parameter"]["p"]["nrmse"] for label in count_labels],
        },
        x_label="Sensor-count scenario",
        y_label="NRMSE",
        title="Parameter NRMSE vs sensor count",
        save_name="fig_4_5_parameter_nrmse_vs_sensor_count",
        fig_dir=fig_dir,
    )
    response_names = [str(v) for v in eval_dataset["output_vars"].tolist()]
    plot_grouped_bars(
        categories=count_labels,
        series={name: [count_metrics[label]["response"][name]["nrmse"] for label in count_labels] for name in response_names},
        x_label="Sensor-count scenario",
        y_label="NRMSE",
        title="Response NRMSE vs sensor count",
        save_name="fig_4_5_response_nrmse_vs_sensor_count",
        fig_dir=fig_dir,
    )

    layout_labels = list(layout_scenarios.keys())
    plot_grouped_bars(
        categories=layout_labels,
        series={
            r"$U_s$": [layout_metrics[label]["parameter"]["Us"]["nrmse"] for label in layout_labels],
            r"$U_b$": [layout_metrics[label]["parameter"]["Ub"]["nrmse"] for label in layout_labels],
            r"$p$": [layout_metrics[label]["parameter"]["p"]["nrmse"] for label in layout_labels],
        },
        x_label="Sensor-layout scenario",
        y_label="NRMSE",
        title="Parameter NRMSE vs sensor layout",
        save_name="fig_4_5_parameter_nrmse_vs_sensor_layout",
        fig_dir=fig_dir,
    )
    plot_grouped_bars(
        categories=layout_labels,
        series={name: [layout_metrics[label]["response"][name]["nrmse"] for label in layout_labels] for name in response_names},
        x_label="Sensor-layout scenario",
        y_label="NRMSE",
        title="Response NRMSE vs sensor layout",
        save_name="fig_4_5_response_nrmse_vs_sensor_layout",
        fig_dir=fig_dir,
    )

    output_vars = [str(v) for v in eval_dataset["output_vars"].tolist()]
    count_case_results: Dict[str, Dict[str, Any]] = {}
    layout_case_results: Dict[str, Dict[str, Any]] = {}
    for case in TYPICAL_CASES:
        count_series: Dict[str, Any] = {}
        y_exact = None
        s = None
        for label in count_labels:
            tag = f"count_{label}"
            bundle = load_bmn_bundle(scenario_root / "count" / label / f"BMN_DD_encoder_{tag}.pth", decoder_ckpt_path, args.device)
            pred = predict_bmn_typical_case(bundle, cfg, case)
            y_exact = pred["y_true"]
            s = pred["s"]
            count_series[label] = pred["y_pred"]
        if y_exact is not None and s is not None:
            count_case_results[case["case_id"]] = {"s": s, "y_true": y_exact, "series_by_label": count_series}
            plot_typical_case_family(
                case_id=case["case_id"],
                desc=case["description"] + " | sensor-count study",
                s=s,
                y_exact=y_exact,
                series_by_label=count_series,
                output_vars=output_vars,
                fig_dir=fig_dir,
                save_name=f"fig_4_5_{case['case_id']}_sensor_count_family",
                show_sensor_positions=False,
            )

        layout_series: Dict[str, Any] = {}
        y_exact = None
        s = None
        for label in layout_labels:
            tag = f"layout_{label}"
            bundle = load_bmn_bundle(scenario_root / "layout" / label / f"BMN_DD_encoder_{tag}.pth", decoder_ckpt_path, args.device)
            pred = predict_bmn_typical_case(bundle, cfg, case)
            y_exact = pred["y_true"]
            s = pred["s"]
            layout_series[label] = pred["y_pred"]
        if y_exact is not None and s is not None:
            layout_case_results[case["case_id"]] = {"s": s, "y_true": y_exact, "series_by_label": layout_series}
            plot_typical_case_family(
                case_id=case["case_id"],
                desc=case["description"] + " | sensor-layout study",
                s=s,
                y_exact=y_exact,
                series_by_label=layout_series,
                output_vars=output_vars,
                fig_dir=fig_dir,
                save_name=f"fig_4_5_{case['case_id']}_sensor_layout_family",
                show_sensor_positions=False,
            )

    if count_case_results:
        plot_typical_case_family_collection(
            cases=TYPICAL_CASES,
            case_results=count_case_results,
            output_vars=output_vars,
            fig_dir=fig_dir,
            save_name="fig_4_5_sensor_count_family_all_cases",
            show_sensor_positions=False,
            title="Typical-case response comparison under different sensor counts",
        )
    if layout_case_results:
        plot_typical_case_family_collection(
            cases=TYPICAL_CASES,
            case_results=layout_case_results,
            output_vars=output_vars,
            fig_dir=fig_dir,
            save_name="fig_4_5_sensor_layout_family_all_cases",
            show_sensor_positions=False,
            title="Typical-case response comparison under different sensor layouts",
        )

    save_json({"status": "completed", "output_root": str(output_root)}, output_root / "analysis_4_5_summary.json")


if __name__ == "__main__":
    main()
