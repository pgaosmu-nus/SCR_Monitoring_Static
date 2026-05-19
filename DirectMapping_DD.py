#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DirectMapping_DD.py

Section 4.4 baseline 1: direct mapping network for sparse-monitoring SCR
static-state reconstruction.

Baseline definition
-------------------
The proposed BMN route is

    observation -> Encoder -> mu_hat -> frozen Decoder -> full-field response.

This baseline removes the intermediate physical/environmental parameter vector
and directly learns the pointwise map

    [s, c, observation] -> y(s),

where c = [Dx, ht], and y(s) is normally [x, z, theta, T, M].

Recommended workflow
--------------------
1. Generate/train the original Decoder dataset first, if it does not already exist:
       python Decoder_DD.py --config para_config.json --mode all

2. Train and evaluate the direct-mapping baseline:
       python DirectMapping_DD.py --config para_config.json --mode all

3. Evaluate on a fixed common test set if you have one:
       python DirectMapping_DD.py --config para_config.json --mode evaluate \
           --checkpoint outputs/BMN_SCR_DD_outputs/DirectMapping_baseline/DirectMapping_DD_model.pth \
           --eval_dataset paper_testset_4_4_exact.npz

Outputs
-------
- DirectMapping_DD_model.pth
- DirectMapping_DD_history.json
- DirectMapping_DD_test_predictions.npz
- DirectMapping_DD_test_metrics.json
- DirectMapping_DD_response_metrics.csv
- DirectMapping_DD_feature_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Decoder_DD import (
    BMNConfig,
    C_NAMES,
    DenseMLP,
    StandardScaler,
    extract_observations_from_fields,
    load_config,
    resolve_sensor_indices,
    set_seed,
    split_case_indices,
)


# =============================================================================
# 0. USER-EDITABLE DEFAULTS
# =============================================================================

DEFAULT_CONFIG_PATH = "para_config.json"
DEFAULT_OUTPUT_SUBDIR = "DirectMapping_baseline"
DEFAULT_MODEL_FILENAME = "DirectMapping_DD_model.pth"
DEFAULT_HISTORY_FILENAME = "DirectMapping_DD_history.json"


# =============================================================================
# 1. Baseline-specific configuration
# =============================================================================


@dataclass
class DirectMappingNetworkConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 5
    activation: str = "gelu"
    dropout: float = 0.0


@dataclass
class DirectMappingTrainingConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 2000
    batch_size: int = 8192
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    print_every: int = 100
    patience: int = 300
    model_filename: str = DEFAULT_MODEL_FILENAME
    history_filename: str = DEFAULT_HISTORY_FILENAME


@dataclass
class DirectMappingConfig:
    network: DirectMappingNetworkConfig = field(default_factory=DirectMappingNetworkConfig)
    training: DirectMappingTrainingConfig = field(default_factory=DirectMappingTrainingConfig)


# =============================================================================
# 2. Small utilities
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


def tensor_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


# =============================================================================
# 3. Dataset construction
# =============================================================================


def extract_case_observations(
    y: np.ndarray,
    s: np.ndarray,
    output_vars: Sequence[str],
    observation_vars: Sequence[str],
    sensor_indices: np.ndarray,
) -> np.ndarray:
    return extract_observations_from_fields(
        y=y,
        s=s,
        output_vars=output_vars,
        observation_vars=observation_vars,
        sensor_indices=sensor_indices,
    ).astype(np.float32)


def build_case_observations(cfg: BMNConfig, dataset: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract sparse observation vectors from the full-field dataset."""
    s = np.asarray(dataset["s"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32)
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    sensor_indices = resolve_sensor_indices(s, cfg.dataset)
    obs = extract_case_observations(y, s, output_vars, cfg.dataset.observation_vars, sensor_indices)

    obs_names = ["x_top", "z_top"]
    for idx in sensor_indices:
        for var in cfg.dataset.observation_vars:
            obs_names.append(f"{var}_idx{int(idx)}")
    return obs.astype(np.float32), sensor_indices.astype(np.int64), obs_names


def build_direct_point_arrays(
    dataset: Dict[str, Any],
    observations: np.ndarray,
    case_indices: np.ndarray,
    input_mode: str = "s_c_o",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert case-level data into pointwise direct-mapping data.

    Parameters
    ----------
    input_mode:
        - "s_c_o": input = [s, Dx, ht, observation]
        - "s_o"  : input = [s, observation]
    """
    s = np.asarray(dataset["s"], dtype=np.float32)
    c = np.asarray(dataset["c"], dtype=np.float32)[case_indices]
    obs = np.asarray(observations, dtype=np.float32)[case_indices]
    y = np.asarray(dataset["y"], dtype=np.float32)[case_indices]

    n_cases, n_nodes, _ = y.shape
    s_col = np.tile(s[None, :, None], (n_cases, 1, 1))
    obs_cols = np.tile(obs[:, None, :], (1, n_nodes, 1))

    if input_mode == "s_c_o":
        c_cols = np.tile(c[:, None, :], (1, n_nodes, 1))
        x_in = np.concatenate([s_col, c_cols, obs_cols], axis=-1)
    elif input_mode == "s_o":
        x_in = np.concatenate([s_col, obs_cols], axis=-1)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    return x_in.reshape(n_cases * n_nodes, -1).astype(np.float32), y.reshape(n_cases * n_nodes, -1).astype(np.float32)


# =============================================================================
# 4. Training
# =============================================================================


def evaluate_scaled_mse(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total += float(loss_fn(pred, yb).detach().cpu())
            count += int(yb.numel())
    return total / max(count, 1)


def train_direct_mapping(
    cfg: BMNConfig,
    direct_cfg: DirectMappingConfig,
    dataset_path: str | Path,
    output_dir: str | Path,
    input_mode: str = "s_c_o",
) -> Path:
    dataset_path = Path(dataset_path)
    output_dir = ensure_dir(output_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Full-field dataset not found: {dataset_path}")

    dataset = load_npz(dataset_path)
    observations, sensor_indices, obs_names = build_case_observations(cfg, dataset)
    n_cases = int(dataset["y"].shape[0])
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]

    splits = split_case_indices(
        n_cases=n_cases,
        train_fraction=float(cfg.dataset.train_fraction),
        val_fraction=float(cfg.dataset.val_fraction),
        seed=int(direct_cfg.training.seed),
    )

    x_train, y_train = build_direct_point_arrays(dataset, observations, splits["train"], input_mode=input_mode)
    x_val, y_val = build_direct_point_arrays(dataset, observations, splits["val"], input_mode=input_mode)

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    y_train_s = y_scaler.transform(y_train)
    x_val_s = x_scaler.transform(x_val)
    y_val_s = y_scaler.transform(y_val)

    device = resolve_device(str(direct_cfg.training.device))
    set_seed(int(direct_cfg.training.seed))

    model = DenseMLP(
        input_dim=x_train_s.shape[1],
        output_dim=y_train_s.shape[1],
        hidden_dim=int(direct_cfg.network.hidden_dim),
        num_hidden_layers=int(direct_cfg.network.num_hidden_layers),
        activation=str(direct_cfg.network.activation),
        dropout=float(direct_cfg.network.dropout),
    ).to(device)

    train_loader = tensor_loader(x_train_s, y_train_s, int(direct_cfg.training.batch_size), shuffle=True)
    val_loader = tensor_loader(x_val_s, y_val_s, int(direct_cfg.training.batch_size), shuffle=False)

    opt = optim.Adam(
        model.parameters(),
        lr=float(direct_cfg.training.lr),
        weight_decay=float(direct_cfg.training.weight_decay),
    )
    loss_fn = nn.MSELoss()

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    no_improve = 0
    t0 = time.time()

    input_names = ["s"] + list(C_NAMES) + obs_names if input_mode == "s_c_o" else ["s"] + obs_names

    print("=" * 88)
    print("Training Section 4.4 direct mapping baseline")
    print(f"Mapping      : {input_names} -> {output_vars}")
    print(f"Dataset      : {dataset_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Cases        : train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Point samples: train={x_train_s.shape[0]}, val={x_val_s.shape[0]}")
    print(f"Input dim    : {x_train_s.shape[1]}")
    print(f"Sensors      : {sensor_indices.tolist()}")
    print(f"Obs vars     : {list(cfg.dataset.observation_vars)}")
    print(f"Device       : {device}")
    print("=" * 88)

    for epoch in range(1, int(direct_cfg.training.epochs) + 1):
        model.train()
        train_sum = 0.0
        n_items = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if float(direct_cfg.training.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(direct_cfg.training.grad_clip))
            opt.step()
            train_sum += float(loss.detach().cpu()) * xb.shape[0]
            n_items += int(xb.shape[0])

        train_loss = train_sum / max(n_items, 1)
        val_loss = evaluate_scaled_mse(model, val_loader, device)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % int(direct_cfg.training.print_every) == 0:
            print(
                f"[DirectMapping] epoch={epoch:5d} | train={train_loss:.3e} | "
                f"val={val_loss:.3e} | best={best_val:.3e} | elapsed={time.time()-t0:.1f}s"
            )

        if int(direct_cfg.training.patience) > 0 and no_improve >= int(direct_cfg.training.patience):
            print(f"[DirectMapping] early stopping at epoch {epoch}; best val={best_val:.3e}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = output_dir / direct_cfg.training.model_filename
    checkpoint = {
        "version": "DirectMapping-DD v0.1",
        "method": "pointwise direct mapping: [s, c, observation] -> y(s)",
        "input_mode": input_mode,
        "model_state_dict": model.state_dict(),
        "direct_mapping_config": asdict(direct_cfg),
        "input_scaler": x_scaler.to_dict(),
        "output_scaler": y_scaler.to_dict(),
        "input_names": input_names,
        "output_vars": output_vars,
        "obs_names": obs_names,
        "sensor_indices": sensor_indices.tolist(),
        "sensor_s": np.asarray(dataset["s"], dtype=np.float32)[sensor_indices].tolist(),
        "observation_vars": list(cfg.dataset.observation_vars),
        "splits": {k: v.tolist() for k, v in splits.items()},
        "dataset_path": str(dataset_path),
        "best_val_loss": float(best_val),
    }
    torch.save(checkpoint, checkpoint_path)
    save_json(history, output_dir / direct_cfg.training.history_filename)
    save_json({k: v.tolist() for k, v in splits.items()}, output_dir / "DirectMapping_DD_splits.json")

    print("=" * 88)
    print(f"Direct mapping baseline saved to: {checkpoint_path}")
    print("=" * 88)
    return checkpoint_path


# =============================================================================
# 5. Prediction and evaluation
# =============================================================================


def load_direct_model(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    net_cfg = checkpoint["direct_mapping_config"]["network"]
    model = DenseMLP(
        input_dim=len(checkpoint["input_names"]),
        output_dim=len(checkpoint["output_vars"]),
        hidden_dim=int(net_cfg["hidden_dim"]),
        num_hidden_layers=int(net_cfg["num_hidden_layers"]),
        activation=str(net_cfg["activation"]),
        dropout=float(net_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_direct_fullfield_np(
    model: nn.Module,
    checkpoint: Dict[str, Any],
    s: np.ndarray,
    c: np.ndarray,
    observation: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Predict a single full-field response."""
    x_scaler = StandardScaler.from_dict(checkpoint["input_scaler"])
    y_scaler = StandardScaler.from_dict(checkpoint["output_scaler"])
    input_mode = checkpoint.get("input_mode", "s_c_o")

    s = np.asarray(s, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    observation = np.asarray(observation, dtype=np.float32)

    if input_mode == "s_c_o":
        x_in = np.concatenate(
            [
                s[:, None],
                np.tile(c[None, :], (len(s), 1)),
                np.tile(observation[None, :], (len(s), 1)),
            ],
            axis=1,
        )
    elif input_mode == "s_o":
        x_in = np.concatenate(
            [
                s[:, None],
                np.tile(observation[None, :], (len(s), 1)),
            ],
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported input_mode in checkpoint: {input_mode}")

    x_s = x_scaler.transform(x_in)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(x_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return y_scaler.inverse_transform(pred_s)


def compute_response_metrics(y_true: np.ndarray, y_pred: np.ndarray, output_vars: Sequence[str], s: np.ndarray) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"response": {}, "features": {}}
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


def evaluate_direct_mapping(
    cfg: BMNConfig,
    checkpoint_path: str | Path,
    train_dataset_path: str | Path,
    output_dir: str | Path,
    device: str,
    eval_dataset_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    train_dataset = load_npz(train_dataset_path)
    model, checkpoint = load_direct_model(checkpoint_path, map_location=device)
    checkpoint_observation_vars = [str(v) for v in checkpoint["observation_vars"]]
    checkpoint_sensor_indices = np.asarray(checkpoint["sensor_indices"], dtype=np.int64)
    checkpoint_obs_names = [str(v) for v in checkpoint["obs_names"]]

    if eval_dataset_path is None:
        dataset = train_dataset
        test_idx = np.asarray(checkpoint["splits"]["test"], dtype=np.int64)
        if test_idx.size == 0:
            raise RuntimeError("No test cases found in checkpoint split.")
        case_indices = test_idx
        eval_name = "checkpoint_test_split"
    else:
        dataset = load_npz(eval_dataset_path)
        case_indices = np.arange(int(dataset["y"].shape[0]), dtype=np.int64)
        eval_name = str(eval_dataset_path)

    s = np.asarray(dataset["s"], dtype=np.float32)
    c = np.asarray(dataset["c"], dtype=np.float32)
    y_all = np.asarray(dataset["y"], dtype=np.float32)
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    observations = extract_case_observations(
        y=y_all,
        s=s,
        output_vars=output_vars,
        observation_vars=checkpoint_observation_vars,
        sensor_indices=checkpoint_sensor_indices,
    )
    sensor_indices = checkpoint_sensor_indices
    obs_names = checkpoint_obs_names

    y_true = y_all[case_indices]
    y_pred = np.zeros_like(y_true)
    time_per_case: List[float] = []

    print("=" * 88)
    print("Evaluating Section 4.4 direct mapping baseline")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Eval set    : {eval_name}")
    print(f"Cases       : {len(case_indices)}")
    print(f"Device      : {device}")
    print("=" * 88)

    for k, case_id in enumerate(case_indices):
        t_case = time.time()
        y_pred[k] = predict_direct_fullfield_np(
            model=model,
            checkpoint=checkpoint,
            s=s,
            c=c[case_id],
            observation=observations[case_id],
            device=device,
        )
        time_per_case.append(time.time() - t_case)

        if (k + 1) % max(1, len(case_indices) // 10) == 0:
            print(f"  evaluated {k+1}/{len(case_indices)} cases")

    metrics = compute_response_metrics(y_true, y_pred, output_vars, s)
    metrics["method"] = "DirectMapping"
    metrics["parameter"] = "not_available_for_direct_mapping"
    metrics["timing"] = {
        "n_test": int(len(case_indices)),
        "total_seconds": float(np.sum(time_per_case)),
        "mean_seconds_per_case": float(np.mean(time_per_case)),
        "median_seconds_per_case": float(np.median(time_per_case)),
    }
    metrics["eval_dataset"] = eval_name

    params = np.asarray(dataset["params"], dtype=np.float32)[case_indices] if "params" in dataset else None
    mu = np.asarray(dataset["mu"], dtype=np.float32)[case_indices] if "mu" in dataset else None

    np.savez_compressed(
        output_dir / "DirectMapping_DD_test_predictions.npz",
        s=s,
        case_indices=case_indices,
        y_true=y_true,
        y_pred=y_pred,
        observations=observations[case_indices],
        c=c[case_indices],
        params=params,
        mu=mu,
        sensor_indices=sensor_indices,
        output_vars=np.asarray(output_vars),
        obs_names=np.asarray(obs_names),
        time_per_case=np.asarray(time_per_case, dtype=np.float32),
    )
    save_json(metrics, output_dir / "DirectMapping_DD_test_metrics.json")

    response_rows: List[Dict[str, Any]] = []
    for name, vals in metrics["response"].items():
        row = {"response": name}
        row.update(vals)
        response_rows.append(row)
    save_csv(response_rows, output_dir / "DirectMapping_DD_response_metrics.csv")

    feature_rows = [{"metric": k, "value": v} for k, v in metrics.get("features", {}).items()]
    save_csv(feature_rows, output_dir / "DirectMapping_DD_feature_metrics.csv")

    print("=" * 88)
    print(f"Direct mapping evaluation saved to: {output_dir}")
    print("=" * 88)
    return metrics


# =============================================================================
# 6. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct mapping baseline for Section 4.4.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to para_config.json.")
    parser.add_argument("--dataset", type=str, default=None, help="Training full-field dataset. Default: cfg.dataset.output_dir/full_dataset_filename.")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Optional fixed common exact-solver testset for evaluation.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Default: cfg.dataset.output_dir/DirectMapping_baseline.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for evaluate mode.")
    parser.add_argument("--mode", type=str, default="all", choices=["train", "evaluate", "all"], help="Execution mode.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["tanh", "gelu", "silu", "relu"])
    parser.add_argument("--input_mode", type=str, default="s_c_o", choices=["s_c_o", "s_o"])
    args = parser.parse_args()

    raw_cfg = load_json_dict(args.config)
    cfg = load_config(args.config)
    direct_cfg = DirectMappingConfig()
    direct_block = raw_cfg.get("direct_mapping", {}) if isinstance(raw_cfg.get("direct_mapping"), dict) else {}
    network_block = direct_block.get("network", {}) if isinstance(direct_block.get("network"), dict) else {}
    training_block = direct_block.get("training", {}) if isinstance(direct_block.get("training"), dict) else {}

    for key, value in network_block.items():
        if hasattr(direct_cfg.network, key):
            setattr(direct_cfg.network, key, value)
    for key, value in training_block.items():
        if hasattr(direct_cfg.training, key):
            setattr(direct_cfg.training, key, value)

    input_mode = str(direct_block.get("input_mode", args.input_mode))
    if input_mode not in {"s_c_o", "s_o"}:
        raise ValueError(f"Unsupported direct_mapping.input_mode in config: {input_mode}")

    dataset_override = direct_block.get("dataset_path")
    output_dir_override = direct_block.get("output_dir")

    if args.device is not None:
        direct_cfg.training.device = args.device
    if args.epochs is not None:
        direct_cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        direct_cfg.training.batch_size = args.batch_size
    if args.lr is not None:
        direct_cfg.training.lr = args.lr
    if args.hidden_dim is not None:
        direct_cfg.network.hidden_dim = args.hidden_dim
    if args.num_hidden_layers is not None:
        direct_cfg.network.num_hidden_layers = args.num_hidden_layers
    if args.activation is not None:
        direct_cfg.network.activation = args.activation

    dataset_path = (
        Path(args.dataset)
        if args.dataset is not None
        else Path(dataset_override)
        if dataset_override is not None
        else Path(cfg.dataset.output_dir) / cfg.dataset.full_dataset_filename
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(output_dir_override)
        if output_dir_override is not None
        else Path(cfg.dataset.output_dir) / DEFAULT_OUTPUT_SUBDIR
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint is not None else output_dir / direct_cfg.training.model_filename
    device = resolve_device(direct_cfg.training.device)

    if args.mode in {"train", "all"}:
        checkpoint_path = train_direct_mapping(
            cfg=cfg,
            direct_cfg=direct_cfg,
            dataset_path=dataset_path,
            output_dir=output_dir,
            input_mode=input_mode,
        )

    if args.mode in {"evaluate", "all"}:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Direct mapping checkpoint not found: {checkpoint_path}")
        evaluate_direct_mapping(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            train_dataset_path=dataset_path,
            output_dir=output_dir,
            device=device,
            eval_dataset_path=args.eval_dataset,
        )


if __name__ == "__main__":
    main()
