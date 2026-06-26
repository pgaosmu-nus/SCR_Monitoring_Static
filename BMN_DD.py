#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMN_DD.py

BMN-SCR-DD v0.3: Decoder-guided Encoder training for SCR static response-field inversion.

This file implements the BMN Encoder under the frozen-Decoder strategy:

    observation o -> Encoder E_psi -> parameter mu_hat
    (c, mu_hat) -> frozen Decoder D_phi* -> full response y_hat -> observation o_hat

Training loss:

    L = L_mu + lambda_observation * L_obs

where
    L_mu  = ||E_psi(o_D) - mu||^2
    L_obs = ||P(D_phi*(c, E_psi(o_D))) - o_D||^2

No order constraint and no full-response loss are used in this version.
The default `build_data` mode generates Encoder data using the frozen Decoder itself:

    (c, mu) -> D_phi*(c, mu) -> y_D -> P(y_D) -> o_D

Thus Encoder and Decoder are not jointly optimized, but Encoder training still receives
observation-consistency gradients through the frozen Decoder.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
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
    MU_NAMES,
    PARAM_NAMES,
    DenseMLP,
    StandardScaler,
    decode_fullfield_np,
    get_encoder_n_cases,
    load_config,
    load_decoder_model,
    resolve_sensor_indices,
    sample_one_case,
    set_seed,
)


# =============================================================================
# 1. Dataset utilities and Encoder model
# =============================================================================


def load_npz_dataset(path: str | Path) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


def validate_encoder_dataset_size(path: str | Path, expected_n_cases: int) -> None:
    dataset = load_npz_dataset(path)
    if "observations" not in dataset:
        raise KeyError(f"Encoder dataset is missing required key 'observations': {path}")
    actual_n_cases = int(np.asarray(dataset["observations"]).shape[0])
    if actual_n_cases != int(expected_n_cases):
        raise ValueError(
            "Encoder dataset size mismatch: "
            f"{path} contains {actual_n_cases} cases, but config expects {int(expected_n_cases)} cases. "
            "Run `python BMN_DD.py --mode build_data` (or `--mode all`) to regenerate the dataset, "
            "or pass the correct dataset path via --encoder_data."
        )


def split_indices(n: int, train_fraction: float, val_fraction: float, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = max(1, int(round(n * train_fraction)))
    n_val = int(round(n * val_fraction))
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)
    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    if val.size == 0 and test.size > 0:
        val = test[:1]
    return {"train": train, "val": val, "test": test}


def make_loader(obs: np.ndarray, mu: np.ndarray, c: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(mu, dtype=torch.float32),
            torch.tensor(c, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class BoundedMLPEncoder(nn.Module):
    """Observation-to-parameter encoder with optional bounded outputs."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        activation: str,
        dropout: float,
        use_layer_norm: bool,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_make_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, output_dim)
        self.register_buffer("lower_bounds", torch.tensor(lower_bounds, dtype=torch.float32))
        self.register_buffer("upper_bounds", torch.tensor(upper_bounds, dtype=torch.float32))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.head(features)
        span = self.upper_bounds - self.lower_bounds
        return self.lower_bounds + span * torch.sigmoid(logits)


def _build_mu_bounds_scaled(cfg: BMNConfig, mu_scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    lower_phys = np.asarray([cfg.ranges.Us_min, cfg.ranges.Ub_min, cfg.ranges.p_min], dtype=np.float32)
    upper_phys = np.asarray([cfg.ranges.Us_max, cfg.ranges.Ub_max, cfg.ranges.p_max], dtype=np.float32)
    lower_scaled = mu_scaler.transform(lower_phys[None, :])[0]
    upper_scaled = mu_scaler.transform(upper_phys[None, :])[0]
    return lower_scaled.astype(np.float32), upper_scaled.astype(np.float32)


def build_encoder_model(
    cfg: BMNConfig,
    obs_dim: int,
    mu_dim: int,
    mu_scaler: StandardScaler,
) -> nn.Module:
    arch = str(getattr(cfg.encoder_network, "architecture", "dense_mlp")).lower()
    if arch == "dense_mlp":
        return DenseMLP(
            input_dim=obs_dim,
            output_dim=mu_dim,
            hidden_dim=int(cfg.encoder_network.hidden_dim),
            num_hidden_layers=int(cfg.encoder_network.num_hidden_layers),
            activation=str(cfg.encoder_network.activation),
            dropout=float(cfg.encoder_network.dropout),
        )
    if arch == "bounded_mlp":
        bounded_output = bool(getattr(cfg.encoder_network, "bounded_output", True))
        if bounded_output:
            lower_bounds, upper_bounds = _build_mu_bounds_scaled(cfg, mu_scaler)
        else:
            lower_bounds = np.full(mu_dim, -6.0, dtype=np.float32)
            upper_bounds = np.full(mu_dim, 6.0, dtype=np.float32)
        return BoundedMLPEncoder(
            input_dim=obs_dim,
            output_dim=mu_dim,
            hidden_dim=int(cfg.encoder_network.hidden_dim),
            num_hidden_layers=int(cfg.encoder_network.num_hidden_layers),
            activation=str(cfg.encoder_network.activation),
            dropout=float(cfg.encoder_network.dropout),
            use_layer_norm=bool(getattr(cfg.encoder_network, "use_layer_norm", True)),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
    raise ValueError(f"Unsupported encoder architecture: {arch}")


def _resolve_device(requested: str) -> str:
    return requested if torch.cuda.is_available() or requested == "cpu" else "cpu"


# =============================================================================
# 2. Frozen Decoder adapter
# =============================================================================


class FrozenDecoderAdapter(nn.Module):
    """Differentiable adapter for a frozen Decoder_DD checkpoint.

    Parameters are frozen, but the forward mapping remains differentiable with
    respect to `mu`. Therefore L_obs can propagate gradients through the frozen
    Decoder into the Encoder.
    """

    def __init__(self, decoder_checkpoint_path: str | Path, device: str) -> None:
        super().__init__()
        model, checkpoint = load_decoder_model(decoder_checkpoint_path, map_location=device)
        self.model = model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.checkpoint = checkpoint
        self.input_names = list(checkpoint["input_names"])
        self.output_vars = list(checkpoint["output_vars"])
        self.x_mean = torch.tensor(checkpoint["input_scaler"]["mean"], dtype=torch.float32, device=device)
        self.x_std = torch.tensor(checkpoint["input_scaler"]["std"], dtype=torch.float32, device=device)
        self.y_mean = torch.tensor(checkpoint["output_scaler"]["mean"], dtype=torch.float32, device=device)
        self.y_std = torch.tensor(checkpoint["output_scaler"]["std"], dtype=torch.float32, device=device)

    def forward(self, s: torch.Tensor, c: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Decode full-field response.

        Parameters
        ----------
        s  : (N,) physical arc-length coordinates.
        c  : (B, 2), [Dx, ht].
        mu : (B, 3), [Us, Ub, p].

        Returns
        -------
        y : (B, N, n_output), physical-scale Decoder response.
        """
        bsz = int(c.shape[0])
        n_nodes = int(s.numel())
        params = torch.cat([c, mu], dim=1)  # [Dx, ht, Us, Ub, p]
        s_feat = s[None, :, None].repeat(bsz, 1, 1)
        p_feat = params[:, None, :].repeat(1, n_nodes, 1)
        x_in = torch.cat([s_feat, p_feat], dim=-1).reshape(bsz * n_nodes, -1)
        x_scaled = (x_in - self.x_mean) / self.x_std
        y_scaled = self.model(x_scaled)
        y_phys = y_scaled * self.y_std + self.y_mean
        return y_phys.reshape(bsz, n_nodes, -1)


# =============================================================================
# 3. Observation extraction and decoder-generated dataset
# =============================================================================


def extract_observation_torch(
    y: torch.Tensor,
    output_vars: List[str],
    observation_vars: List[str],
    sensor_indices: np.ndarray,
) -> torch.Tensor:
    var_to_col = {name: i for i, name in enumerate(output_vars)}
    required = ["x", "z", *observation_vars]
    missing = [name for name in required if name not in var_to_col]
    if missing:
        raise ValueError(f"Missing variables in Decoder output: {missing}; output_vars={output_vars}")
    obs_terms: List[torch.Tensor] = []
    obs_terms.append(y[:, -1, var_to_col["x"]])
    obs_terms.append(y[:, -1, var_to_col["z"]])
    for idx in sensor_indices:
        for var in observation_vars:
            obs_terms.append(y[:, int(idx), var_to_col[var]])
    return torch.stack(obs_terms, dim=1)


def generate_cases_for_encoder(cfg: BMNConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample admissible cases for Encoder training without calling the exact solver."""
    rng = np.random.default_rng(int(cfg.dataset.seed) + 1009)
    target_cases = get_encoder_n_cases(cfg)
    params: List[List[float]] = []
    while len(params) < target_cases:
        case = sample_one_case(rng, cfg.ranges, cfg.physical)
        params.append([case[name] for name in PARAM_NAMES])
    params_arr = np.asarray(params, dtype=np.float32)
    c_arr = params_arr[:, [PARAM_NAMES.index(name) for name in C_NAMES]].astype(np.float32)
    mu_arr = params_arr[:, [PARAM_NAMES.index(name) for name in MU_NAMES]].astype(np.float32)
    return params_arr, c_arr, mu_arr


@torch.no_grad()
def build_decoder_generated_encoder_dataset(
    cfg: BMNConfig,
    decoder_checkpoint_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Path:
    """Generate Encoder training data by the frozen Decoder.

    Data-generation route:
        sample (c, mu) -> y_D = D_phi*(c, mu) -> o_D = P(y_D)
    """
    output_dir = Path(output_dir) if output_dir is not None else Path(cfg.dataset.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(str(cfg.encoder_training.device))
    decoder_adapter = FrozenDecoderAdapter(decoder_checkpoint_path, device=device)
    output_vars = list(decoder_adapter.output_vars)

    params, c, mu = generate_cases_for_encoder(cfg)
    s_np = np.linspace(0.0, float(cfg.physical.L), int(cfg.dataset.n_nodes), dtype=np.float32)
    sensor_indices = resolve_sensor_indices(s_np, cfg.dataset)
    s_t = torch.tensor(s_np, dtype=torch.float32, device=device)

    y_chunks: List[np.ndarray] = []
    obs_chunks: List[np.ndarray] = []
    batch_size = max(1, int(cfg.encoder_training.batch_size))
    t0 = time.time()

    print("=" * 88)
    print("Building Encoder dataset from frozen Decoder")
    print(f"Decoder checkpoint: {decoder_checkpoint_path}")
    print(f"Cases             : {len(mu)}")
    print(f"Nodes/case        : {len(s_np)}")
    print(f"Sensor indices    : {sensor_indices.tolist()}")
    print(f"Observation vars  : {list(cfg.dataset.observation_vars)}")
    print(f"Device            : {device}")
    print("=" * 88)

    for start in range(0, len(mu), batch_size):
        end = min(start + batch_size, len(mu))
        c_b = torch.tensor(c[start:end], dtype=torch.float32, device=device)
        mu_b = torch.tensor(mu[start:end], dtype=torch.float32, device=device)
        y_b = decoder_adapter(s_t, c_b, mu_b)
        obs_b = extract_observation_torch(y_b, output_vars, list(cfg.dataset.observation_vars), sensor_indices)
        y_chunks.append(y_b.detach().cpu().numpy().astype(np.float32))
        obs_chunks.append(obs_b.detach().cpu().numpy().astype(np.float32))
        if end == len(mu) or end % max(batch_size * 10, 1) == 0:
            print(f"  generated {end:5d}/{len(mu)} | elapsed={time.time() - t0:.1f}s")

    y = np.concatenate(y_chunks, axis=0).astype(np.float32)
    obs = np.concatenate(obs_chunks, axis=0).astype(np.float32)
    obs_names = ["x_top", "z_top"]
    for idx in sensor_indices:
        for var in cfg.dataset.observation_vars:
            obs_names.append(f"{var}_idx{int(idx)}")

    save_path = output_dir / cfg.dataset.encoder_dataset_filename
    np.savez_compressed(
        save_path,
        observations=obs,
        c=c,
        mu=mu,
        y=y,
        params=params,
        s=s_np,
        sensor_indices=sensor_indices.astype(np.int64),
        sensor_s=s_np[sensor_indices].astype(np.float32),
        obs_names=np.asarray(obs_names),
        output_vars=np.asarray(output_vars),
        observation_vars=np.asarray(cfg.dataset.observation_vars),
        param_names=np.asarray(PARAM_NAMES),
        c_names=np.asarray(C_NAMES),
        mu_names=np.asarray(MU_NAMES),
        source="frozen_decoder",
        decoder_checkpoint_path=str(decoder_checkpoint_path),
        config_json=np.asarray(json.dumps(asdict(cfg), ensure_ascii=False)),
    )
    print("=" * 88)
    print(f"Decoder-generated Encoder dataset saved: {save_path}")
    print(f"Observations shape : {obs.shape}")
    print(f"Full response shape: {y.shape}")
    print("=" * 88)
    return save_path


# =============================================================================
# 4. Training and evaluation
# =============================================================================


def evaluate_encoder_losses(
    encoder: nn.Module,
    loader: DataLoader,
    obs_scaler: StandardScaler,
    mu_scaler: StandardScaler,
    decoder_adapter: FrozenDecoderAdapter,
    s: torch.Tensor,
    output_vars: List[str],
    observation_vars: List[str],
    sensor_indices: np.ndarray,
    response_var_indices: Sequence[int],
    response_var_scales: torch.Tensor,
    lambda_response: float,
    lambda_observation: float,
    lambda_order: float,
    device: str,
) -> Dict[str, float]:
    encoder.eval()
    mse_sum = nn.MSELoss(reduction="sum")
    obs_mean = torch.tensor(obs_scaler.mean, dtype=torch.float32, device=device)
    obs_std = torch.tensor(obs_scaler.std, dtype=torch.float32, device=device)
    mu_mean = torch.tensor(mu_scaler.mean, dtype=torch.float32, device=device)
    mu_std = torch.tensor(mu_scaler.std, dtype=torch.float32, device=device)
    sum_mu = 0.0
    sum_obs = 0.0
    sum_response = 0.0
    sum_order = 0.0
    n_mu = 0
    n_obs = 0
    n_response = 0
    n_order = 0
    with torch.no_grad():
        for obs_b_s, mu_b_s, c_b, y_b in loader:
            obs_b_s = obs_b_s.to(device)
            mu_b_s = mu_b_s.to(device)
            c_b = c_b.to(device)
            y_b = y_b.to(device)
            pred_mu_s = encoder(obs_b_s)
            sum_mu += float(mse_sum(pred_mu_s, mu_b_s).detach().cpu())
            n_mu += int(mu_b_s.numel())
            pred_mu_phys = pred_mu_s * mu_std + mu_mean
            if lambda_observation > 0.0 or (lambda_response > 0.0 and len(response_var_indices) > 0):
                y_pred = decoder_adapter(s, c_b, pred_mu_phys)
            if lambda_observation > 0.0:
                obs_pred = extract_observation_torch(y_pred, output_vars, observation_vars, sensor_indices)
                obs_pred_s = (obs_pred - obs_mean) / obs_std
                sum_obs += float(mse_sum(obs_pred_s, obs_b_s).detach().cpu())
                n_obs += int(obs_b_s.numel())
            if lambda_response > 0.0 and len(response_var_indices) > 0:
                y_pred_sel = y_pred[:, :, list(response_var_indices)]
                y_true_sel = y_b[:, :, list(response_var_indices)]
                diff_s = (y_pred_sel - y_true_sel) / response_var_scales.view(1, 1, -1)
                sum_response += float(torch.sum(diff_s**2).detach().cpu())
                n_response += int(diff_s.numel())
            if lambda_order > 0.0:
                order_violation = torch.relu(pred_mu_phys[:, 1] - pred_mu_phys[:, 0])
                sum_order += float(torch.sum(order_violation**2).detach().cpu())
                n_order += int(order_violation.numel())
    mu_loss = sum_mu / max(n_mu, 1)
    obs_loss = sum_obs / max(n_obs, 1) if lambda_observation > 0.0 else 0.0
    response_loss = sum_response / max(n_response, 1) if lambda_response > 0.0 and len(response_var_indices) > 0 else 0.0
    order_loss = sum_order / max(n_order, 1) if lambda_order > 0.0 else 0.0
    return {
        "mu": mu_loss,
        "observation": obs_loss,
        "response": response_loss,
        "order": order_loss,
        "total": mu_loss + float(lambda_response) * response_loss + float(lambda_observation) * obs_loss + float(lambda_order) * order_loss,
    }


def _resolve_response_var_indices(output_vars: Sequence[str], requested_vars: Sequence[str]) -> List[int]:
    var_to_idx = {name: i for i, name in enumerate(output_vars)}
    indices: List[int] = []
    for name in requested_vars:
        if name not in var_to_idx:
            raise ValueError(f"Response loss variable {name!r} is absent from output_vars={list(output_vars)}")
        indices.append(var_to_idx[name])
    return indices


def _build_response_var_scales(
    y_train: np.ndarray,
    response_var_indices: Sequence[int],
    scale_floor: float,
    device: str,
) -> torch.Tensor:
    if len(response_var_indices) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    scales = np.std(y_train[:, :, list(response_var_indices)], axis=(0, 1)).astype(np.float32)
    scales = np.maximum(scales, float(scale_floor))
    return torch.tensor(scales, dtype=torch.float32, device=device)


def _grad_norm(loss: torch.Tensor, params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [torch.sum(g.detach() ** 2) for g in grads if g is not None]
    if not norms:
        return torch.zeros((), dtype=loss.dtype, device=loss.device)
    return torch.sqrt(torch.stack(norms).sum())


def compute_test_metrics(
    encoder: nn.Module,
    dataset: Dict[str, Any],
    test_idx: np.ndarray,
    obs_scaler: StandardScaler,
    mu_scaler: StandardScaler,
    decoder_adapter: FrozenDecoderAdapter,
    device: str,
) -> Dict[str, Any]:
    obs = np.asarray(dataset["observations"], dtype=np.float32)[test_idx]
    mu_true = np.asarray(dataset["mu"], dtype=np.float32)[test_idx]
    c = np.asarray(dataset["c"], dtype=np.float32)[test_idx]
    y_true = np.asarray(dataset["y"], dtype=np.float32)[test_idx]
    s = torch.tensor(np.asarray(dataset["s"], dtype=np.float32), dtype=torch.float32, device=device)
    encoder.eval()
    with torch.no_grad():
        obs_s = torch.tensor(obs_scaler.transform(obs), dtype=torch.float32, device=device)
        mu_pred_s = encoder(obs_s).detach().cpu().numpy()
        mu_pred = mu_scaler.inverse_transform(mu_pred_s)
        y_pred = decoder_adapter(
            s,
            torch.tensor(c, dtype=torch.float32, device=device),
            torch.tensor(mu_pred, dtype=torch.float32, device=device),
        ).detach().cpu().numpy()
    metrics: Dict[str, Any] = {}
    for i, name in enumerate(MU_NAMES):
        diff = mu_pred[:, i] - mu_true[:, i]
        metrics[f"rmse_{name}"] = float(np.sqrt(np.mean(diff**2)))
        metrics[f"mae_{name}"] = float(np.mean(np.abs(diff)))
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    for j, name in enumerate(output_vars):
        diff = y_pred[:, :, j] - y_true[:, :, j]
        metrics[f"rmse_y_{name}"] = float(np.sqrt(np.mean(diff**2)))
    metrics["n_test"] = int(len(test_idx))
    return metrics


def train_encoder(
    cfg: BMNConfig,
    encoder_dataset_path: str | Path,
    decoder_checkpoint_path: str | Path,
    output_dir: Optional[str | Path] = None,
) -> Path:
    output_dir = Path(output_dir) if output_dir is not None else Path(cfg.dataset.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_npz_dataset(encoder_dataset_path)
    obs = np.asarray(dataset["observations"], dtype=np.float32)
    mu = np.asarray(dataset["mu"], dtype=np.float32)
    c = np.asarray(dataset["c"], dtype=np.float32)
    y = np.asarray(dataset["y"], dtype=np.float32)
    s_np = np.asarray(dataset["s"], dtype=np.float32)
    output_vars = [str(v) for v in dataset["output_vars"].tolist()]
    observation_vars = [str(v) for v in dataset["observation_vars"].tolist()]
    sensor_indices = np.asarray(dataset["sensor_indices"], dtype=np.int64)

    splits = split_indices(len(obs), float(cfg.dataset.train_fraction), float(cfg.dataset.val_fraction), int(cfg.encoder_training.seed))
    obs_scaler = StandardScaler().fit(obs[splits["train"]])
    mu_scaler = StandardScaler().fit(mu[splits["train"]])
    obs_s = obs_scaler.transform(obs)
    mu_s = mu_scaler.transform(mu)

    train_loader = make_loader(obs_s[splits["train"]], mu_s[splits["train"]], c[splits["train"]], y[splits["train"]], int(cfg.encoder_training.batch_size), True)
    val_loader = make_loader(obs_s[splits["val"]], mu_s[splits["val"]], c[splits["val"]], y[splits["val"]], int(cfg.encoder_training.batch_size), False)

    device = _resolve_device(str(cfg.encoder_training.device))
    set_seed(int(cfg.encoder_training.seed))
    lambda_obs = float(getattr(cfg.encoder_training, "lambda_observation", 0.1))
    if lambda_obs < 0.0:
        lambda_obs = 0.0
    lambda_resp = float(getattr(cfg.encoder_training, "lambda_response", 0.0))
    if lambda_resp < 0.0:
        lambda_resp = 0.0
    lambda_order = float(getattr(cfg.encoder_training, "lambda_order", 0.0))
    if lambda_order < 0.0:
        lambda_order = 0.0
    response_stage2_start_epoch = max(1, int(getattr(cfg.encoder_training, "response_stage2_start_epoch", 1)))
    response_var_indices = _resolve_response_var_indices(
        output_vars,
        list(getattr(cfg.encoder_training, "response_loss_vars", ["theta", "T", "M"])),
    ) if lambda_resp > 0.0 else []
    response_var_scales = _build_response_var_scales(
        y[splits["train"]],
        response_var_indices,
        float(getattr(cfg.encoder_training, "response_scale_floor", 1.0e-6)),
        device,
    )
    response_grad_target_ratio = max(0.0, float(getattr(cfg.encoder_training, "response_grad_target_ratio", 0.2)))
    response_lambda_ema = float(getattr(cfg.encoder_training, "response_lambda_ema", 0.9))
    response_lambda_ema = min(max(response_lambda_ema, 0.0), 0.9999)
    response_lambda_min = max(0.0, float(getattr(cfg.encoder_training, "response_lambda_min", 1.0e-4)))
    response_lambda_max = max(response_lambda_min, float(getattr(cfg.encoder_training, "response_lambda_max", 10.0)))
    response_bound_growth = max(1.1, float(getattr(cfg.encoder_training, "response_bound_growth", 2.0)))
    response_bound_saturation_steps = max(1, int(getattr(cfg.encoder_training, "response_bound_saturation_steps", 100)))
    lambda_resp_dynamic = response_lambda_min if lambda_resp > 0.0 else 0.0
    resp_sat_max_steps = 0
    resp_sat_min_steps = 0

    obs_mean = torch.tensor(obs_scaler.mean, dtype=torch.float32, device=device)
    obs_std = torch.tensor(obs_scaler.std, dtype=torch.float32, device=device)
    mu_mean = torch.tensor(mu_scaler.mean, dtype=torch.float32, device=device)
    mu_std = torch.tensor(mu_scaler.std, dtype=torch.float32, device=device)
    encoder = build_encoder_model(cfg, obs_dim=obs.shape[1], mu_dim=mu.shape[1], mu_scaler=mu_scaler).to(device)
    decoder_adapter = FrozenDecoderAdapter(decoder_checkpoint_path, device=device)
    s = torch.tensor(s_np, dtype=torch.float32, device=device)

    opt = optim.Adam(encoder.parameters(), lr=float(cfg.encoder_training.lr), weight_decay=float(cfg.encoder_training.weight_decay))
    mse = nn.MSELoss()
    history: Dict[str, List[float]] = {
        "train_total": [],
        "train_mu": [],
        "train_field": [],
        "train_response": [],
        "train_observation": [],
        "train_order": [],
        "val_total": [],
        "val_mu": [],
        "val_field": [],
        "val_response": [],
        "val_observation": [],
        "val_order": [],
        "lambda_field": [],
        "lambda_response": [],
        "lambda_response_min": [],
        "lambda_response_max": [],
        "lr": [],
    }
    best_train_total = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    print("=" * 88)
    print("Training BMN_DD Encoder with frozen-Decoder observation consistency")
    print(f"Encoder data : {encoder_dataset_path}")
    print(f"Data source  : {str(dataset.get('source', 'unknown'))}")
    print(f"Decoder ckpt : {decoder_checkpoint_path}")
    print(f"Train/Val/Test cases: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Device       : {device}")
    print(f"Architecture : {str(getattr(cfg.encoder_network, 'architecture', 'dense_mlp'))}")
    print(f"Loss         : L = L_mu + lambda_field * L_field + {lambda_obs:g} * L_obs + {lambda_order:g} * L_order")
    if lambda_resp > 0.0:
        print(f"Response vars : {[output_vars[i] for i in response_var_indices]}")
        print(f"Stage-2 epoch : {response_stage2_start_epoch}")
        print(f"Field lambda  : min={response_lambda_min:g}, max={response_lambda_max:g}, target_grad_ratio={response_grad_target_ratio:g}")
    print("=" * 88)

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]

    for epoch in range(1, int(cfg.encoder_training.epochs) + 1):
        encoder.train()
        sum_total = 0.0
        sum_mu = 0.0
        sum_response = 0.0
        sum_obs = 0.0
        sum_order = 0.0
        n_batches = 0
        lambda_resp_epoch = 0.0
        for obs_b_s, mu_b_s, c_b, y_b in train_loader:
            obs_b_s = obs_b_s.to(device)
            mu_b_s = mu_b_s.to(device)
            c_b = c_b.to(device)
            y_b = y_b.to(device)
            opt.zero_grad(set_to_none=True)
            pred_mu_s = encoder(obs_b_s)
            loss_mu = mse(pred_mu_s, mu_b_s)

            pred_mu_phys = pred_mu_s * mu_std + mu_mean
            y_pred = decoder_adapter(s, c_b, pred_mu_phys)
            if lambda_resp > 0.0 and epoch >= response_stage2_start_epoch and len(response_var_indices) > 0:
                y_pred_sel = y_pred[:, :, response_var_indices]
                y_true_sel = y_b[:, :, response_var_indices]
                diff_s = (y_pred_sel - y_true_sel) / response_var_scales.view(1, 1, -1)
                loss_resp = torch.mean(diff_s**2)
                grad_mu = _grad_norm(loss_mu, encoder_params)
                grad_resp = _grad_norm(loss_resp, encoder_params)
                if float(grad_resp.detach().cpu()) > 0.0:
                    candidate = response_grad_target_ratio * float((grad_mu / (grad_resp + 1.0e-12)).detach().cpu())
                else:
                    candidate = response_lambda_max
                lambda_resp_dynamic = response_lambda_ema * lambda_resp_dynamic + (1.0 - response_lambda_ema) * candidate
                if lambda_resp_dynamic >= response_lambda_max:
                    resp_sat_max_steps += 1
                    resp_sat_min_steps = 0
                elif lambda_resp_dynamic <= response_lambda_min:
                    resp_sat_min_steps += 1
                    resp_sat_max_steps = 0
                else:
                    resp_sat_max_steps = 0
                    resp_sat_min_steps = 0
                if resp_sat_max_steps >= response_bound_saturation_steps:
                    response_lambda_max *= response_bound_growth
                    resp_sat_max_steps = 0
                if resp_sat_min_steps >= response_bound_saturation_steps:
                    response_lambda_min /= response_bound_growth
                    resp_sat_min_steps = 0
                response_lambda_max = max(response_lambda_max, response_lambda_min)
                lambda_resp_batch = min(max(lambda_resp_dynamic, response_lambda_min), response_lambda_max)
            else:
                loss_resp = torch.zeros((), dtype=torch.float32, device=device)
                lambda_resp_batch = 0.0
            if lambda_obs > 0.0:
                obs_pred = extract_observation_torch(y_pred, output_vars, observation_vars, sensor_indices)
                obs_pred_s = (obs_pred - obs_mean) / obs_std
                loss_obs = mse(obs_pred_s, obs_b_s)
            else:
                loss_obs = torch.zeros((), dtype=torch.float32, device=device)
            if lambda_order > 0.0:
                order_violation = torch.relu(pred_mu_phys[:, 1] - pred_mu_phys[:, 0])
                loss_order = torch.mean(order_violation**2)
            else:
                loss_order = torch.zeros((), dtype=torch.float32, device=device)

            total = loss_mu + lambda_resp_batch * loss_resp + lambda_obs * loss_obs + lambda_order * loss_order
            total.backward()
            if cfg.encoder_training.grad_clip and cfg.encoder_training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), float(cfg.encoder_training.grad_clip))
            opt.step()

            sum_total += float(total.detach().cpu())
            sum_mu += float(loss_mu.detach().cpu())
            sum_response += float(loss_resp.detach().cpu())
            sum_obs += float(loss_obs.detach().cpu())
            sum_order += float(loss_order.detach().cpu())
            lambda_resp_epoch += float(lambda_resp_batch)
            n_batches += 1

        val_losses = evaluate_encoder_losses(
            encoder=encoder,
            loader=val_loader,
            obs_scaler=obs_scaler,
            mu_scaler=mu_scaler,
            decoder_adapter=decoder_adapter,
            s=s,
            output_vars=output_vars,
            observation_vars=observation_vars,
            sensor_indices=sensor_indices,
            response_var_indices=response_var_indices,
            response_var_scales=response_var_scales,
            lambda_response=(lambda_resp_epoch / max(n_batches, 1)),
            lambda_observation=lambda_obs,
            lambda_order=lambda_order,
            device=device,
        )
        train_total = sum_total / max(n_batches, 1)
        train_mu = sum_mu / max(n_batches, 1)
        train_response = sum_response / max(n_batches, 1)
        train_obs = sum_obs / max(n_batches, 1)
        train_order = sum_order / max(n_batches, 1)
        lambda_resp_epoch = lambda_resp_epoch / max(n_batches, 1)
        history["train_total"].append(train_total)
        history["train_mu"].append(train_mu)
        history["train_field"].append(train_response)
        history["train_response"].append(train_response)
        history["train_observation"].append(train_obs)
        history["train_order"].append(train_order)
        history["val_total"].append(float(val_losses["total"]))
        history["val_mu"].append(float(val_losses["mu"]))
        history["val_field"].append(float(val_losses["response"]))
        history["val_response"].append(float(val_losses["response"]))
        history["val_observation"].append(float(val_losses["observation"]))
        history["val_order"].append(float(val_losses["order"]))
        history["lambda_field"].append(float(lambda_resp_epoch))
        history["lambda_response"].append(float(lambda_resp_epoch))
        history["lambda_response_min"].append(float(response_lambda_min))
        history["lambda_response_max"].append(float(response_lambda_max))
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        if train_total < best_train_total:
            best_train_total = float(train_total)
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % int(cfg.encoder_training.print_every) == 0:
            print(
                f"[Encoder] epoch={epoch:5d} | total={train_total:.3e} | mu={train_mu:.3e} | "
                f"field={train_response:.3e} | obs={train_obs:.3e} | order={train_order:.3e} | "
                f"val_total={val_losses['total']:.3e} | "
                f"val_mu={val_losses['mu']:.3e} | val_obs={val_losses['observation']:.3e} | "
                f"val_field={val_losses['response']:.3e} | val_order={val_losses['order']:.3e} | "
                f"lambda_field={lambda_resp_epoch:.3e} [{response_lambda_min:.1e}, {response_lambda_max:.1e}] | "
                f"best_train_total={best_train_total:.3e} | elapsed={time.time() - t0:.1f}s"
            )
        if cfg.encoder_training.patience > 0 and no_improve >= int(cfg.encoder_training.patience):
            print(f"[Encoder] early stopping at epoch {epoch}; best train_total={best_train_total:.3e}")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    metrics = compute_test_metrics(encoder, dataset, splits["test"], obs_scaler, mu_scaler, decoder_adapter, device) if len(splits["test"]) > 0 else {}
    ckpt_path = output_dir / cfg.encoder_training.model_filename
    checkpoint = {
        "version": "BMN-SCR-DD v0.3 Encoder",
        "training_scheme": "frozen_decoder_guided: L_mu + lambda_field * L_field + lambda_observation * L_obs",
        "lambda_field": history["lambda_field"][-1] if history["lambda_field"] else 0.0,
        "lambda_response": history["lambda_response"][-1] if history["lambda_response"] else 0.0,
        "lambda_observation": lambda_obs,
        "model_state_dict": encoder.state_dict(),
        "config": asdict(cfg),
        "obs_scaler": obs_scaler.to_dict(),
        "mu_scaler": mu_scaler.to_dict(),
        "obs_names": [str(v) for v in dataset["obs_names"].tolist()],
        "c_names": C_NAMES,
        "mu_names": MU_NAMES,
        "sensor_indices": sensor_indices.tolist(),
        "sensor_s": np.asarray(dataset["sensor_s"], dtype=float).tolist(),
        "observation_vars": observation_vars,
        "output_vars": output_vars,
        "decoder_checkpoint_path": str(decoder_checkpoint_path),
        "encoder_dataset_path": str(encoder_dataset_path),
        "splits": {k: v.tolist() for k, v in splits.items()},
        "best_train_total_loss": best_train_total,
        "test_metrics": metrics,
    }
    torch.save(checkpoint, ckpt_path)
    with open(output_dir / cfg.encoder_training.history_filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "BMN_DD_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("=" * 88)
    print(f"BMN_DD encoder training finished: {ckpt_path}")
    print(f"Test metrics: {json.dumps(metrics, indent=2)}")
    print("=" * 88)
    return ckpt_path


# =============================================================================
# 5. Inference API
# =============================================================================


def load_encoder_model(encoder_checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(encoder_checkpoint_path, map_location=map_location)
    cfg = BMNConfig()
    cfg = cfg if "config" not in ckpt else cfg
    if "config" in ckpt:
        cfg_json = ckpt["config"]
        cfg.encoder_network.architecture = str(cfg_json["encoder_network"].get("architecture", cfg.encoder_network.architecture))
        cfg.encoder_network.hidden_dim = int(cfg_json["encoder_network"]["hidden_dim"])
        cfg.encoder_network.num_hidden_layers = int(cfg_json["encoder_network"]["num_hidden_layers"])
        cfg.encoder_network.activation = str(cfg_json["encoder_network"]["activation"])
        cfg.encoder_network.dropout = float(cfg_json["encoder_network"].get("dropout", cfg.encoder_network.dropout))
        cfg.encoder_network.use_layer_norm = bool(cfg_json["encoder_network"].get("use_layer_norm", cfg.encoder_network.use_layer_norm))
        cfg.encoder_network.bounded_output = bool(cfg_json["encoder_network"].get("bounded_output", cfg.encoder_network.bounded_output))
        cfg.ranges.Us_min = float(cfg_json["ranges"].get("Us_min", cfg.ranges.Us_min))
        cfg.ranges.Us_max = float(cfg_json["ranges"].get("Us_max", cfg.ranges.Us_max))
        cfg.ranges.Ub_min = float(cfg_json["ranges"].get("Ub_min", cfg.ranges.Ub_min))
        cfg.ranges.Ub_max = float(cfg_json["ranges"].get("Ub_max", cfg.ranges.Ub_max))
        cfg.ranges.p_min = float(cfg_json["ranges"].get("p_min", cfg.ranges.p_min))
        cfg.ranges.p_max = float(cfg_json["ranges"].get("p_max", cfg.ranges.p_max))
    mu_scaler = StandardScaler.from_dict(ckpt["mu_scaler"])
    model = build_encoder_model(
        cfg=cfg,
        obs_dim=len(ckpt["obs_names"]),
        mu_dim=len(ckpt["mu_names"]),
        mu_scaler=mu_scaler,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def predict_from_observation(
    observation: np.ndarray,
    c: np.ndarray,
    s: np.ndarray,
    encoder_checkpoint_path: str | Path,
    decoder_checkpoint_path: Optional[str | Path] = None,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    encoder, eckpt = load_encoder_model(encoder_checkpoint_path, map_location=device)
    encoder = encoder.to(device)
    obs_scaler = StandardScaler.from_dict(eckpt["obs_scaler"])
    mu_scaler = StandardScaler.from_dict(eckpt["mu_scaler"])
    decoder_checkpoint_path = decoder_checkpoint_path or eckpt["decoder_checkpoint_path"]
    decoder_model, dckpt = load_decoder_model(decoder_checkpoint_path, map_location=device)
    obs_s = obs_scaler.transform(np.asarray(observation, dtype=np.float32)[None, :])
    with torch.no_grad():
        mu_s = encoder(torch.tensor(obs_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    mu = mu_scaler.inverse_transform(mu_s)[0]
    y = decode_fullfield_np(decoder_model, dckpt, np.asarray(s, dtype=np.float32), np.asarray(c, dtype=np.float32), mu, device=device)
    return {"mu": mu, "y": y, "output_vars": np.asarray(dckpt["output_vars"])}


# =============================================================================
# 6. CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="BMN-SCR-DD v0.3 frozen-Decoder-guided Encoder training")
    parser.add_argument("--config", type=str, default="para_config.json", help="Path to para_config.json")
    parser.add_argument("--encoder_data", type=str, default=None, help="Path to bmn_encoder_dataset.npz")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Path to Decoder_DD_model.pth")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "build_data", "all"], help="Execution mode")
    args = parser.parse_args()

    cfg = load_config(args.config) if Path(args.config).exists() else BMNConfig()
    output_dir = Path(args.output_dir) if args.output_dir is not None else Path(cfg.dataset.output_dir)
    decoder_ckpt = Path(args.decoder_ckpt) if args.decoder_ckpt is not None else output_dir / cfg.decoder_training.model_filename
    encoder_data = Path(args.encoder_data) if args.encoder_data is not None else output_dir / cfg.dataset.encoder_dataset_filename

    if args.mode in {"build_data", "all"}:
        if not decoder_ckpt.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt}. Train Decoder first or pass --decoder_ckpt.")
        encoder_data = build_decoder_generated_encoder_dataset(cfg, decoder_ckpt, output_dir=output_dir)

    if args.mode in {"train", "all"}:
        if not encoder_data.exists():
            raise FileNotFoundError(
                f"Encoder dataset not found: {encoder_data}. Run `python BMN_DD.py --mode build_data` first, "
                "or pass --encoder_data."
            )
        validate_encoder_dataset_size(encoder_data, get_encoder_n_cases(cfg))
        if not decoder_ckpt.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt}. Train Decoder first or pass --decoder_ckpt.")
        train_encoder(cfg, encoder_data, decoder_ckpt, output_dir=output_dir)


if __name__ == "__main__":
    main()
