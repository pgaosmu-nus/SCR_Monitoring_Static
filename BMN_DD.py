#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMN_DD.py

BMN-SCR-DD v0.2: bidirectional mapping network for SCR static response-field inversion.

This file implements the Encoder part of the proposed framework:

    sparse observation o  ->  global parameters mu_hat  ->  frozen Decoder  ->  full response y_hat

Default definitions are consistent with Decoder_DD.py:
- observation o: [x_top, z_top, T_i, M_i, theta_i, ...]
- known condition c: [Dx, ht]
- parameter mu: [Us, Ub, p]
- frozen Decoder input: [s, Dx, ht, Us, Ub, p]

The baseline training follows the paper idea: Encoder is trained by parameter-space supervision.
The default Encoder is a PIBAE-inspired bounded MLP: it uses LayerNorm/GELU hidden blocks and
a sigmoid-bounded physical-parameter head. Optional decoder-assisted response/observation losses are
included through config switches, but default to 0.0.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

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
    build_encoder_dataset,
    config_from_dict,
    decode_fullfield_np,
    extract_observations_from_fields,
    load_config,
    load_decoder_model,
    resolve_sensor_indices,
    save_config,
    set_seed,
)


# =============================================================================
# 1. Dataset and model loading
# =============================================================================


def load_npz_dataset(path: str | Path) -> Dict[str, Any]:
    npz = np.load(path, allow_pickle=True)
    return {key: npz[key] for key in npz.files}


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
    if val.size == 0:
        val = test[: min(1, test.size)]
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


class BoundedPIBAEEncoder(nn.Module):
    """PIBAE-inspired bounded Encoder for SCR parameter inversion.

    Differences from the uploaded PIBAE_Encoder.py:
    1. The observation dimension is not hard-coded; it follows the BMN dataset.
    2. Inputs still use the dataset-derived StandardScaler, rather than fixed T/M constants.
    3. The network outputs physical parameters inside prescribed ranges via a sigmoid head,
       then converts them to the same scaled parameter space used by the training loss.
    """

    def __init__(
        self,
        obs_dim: int,
        mu_names: Sequence[str],
        mu_bounds: Dict[str, Tuple[float, float]],
        mu_scaler: StandardScaler,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.mu_names = list(mu_names)
        lows = [float(mu_bounds[name][0]) for name in self.mu_names]
        highs = [float(mu_bounds[name][1]) for name in self.mu_names]
        self.register_buffer("mu_low", torch.tensor(lows, dtype=torch.float32))
        self.register_buffer("mu_high", torch.tensor(highs, dtype=torch.float32))
        self.register_buffer("mu_mean", torch.tensor(mu_scaler.mean, dtype=torch.float32))
        self.register_buffer("mu_std", torch.tensor(mu_scaler.std, dtype=torch.float32))

        layers: List[nn.Module] = []
        in_dim = obs_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_make_local_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, len(self.mu_names)))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_physical(self, x_scaled: torch.Tensor) -> torch.Tensor:
        raw = self.net(x_scaled)
        unit = torch.sigmoid(raw)
        return self.mu_low + unit * (self.mu_high - self.mu_low)

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        mu_phys = self.forward_physical(x_scaled)
        return (mu_phys - self.mu_mean) / self.mu_std


def _make_local_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def get_mu_bounds_from_config(cfg: BMNConfig) -> Dict[str, Tuple[float, float]]:
    return {
        "Us": (float(cfg.ranges.Us_min), float(cfg.ranges.Us_max)),
        "Ub": (float(cfg.ranges.Ub_min), float(cfg.ranges.Ub_max)),
        "p": (float(cfg.ranges.p_min), float(cfg.ranges.p_max)),
    }


def build_encoder_model(cfg: BMNConfig, obs_dim: int, mu_dim: int, mu_scaler: Optional[StandardScaler] = None) -> nn.Module:
    arch = str(getattr(cfg.encoder_network, "architecture", "dense_mlp")).lower()
    if arch in {"bounded_mlp", "pibae_mlp", "bounded"}:
        if mu_scaler is None:
            raise ValueError("mu_scaler is required for bounded_mlp Encoder.")
        if mu_dim != len(MU_NAMES):
            raise ValueError("bounded_mlp currently expects mu=[Us, Ub, p].")
        return BoundedPIBAEEncoder(
            obs_dim=obs_dim,
            mu_names=MU_NAMES,
            mu_bounds=get_mu_bounds_from_config(cfg),
            mu_scaler=mu_scaler,
            hidden_dim=cfg.encoder_network.hidden_dim,
            num_hidden_layers=cfg.encoder_network.num_hidden_layers,
            activation=cfg.encoder_network.activation,
            dropout=cfg.encoder_network.dropout,
            use_layer_norm=bool(getattr(cfg.encoder_network, "use_layer_norm", True)),
        )
    return DenseMLP(
        input_dim=obs_dim,
        output_dim=mu_dim,
        hidden_dim=cfg.encoder_network.hidden_dim,
        num_hidden_layers=cfg.encoder_network.num_hidden_layers,
        activation=cfg.encoder_network.activation,
        dropout=cfg.encoder_network.dropout,
    )


# =============================================================================
# 2. Differentiable frozen Decoder adapter
# =============================================================================


class FrozenDecoderAdapter(nn.Module):
    """Torch adapter for frozen Decoder_DD checkpoint.

    It receives physical-scale c and mu tensors and returns physical-scale y tensor.
    The transformation through the Decoder remains differentiable with respect to mu,
    so optional response losses can supervise the Encoder through the frozen Decoder.
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
        c  : (B, 2) [Dx, ht].
        mu : (B, 3) [Us, Ub, p].

        Returns
        -------
        y  : (B, N, n_output) physical-scale response.
        """
        bsz = c.shape[0]
        n_nodes = s.numel()
        params = torch.cat([c, mu], dim=1)  # [Dx, ht, Us, Ub, p]
        s_feat = s[None, :, None].repeat(bsz, 1, 1)
        p_feat = params[:, None, :].repeat(1, n_nodes, 1)
        x_in = torch.cat([s_feat, p_feat], dim=-1).reshape(bsz * n_nodes, -1)
        x_scaled = (x_in - self.x_mean) / self.x_std
        y_scaled = self.model(x_scaled)
        y_phys = y_scaled * self.y_std + self.y_mean
        return y_phys.reshape(bsz, n_nodes, -1)


# =============================================================================
# 3. Observation extraction from decoded fields
# =============================================================================


def extract_observation_torch(y: torch.Tensor, output_vars: List[str], observation_vars: List[str], sensor_indices: np.ndarray) -> torch.Tensor:
    var_to_col = {name: i for i, name in enumerate(output_vars)}
    obs_terms: List[torch.Tensor] = []
    obs_terms.append(y[:, -1, var_to_col["x"]])
    obs_terms.append(y[:, -1, var_to_col["z"]])
    for idx in sensor_indices:
        for var in observation_vars:
            obs_terms.append(y[:, int(idx), var_to_col[var]])
    return torch.stack(obs_terms, dim=1)


# =============================================================================
# 4. Training and evaluation
# =============================================================================


def evaluate_encoder(
    encoder: nn.Module,
    loader: DataLoader,
    obs_scaler: StandardScaler,
    mu_scaler: StandardScaler,
    device: str,
) -> float:
    encoder.eval()
    mse = nn.MSELoss(reduction="sum")
    total = 0.0
    count = 0
    with torch.no_grad():
        for obs_s, mu_s, _, _ in loader:
            obs_s = obs_s.to(device)
            mu_s = mu_s.to(device)
            pred = encoder(obs_s)
            total += float(mse(pred, mu_s).detach().cpu())
            count += int(mu_s.numel())
    return total / max(count, 1)


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

    splits = split_indices(len(obs), cfg.dataset.train_fraction, cfg.dataset.val_fraction, cfg.encoder_training.seed)
    obs_scaler = StandardScaler().fit(obs[splits["train"]])
    mu_scaler = StandardScaler().fit(mu[splits["train"]])
    obs_s = obs_scaler.transform(obs)
    mu_s = mu_scaler.transform(mu)

    train_loader = make_loader(obs_s[splits["train"]], mu_s[splits["train"]], c[splits["train"]], y[splits["train"]], cfg.encoder_training.batch_size, True)
    val_loader = make_loader(obs_s[splits["val"]], mu_s[splits["val"]], c[splits["val"]], y[splits["val"]], cfg.encoder_training.batch_size, False)

    device = cfg.encoder_training.device if torch.cuda.is_available() or cfg.encoder_training.device == "cpu" else "cpu"
    set_seed(cfg.encoder_training.seed)
    encoder = build_encoder_model(cfg, obs_dim=obs.shape[1], mu_dim=mu.shape[1], mu_scaler=mu_scaler).to(device)
    decoder_adapter = FrozenDecoderAdapter(decoder_checkpoint_path, device=device)
    s = torch.tensor(s_np, dtype=torch.float32, device=device)

    opt = optim.Adam(encoder.parameters(), lr=cfg.encoder_training.lr, weight_decay=cfg.encoder_training.weight_decay)
    mse = nn.MSELoss()
    history: Dict[str, List[float]] = {"train_total": [], "train_mu": [], "train_response": [], "train_observation": [], "train_order": [], "val_mu": [], "lr": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()

    y_scaler_dict = decoder_adapter.checkpoint["output_scaler"]
    y_mean = torch.tensor(y_scaler_dict["mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(y_scaler_dict["std"], dtype=torch.float32, device=device)
    obs_mean = torch.tensor(obs_scaler.mean, dtype=torch.float32, device=device)
    obs_std = torch.tensor(obs_scaler.std, dtype=torch.float32, device=device)

    print("=" * 88)
    print("Training BMN_DD Encoder")
    print(f"Encoder data : {encoder_dataset_path}")
    print(f"Decoder ckpt : {decoder_checkpoint_path}")
    print(f"Train/Val/Test cases: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    print(f"Device       : {device}")
    print(f"lambda_response={cfg.encoder_training.lambda_response}, lambda_observation={cfg.encoder_training.lambda_observation}, lambda_order={cfg.encoder_training.lambda_order}")
    print("=" * 88)

    for epoch in range(1, cfg.encoder_training.epochs + 1):
        encoder.train()
        sum_total = sum_mu = sum_resp = sum_obs = sum_order = 0.0
        n_batches = 0
        for obs_b_s, mu_b_s, c_b, y_b in train_loader:
            obs_b_s = obs_b_s.to(device)
            mu_b_s = mu_b_s.to(device)
            c_b = c_b.to(device)
            y_b = y_b.to(device)
            opt.zero_grad(set_to_none=True)
            pred_mu_s = encoder(obs_b_s)
            loss_mu = mse(pred_mu_s, mu_b_s)
            loss_response = torch.tensor(0.0, device=device)
            loss_observation = torch.tensor(0.0, device=device)
            pred_mu_phys = pred_mu_s * torch.tensor(mu_scaler.std, dtype=torch.float32, device=device) + torch.tensor(mu_scaler.mean, dtype=torch.float32, device=device)
            loss_order = torch.mean(torch.relu(pred_mu_phys[:, 1] - pred_mu_phys[:, 0]) ** 2)
            if cfg.encoder_training.lambda_response > 0.0 or cfg.encoder_training.lambda_observation > 0.0:
                y_pred = decoder_adapter(s, c_b, pred_mu_phys)
                if cfg.encoder_training.lambda_response > 0.0:
                    y_pred_s = (y_pred - y_mean) / y_std
                    y_true_s = (y_b - y_mean) / y_std
                    loss_response = mse(y_pred_s, y_true_s)
                if cfg.encoder_training.lambda_observation > 0.0:
                    obs_pred = extract_observation_torch(y_pred, output_vars, observation_vars, sensor_indices)
                    obs_pred_s = (obs_pred - obs_mean) / obs_std
                    loss_observation = mse(obs_pred_s, obs_b_s)
            total = loss_mu + cfg.encoder_training.lambda_response * loss_response + cfg.encoder_training.lambda_observation * loss_observation + cfg.encoder_training.lambda_order * loss_order
            total.backward()
            if cfg.encoder_training.grad_clip and cfg.encoder_training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), cfg.encoder_training.grad_clip)
            opt.step()
            sum_total += float(total.detach().cpu())
            sum_mu += float(loss_mu.detach().cpu())
            sum_resp += float(loss_response.detach().cpu())
            sum_obs += float(loss_observation.detach().cpu())
            sum_order += float(loss_order.detach().cpu())
            n_batches += 1

        val_mu = evaluate_encoder(encoder, val_loader, obs_scaler, mu_scaler, device)
        history["train_total"].append(sum_total / max(n_batches, 1))
        history["train_mu"].append(sum_mu / max(n_batches, 1))
        history["train_response"].append(sum_resp / max(n_batches, 1))
        history["train_observation"].append(sum_obs / max(n_batches, 1))
        history["train_order"].append(sum_order / max(n_batches, 1))
        history["val_mu"].append(val_mu)
        history["lr"].append(float(opt.param_groups[0]["lr"]))

        if val_mu < best_val:
            best_val = val_mu
            best_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % cfg.encoder_training.print_every == 0:
            print(f"[Encoder] epoch={epoch:5d} | total={history['train_total'][-1]:.3e} | mu={history['train_mu'][-1]:.3e} | val_mu={val_mu:.3e} | best={best_val:.3e} | elapsed={time.time()-t0:.1f}s")
        if cfg.encoder_training.patience > 0 and no_improve >= cfg.encoder_training.patience:
            print(f"[Encoder] early stopping at epoch {epoch}; best val_mu={best_val:.3e}")
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)
    metrics = compute_test_metrics(encoder, dataset, splits["test"], obs_scaler, mu_scaler, decoder_adapter, device) if len(splits["test"]) > 0 else {}
    ckpt_path = output_dir / cfg.encoder_training.model_filename
    checkpoint = {
        "version": "BMN-SCR-DD v0.2 Encoder",
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
        "best_val_mu_loss": best_val,
        "test_metrics": metrics,
        "encoder_architecture": str(getattr(cfg.encoder_network, "architecture", "dense_mlp")),
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
    cfg = config_from_dict(ckpt["config"])
    # Backward compatibility: v0.1 checkpoints did not store an architecture flag and
    # used the unconstrained DenseMLP. v0.2 checkpoints store and use bounded_mlp.
    raw_net_cfg = ckpt.get("config", {}).get("encoder_network", {})
    if "architecture" not in raw_net_cfg and "encoder_architecture" not in ckpt:
        cfg.encoder_network.architecture = "dense_mlp"
    elif "encoder_architecture" in ckpt:
        cfg.encoder_network.architecture = str(ckpt["encoder_architecture"])
    mu_scaler = StandardScaler.from_dict(ckpt["mu_scaler"])
    model = build_encoder_model(
        cfg,
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
    parser = argparse.ArgumentParser(description="BMN-SCR-DD v0.2 Encoder training and inference")
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
        encoder_data = build_encoder_dataset(cfg, output_dir / cfg.dataset.full_dataset_filename)
    if args.mode in {"train", "all"}:
        if not encoder_data.exists():
            raise FileNotFoundError(f"Encoder dataset not found: {encoder_data}. Run Decoder_DD.py --mode extract_encoder_data first.")
        if not decoder_ckpt.exists():
            raise FileNotFoundError(f"Decoder checkpoint not found: {decoder_ckpt}. Run Decoder_DD.py --mode train_decoder first.")
        train_encoder(cfg, encoder_data, decoder_ckpt, output_dir=output_dir)


if __name__ == "__main__":
    main()
