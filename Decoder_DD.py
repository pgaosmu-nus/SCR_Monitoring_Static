#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoder_DD.py

BMN-SCR-DD v0.2: pure data-driven Decoder for SCR static response-field generation.

Role in the proposed BMN framework
----------------------------------
Decoder: (known conditions c, global parameters mu, arc-length coordinate s) -> full-field response y(s)

Default definitions
-------------------
Known conditions c       : [Dx, ht]
Global parameters mu     : [Us, Ub, p]
Decoder pointwise input  : [s, Dx, ht, Us, Ub, p]
Full-field output y(s)   : [x, z, theta, T, M] by default
Sparse observation o     : [x_top, z_top, T_i, M_i, theta_i, ...]

This file provides:
1. para_config.json generation;
2. parameter-space sampling;
3. exact-solver based full-field dataset generation;
4. pure data-supervised Decoder training;
5. sparse observation / full response dataset extraction for BMN_DD.py.

Expected companion files in the same directory:
- scr_exact_bvp_solver.py
- BMN_DD.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from scr_exact_bvp_solver import (
        solve_scr_exact,
        compute_local_resultants_from_global,
        PhysicalConfig as ExactPhysicalConfig,
        SolverConfig as ExactSolverConfig,
    )
    EXACT_SOLVER_AVAILABLE = True
    EXACT_SOLVER_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    solve_scr_exact = None
    compute_local_resultants_from_global = None
    ExactPhysicalConfig = None
    ExactSolverConfig = None
    EXACT_SOLVER_AVAILABLE = False
    EXACT_SOLVER_IMPORT_ERROR = repr(exc)


# =============================================================================
# 1. Configurations
# =============================================================================


@dataclass
class PhysicalConfig:
    D_o: float = 0.4064
    t: float = 0.0254
    E_steel: float = 2.1e11
    rho_s: float = 7850.0
    rho_w: float = 1025.0
    g: float = 9.81
    C_d: float = 1.2
    L: float = 2500.0
    water_depth: float = 1000.0
    x_bottom: float = 0.0
    k_b: float = 5.0e3

    @property
    def D_i(self) -> float:
        return self.D_o - 2.0 * self.t

    @property
    def A_s(self) -> float:
        return math.pi / 4.0 * (self.D_o**2 - self.D_i**2)

    @property
    def A_o(self) -> float:
        return math.pi / 4.0 * self.D_o**2

    @property
    def I(self) -> float:
        return math.pi / 64.0 * (self.D_o**4 - self.D_i**4)

    @property
    def EI(self) -> float:
        return self.E_steel * self.I

    @property
    def w_eff(self) -> float:
        return (self.rho_s * self.A_s - self.rho_w * self.A_o) * self.g

    @property
    def z_bed(self) -> float:
        return -self.water_depth

    @property
    def z_lower_equilibrium(self) -> float:
        return self.z_bed - self.w_eff / self.k_b


@dataclass
class ParameterRanges:
    # Environment / current parameters, corresponding to mu = [Us, Ub, p].
    Us_min: float = 0.5
    Us_max: float = 2.5
    Ub_min: float = 0.0
    Ub_max: float = 0.8
    p_min: float = 1.0 / 7.0
    p_max: float = 1.0 / 3.0

    # Known top-geometry conditions, corresponding to c = [Dx, ht].
    Dx_min: float = 1700.0
    Dx_max: float = 1900.0
    ht_min: float = -10.0
    ht_max: float = 10.0


@dataclass
class ExactSolverOptions:
    use_fast_solver_first: bool = True
    verbose: bool = False
    tol_stage0: float = 1.0e-4
    tol_stage1: float = 3.0e-4
    tol_stage2: float = 3.0e-4
    max_nodes: int = 50000
    max_nodes_fast: int = 12000


@dataclass
class DatasetConfig:
    # Legacy shared case count kept for backward compatibility with old configs.
    n_cases: int = 200
    decoder_n_cases: int = 200
    encoder_n_cases: int = 200
    n_nodes: int = 256
    seed: int = 42
    max_sample_attempts: int = 50000
    output_dir: str = "outputs/BMN_SCR_DD_outputs"
    full_dataset_filename: str = "decoder_fullfield_dataset.npz"
    encoder_dataset_filename: str = "bmn_encoder_dataset.npz"
    output_vars: List[str] = field(default_factory=lambda: ["x", "z", "theta", "T", "M"])
    observation_vars: List[str] = field(default_factory=lambda: ["T", "M", "theta"])
    # If sensor_indices is empty, sensor_s is used. If both are empty, equally spaced internal sensors are generated.
    sensor_indices: List[int] = field(default_factory=list)
    sensor_s: List[float] = field(default_factory=list)
    n_default_sensors: int = 6
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    save_failed_cases: bool = True


@dataclass
class DecoderNetworkConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 5
    activation: str = "tanh"
    dropout: float = 0.0


@dataclass
class DecoderTrainingConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 2000
    batch_size: int = 8192
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    print_every: int = 100
    patience: int = 300
    model_filename: str = "Decoder_DD_model.pth"
    history_filename: str = "Decoder_DD_history.json"


@dataclass
class EncoderNetworkConfig:
    # ``architecture`` controls only the Encoder used by BMN_DD.py.
    # ``bounded_mlp`` is a PIBAE-inspired MLP: LayerNorm + GELU + sigmoid-bounded
    # physical parameters, while still using the general observation/scaler pipeline.
    # ``dense_mlp`` keeps the original v0.1 unconstrained scaled-output MLP.
    architecture: str = "bounded_mlp"
    hidden_dim: int = 256
    num_hidden_layers: int = 4
    activation: str = "gelu"
    dropout: float = 0.0
    use_layer_norm: bool = True
    bounded_output: bool = True


@dataclass
class EncoderTrainingConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 1500
    batch_size: int = 128
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    print_every: int = 100
    patience: int = 300
    model_filename: str = "BMN_DD_encoder.pth"
    history_filename: str = "BMN_DD_history.json"
    # Optional decoder-assisted losses. The paper baseline uses parameter loss only.
    lambda_response: float = 0.0
    response_stage2_start_epoch: int = 1
    response_loss_vars: List[str] = field(default_factory=lambda: ["theta", "T", "M"])
    response_grad_target_ratio: float = 0.2
    response_lambda_ema: float = 0.9
    response_lambda_min: float = 1.0e-4
    response_lambda_max: float = 10.0
    response_bound_growth: float = 2.0
    response_bound_saturation_steps: int = 100
    response_scale_floor: float = 1.0e-6
    lambda_observation: float = 0.0
    # Soft physical-order penalty for current-profile parameters.
    # The training samples satisfy Us >= Ub; this term prevents the Encoder from
    # producing non-physical inverted current profiles near domain boundaries.
    lambda_order: float = 1.0


@dataclass
class BMNConfig:
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    ranges: ParameterRanges = field(default_factory=ParameterRanges)
    exact_solver: ExactSolverOptions = field(default_factory=ExactSolverOptions)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    decoder_network: DecoderNetworkConfig = field(default_factory=DecoderNetworkConfig)
    decoder_training: DecoderTrainingConfig = field(default_factory=DecoderTrainingConfig)
    encoder_network: EncoderNetworkConfig = field(default_factory=EncoderNetworkConfig)
    encoder_training: EncoderTrainingConfig = field(default_factory=EncoderTrainingConfig)


def get_decoder_n_cases(cfg: BMNConfig) -> int:
    value = int(getattr(cfg.dataset, "decoder_n_cases", 0))
    return value if value > 0 else int(cfg.dataset.n_cases)


def get_encoder_n_cases(cfg: BMNConfig) -> int:
    value = int(getattr(cfg.dataset, "encoder_n_cases", 0))
    return value if value > 0 else int(cfg.dataset.n_cases)


# =============================================================================
# 2. Generic utilities
# =============================================================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _update_dataclass_from_dict(obj: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    return obj


def config_from_dict(data: Dict[str, Any]) -> BMNConfig:
    cfg = BMNConfig()
    for section_name, section_cls in [
        ("physical", PhysicalConfig),
        ("ranges", ParameterRanges),
        ("exact_solver", ExactSolverOptions),
        ("dataset", DatasetConfig),
        ("decoder_network", DecoderNetworkConfig),
        ("decoder_training", DecoderTrainingConfig),
        ("encoder_network", EncoderNetworkConfig),
        ("encoder_training", EncoderTrainingConfig),
    ]:
        if section_name in data and isinstance(data[section_name], dict):
            section_obj = getattr(cfg, section_name)
            _update_dataclass_from_dict(section_obj, data[section_name])
    return cfg


def load_config(path: str | Path) -> BMNConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = config_from_dict(data)
    physics_ref = str(data.get("physics_config", "physics_config.json"))
    physics_path = Path(physics_ref)
    if not physics_path.is_absolute():
        physics_path = path.parent / physics_path
    if physics_path.exists():
        with open(physics_path, "r", encoding="utf-8") as f:
            physics_data = json.load(f)
        physical_block = physics_data.get("physical", physics_data)
        if isinstance(physical_block, dict):
            _update_dataclass_from_dict(cfg.physical, physical_block)
    return cfg


def save_config(
    cfg: BMNConfig,
    path: str | Path,
    physics_path: Optional[str | Path] = None,
    physics_ref: str = "physics_config.json",
) -> None:
    path = Path(path)
    cfg_dict = asdict(cfg)
    cfg_dict.pop("physical", None)
    cfg_dict["physics_config"] = physics_ref
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
    if physics_path is not None:
        save_physics_config(cfg.physical, physics_path)


def save_physics_config(phys: PhysicalConfig, path: str | Path) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"physical": asdict(phys)}, f, indent=2, ensure_ascii=False)


@dataclass
class StandardScaler:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    eps: float = 1.0e-12

    def fit(self, x: np.ndarray) -> "StandardScaler":
        arr = np.asarray(x, dtype=np.float64)
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (np.asarray(x, dtype=np.float64) * self.std + self.mean).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandardScaler":
        obj = cls(eps=float(data.get("eps", 1.0e-12)))
        obj.mean = None if data.get("mean") is None else np.asarray(data["mean"], dtype=np.float64)
        obj.std = None if data.get("std") is None else np.asarray(data["std"], dtype=np.float64)
        return obj


def make_activation(name: str) -> nn.Module:
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


class DenseMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, activation: str, dropout: float = 0.0) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), make_activation(activation)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), make_activation(activation)])
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# 3. Parameter sampling and exact solution generation
# =============================================================================


PARAM_NAMES = ["Dx", "ht", "Us", "Ub", "p"]
C_NAMES = ["Dx", "ht"]
MU_NAMES = ["Us", "Ub", "p"]
ALL_RESPONSE_VARS = ["x", "z", "theta", "T", "M", "Q", "H", "V"]


def case_geometry_is_admissible(dx: float, ht: float, phys: PhysicalConfig) -> bool:
    x0 = phys.x_bottom
    z0 = phys.z_lower_equilibrium
    x1 = phys.x_bottom + dx
    z1 = -ht
    straight = math.hypot(x1 - x0, z1 - z0)
    # Additional mild geometric screen used in previous scripts: avoid excessive slack for monotonic 2D layouts.
    vertical = abs(z1 - z0)
    return straight < phys.L and phys.L <= dx + vertical + 1.0e-9


def sample_one_case(rng: np.random.Generator, ranges: ParameterRanges, phys: PhysicalConfig) -> Dict[str, float]:
    while True:
        ub = float(rng.uniform(ranges.Ub_min, ranges.Ub_max))
        us = float(rng.uniform(max(ub, ranges.Us_min), ranges.Us_max))
        p = float(rng.uniform(ranges.p_min, ranges.p_max))
        dx = float(rng.uniform(ranges.Dx_min, ranges.Dx_max))
        ht = float(rng.uniform(ranges.ht_min, ranges.ht_max))
        if case_geometry_is_admissible(dx, ht, phys):
            return {"Dx": dx, "ht": ht, "Us": us, "Ub": ub, "p": p}


def make_exact_physical_config(phys: PhysicalConfig) -> Any:
    if ExactPhysicalConfig is None:
        raise RuntimeError(f"Exact solver unavailable: {EXACT_SOLVER_IMPORT_ERROR}")
    return ExactPhysicalConfig(
        D_o=phys.D_o,
        t=phys.t,
        E_steel=phys.E_steel,
        rho_s=phys.rho_s,
        rho_w=phys.rho_w,
        g=phys.g,
        C_d=phys.C_d,
        L=phys.L,
        water_depth=phys.water_depth,
        x_bottom=phys.x_bottom,
        k_b=phys.k_b,
    )


def make_exact_solver_config(opts: ExactSolverOptions, n_nodes: int) -> Any:
    if ExactSolverConfig is None:
        raise RuntimeError(f"Exact solver unavailable: {EXACT_SOLVER_IMPORT_ERROR}")
    return ExactSolverConfig(
        n_eval=int(n_nodes),
        use_fast_solver_first=bool(opts.use_fast_solver_first),
        verbose=bool(opts.verbose),
        tol_stage0=float(opts.tol_stage0),
        tol_stage1=float(opts.tol_stage1),
        tol_stage2=float(opts.tol_stage2),
        max_nodes=int(opts.max_nodes),
        max_nodes_fast=int(opts.max_nodes_fast),
    )


def compute_exact_case(case: Dict[str, float], phys: PhysicalConfig, exact_opts: ExactSolverOptions, n_nodes: int) -> Optional[Dict[str, np.ndarray]]:
    if not EXACT_SOLVER_AVAILABLE:
        raise RuntimeError(f"scr_exact_bvp_solver.py could not be imported: {EXACT_SOLVER_IMPORT_ERROR}")
    exact_phys = make_exact_physical_config(phys)
    solver_cfg = make_exact_solver_config(exact_opts, n_nodes=n_nodes)
    s, y, success, info = solve_scr_exact(
        Us=case["Us"],
        Ub=case["Ub"],
        p_exp=case["p"],
        Dx=case["Dx"],
        ht=case["ht"],
        phys=exact_phys,
        solver_cfg=solver_cfg,
    )
    if not success:
        return None
    x, z, theta, H, V, M = y
    T, Q = compute_local_resultants_from_global(theta, H, V)
    return {
        "s": np.asarray(s, dtype=np.float64),
        "x": np.asarray(x, dtype=np.float64),
        "z": np.asarray(z, dtype=np.float64),
        "theta": np.asarray(theta, dtype=np.float64),
        "H": np.asarray(H, dtype=np.float64),
        "V": np.asarray(V, dtype=np.float64),
        "T": np.asarray(T, dtype=np.float64),
        "Q": np.asarray(Q, dtype=np.float64),
        "M": np.asarray(M, dtype=np.float64),
        "info_json": json.dumps(info, ensure_ascii=False),
    }


def generate_decoder_dataset(cfg: BMNConfig) -> Path:
    out_dir = ensure_dir(cfg.dataset.output_dir)
    save_path = out_dir / cfg.dataset.full_dataset_filename
    rng = np.random.default_rng(cfg.dataset.seed)
    target_cases = get_decoder_n_cases(cfg)
    params: List[List[float]] = []
    raw_fields: Dict[str, List[np.ndarray]] = {key: [] for key in ALL_RESPONSE_VARS}
    info_json: List[str] = []
    failed_cases: List[Dict[str, Any]] = []
    s_grid: Optional[np.ndarray] = None
    attempts = 0
    t0 = time.time()

    print("=" * 88)
    print("Generating exact-solver dataset for Decoder_DD")
    print(f"Target cases : {target_cases}")
    print(f"Nodes/case   : {cfg.dataset.n_nodes}")
    print(f"Output       : {save_path}")
    print("=" * 88)

    while len(params) < target_cases:
        attempts += 1
        if attempts > cfg.dataset.max_sample_attempts:
            raise RuntimeError(f"Only collected {len(params)} valid exact cases within {attempts} attempts.")
        case = sample_one_case(rng, cfg.ranges, cfg.physical)
        try:
            exact = compute_exact_case(case, cfg.physical, cfg.exact_solver, cfg.dataset.n_nodes)
        except Exception as exc:
            exact = None
            failed_cases.append({"case": case, "reason": repr(exc)})
        if exact is None:
            failed_cases.append({"case": case, "reason": "exact_solver_failed"})
            continue
        if s_grid is None:
            s_grid = exact["s"]
        params.append([case[name] for name in PARAM_NAMES])
        for key in ALL_RESPONSE_VARS:
            raw_fields[key].append(exact[key])
        info_json.append(exact.get("info_json", "{}"))
        if len(params) == 1 or len(params) % max(1, target_cases // 10) == 0:
            print(f"  collected {len(params):5d}/{target_cases} | attempts={attempts} | elapsed={time.time()-t0:.1f}s")

    assert s_grid is not None
    params_arr = np.asarray(params, dtype=np.float32)
    c_arr = params_arr[:, [PARAM_NAMES.index(name) for name in C_NAMES]]
    mu_arr = params_arr[:, [PARAM_NAMES.index(name) for name in MU_NAMES]]
    output_vars = list(cfg.dataset.output_vars)
    for var in output_vars:
        if var not in ALL_RESPONSE_VARS:
            raise ValueError(f"Unsupported output var {var!r}; available={ALL_RESPONSE_VARS}")
    y_arr = np.stack([np.stack(raw_fields[var], axis=0) for var in output_vars], axis=-1).astype(np.float32)
    arrays = {
        "s": s_grid.astype(np.float32),
        "params": params_arr,
        "c": c_arr.astype(np.float32),
        "mu": mu_arr.astype(np.float32),
        "y": y_arr,
        "param_names": np.asarray(PARAM_NAMES),
        "c_names": np.asarray(C_NAMES),
        "mu_names": np.asarray(MU_NAMES),
        "output_vars": np.asarray(output_vars),
        "info_json": np.asarray(info_json),
        "config_json": np.asarray(json.dumps(asdict(cfg), ensure_ascii=False)),
    }
    for key, vals in raw_fields.items():
        arrays[f"field_{key}"] = np.stack(vals, axis=0).astype(np.float32)
    np.savez_compressed(save_path, **arrays)
    if cfg.dataset.save_failed_cases:
        with open(out_dir / "decoder_dataset_failed_cases.json", "w", encoding="utf-8") as f:
            json.dump(failed_cases, f, indent=2, ensure_ascii=False)
    save_config(
        cfg,
        out_dir / "para_config_used.json",
        physics_path=out_dir / "physics_config_used.json",
        physics_ref="physics_config_used.json",
    )
    print("=" * 88)
    print(f"Dataset generation finished: {save_path}")
    print(f"Successful cases: {len(params)} | failed attempts: {len(failed_cases)}")
    print("=" * 88)
    return save_path


# =============================================================================
# 4. Decoder model training and inference
# =============================================================================


def build_decoder_point_arrays(dataset: Dict[str, np.ndarray], case_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(dataset["s"], dtype=np.float32)
    params = np.asarray(dataset["params"], dtype=np.float32)[case_indices]
    y = np.asarray(dataset["y"], dtype=np.float32)[case_indices]
    n_cases, n_nodes, n_out = y.shape
    s_col = np.tile(s[None, :, None], (n_cases, 1, 1))
    p_cols = np.tile(params[:, None, :], (1, n_nodes, 1))
    x_in = np.concatenate([s_col, p_cols], axis=-1).reshape(n_cases * n_nodes, 1 + len(PARAM_NAMES))
    y_out = y.reshape(n_cases * n_nodes, n_out)
    return x_in.astype(np.float32), y_out.astype(np.float32)


def split_case_indices(n_cases: int, train_fraction: float, val_fraction: float, seed: int) -> Dict[str, np.ndarray]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_cases)
    rng.shuffle(indices)
    n_train = max(1, int(round(n_cases * train_fraction)))
    n_val = int(round(n_cases * val_fraction))
    if n_train + n_val >= n_cases:
        n_train = max(1, n_cases - n_val - 1)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    if val_idx.size == 0:
        val_idx = test_idx[: min(1, test_idx.size)]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, device: str) -> DataLoader:
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate_decoder_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    mse = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            total += float(mse(pred, yb).detach().cpu())
            count += int(yb.numel())
    return total / max(count, 1)


def train_decoder(cfg: BMNConfig, dataset_path: Optional[str | Path] = None) -> Path:
    out_dir = ensure_dir(cfg.dataset.output_dir)
    dataset_path = Path(dataset_path) if dataset_path is not None else out_dir / cfg.dataset.full_dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Decoder dataset not found: {dataset_path}")
    data_npz = np.load(dataset_path, allow_pickle=True)
    dataset = {key: data_npz[key] for key in data_npz.files}
    n_cases = int(dataset["params"].shape[0])
    splits = split_case_indices(n_cases, cfg.dataset.train_fraction, cfg.dataset.val_fraction, cfg.decoder_training.seed)
    x_train, y_train = build_decoder_point_arrays(dataset, splits["train"])
    x_val, y_val = build_decoder_point_arrays(dataset, splits["val"])

    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_s = x_scaler.transform(x_train)
    y_train_s = y_scaler.transform(y_train)
    x_val_s = x_scaler.transform(x_val)
    y_val_s = y_scaler.transform(y_val)

    device = cfg.decoder_training.device if torch.cuda.is_available() or cfg.decoder_training.device == "cpu" else "cpu"
    set_seed(cfg.decoder_training.seed)
    model = DenseMLP(
        input_dim=x_train_s.shape[1],
        output_dim=y_train_s.shape[1],
        hidden_dim=cfg.decoder_network.hidden_dim,
        num_hidden_layers=cfg.decoder_network.num_hidden_layers,
        activation=cfg.decoder_network.activation,
        dropout=cfg.decoder_network.dropout,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.decoder_training.lr, weight_decay=cfg.decoder_training.weight_decay)
    loss_fn = nn.MSELoss()
    train_loader = _make_loader(x_train_s, y_train_s, cfg.decoder_training.batch_size, True, device)
    val_loader = _make_loader(x_val_s, y_val_s, cfg.decoder_training.batch_size, False, device)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0
    t0 = time.time()
    print("=" * 88)
    print("Training pure data-supervised Decoder_DD")
    print(f"Dataset     : {dataset_path}")
    print(f"Train cases : {len(splits['train'])} | Val cases: {len(splits['val'])} | Test cases: {len(splits['test'])}")
    print(f"Device      : {device}")
    print("=" * 88)

    for epoch in range(1, cfg.decoder_training.epochs + 1):
        model.train()
        running = 0.0
        n_items = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.decoder_training.grad_clip and cfg.decoder_training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.decoder_training.grad_clip)
            opt.step()
            running += float(loss.detach().cpu()) * xb.shape[0]
            n_items += xb.shape[0]
        train_loss = running / max(n_items, 1)
        val_loss = evaluate_decoder_loss(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if epoch == 1 or epoch % cfg.decoder_training.print_every == 0:
            print(f"[Decoder] epoch={epoch:5d} | train={train_loss:.3e} | val={val_loss:.3e} | best={best_val:.3e} | elapsed={time.time()-t0:.1f}s")
        if cfg.decoder_training.patience > 0 and no_improve >= cfg.decoder_training.patience:
            print(f"[Decoder] early stopping at epoch {epoch}; best val={best_val:.3e}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = out_dir / cfg.decoder_training.model_filename
    checkpoint = {
        "version": "BMN-SCR-DD v0.1 Decoder",
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "input_scaler": x_scaler.to_dict(),
        "output_scaler": y_scaler.to_dict(),
        "input_names": ["s"] + PARAM_NAMES,
        "output_vars": [str(v) for v in dataset["output_vars"].tolist()],
        "param_names": PARAM_NAMES,
        "c_names": C_NAMES,
        "mu_names": MU_NAMES,
        "splits": {k: v.tolist() for k, v in splits.items()},
        "dataset_path": str(dataset_path),
        "best_val_loss": best_val,
    }
    torch.save(checkpoint, ckpt_path)
    with open(out_dir / cfg.decoder_training.history_filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("=" * 88)
    print(f"Decoder training finished: {ckpt_path}")
    print("=" * 88)
    return ckpt_path


def load_decoder_model(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Tuple[DenseMLP, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    net_cfg = checkpoint["config"]["decoder_network"]
    input_dim = len(checkpoint["input_names"])
    output_dim = len(checkpoint["output_vars"])
    model = DenseMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=int(net_cfg["hidden_dim"]),
        num_hidden_layers=int(net_cfg["num_hidden_layers"]),
        activation=str(net_cfg["activation"]),
        dropout=float(net_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def decode_fullfield_np(
    model: nn.Module,
    checkpoint: Dict[str, Any],
    s: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Decode full-field response for a single case.

    Parameters
    ----------
    s : (N,) physical arc-length coordinates.
    c : (2,) [Dx, ht].
    mu: (3,) [Us, Ub, p].

    Returns
    -------
    y : (N, n_output) physical-scale decoder response.
    """
    x_scaler = StandardScaler.from_dict(checkpoint["input_scaler"])
    y_scaler = StandardScaler.from_dict(checkpoint["output_scaler"])
    params = np.asarray([c[0], c[1], mu[0], mu[1], mu[2]], dtype=np.float32)
    x_in = np.concatenate([s[:, None].astype(np.float32), np.tile(params[None, :], (len(s), 1))], axis=1)
    x_s = x_scaler.transform(x_in)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.tensor(x_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    return y_scaler.inverse_transform(pred_s)


# =============================================================================
# 5. Observation extraction and BMN dataset construction
# =============================================================================


def resolve_sensor_indices(s: np.ndarray, dataset_cfg: DatasetConfig) -> np.ndarray:
    n = len(s)
    if dataset_cfg.sensor_indices:
        idx = np.asarray(dataset_cfg.sensor_indices, dtype=int)
    elif dataset_cfg.sensor_s:
        target_s = np.asarray(dataset_cfg.sensor_s, dtype=float)
        idx = np.asarray([int(np.argmin(np.abs(s - val))) for val in target_s], dtype=int)
    else:
        # Internal sensors; top location is appended separately.
        idx = np.linspace(1, n - 2, int(dataset_cfg.n_default_sensors), dtype=int)
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"sensor indices out of bounds for n={n}: {idx}")
    return np.unique(idx)


def extract_observations_from_fields(
    y: np.ndarray,
    s: np.ndarray,
    output_vars: Sequence[str],
    observation_vars: Sequence[str],
    sensor_indices: np.ndarray,
) -> np.ndarray:
    var_to_col = {name: i for i, name in enumerate(output_vars)}
    for var in ["x", "z", *observation_vars]:
        if var not in var_to_col:
            raise ValueError(f"Variable {var!r} is required for observation extraction but absent in output_vars={output_vars}")
    obs_list: List[np.ndarray] = []
    obs_list.append(y[:, -1, var_to_col["x"]])
    obs_list.append(y[:, -1, var_to_col["z"]])
    for idx in sensor_indices:
        for var in observation_vars:
            obs_list.append(y[:, idx, var_to_col[var]])
    return np.stack(obs_list, axis=1).astype(np.float32)


def build_encoder_dataset(cfg: BMNConfig, dataset_path: Optional[str | Path] = None) -> Path:
    out_dir = ensure_dir(cfg.dataset.output_dir)
    dataset_path = Path(dataset_path) if dataset_path is not None else out_dir / cfg.dataset.full_dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"Full-field dataset not found: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)
    s = np.asarray(data["s"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.float32)
    c = np.asarray(data["c"], dtype=np.float32)
    mu = np.asarray(data["mu"], dtype=np.float32)
    params = np.asarray(data["params"], dtype=np.float32)
    output_vars = [str(v) for v in data["output_vars"].tolist()]
    sensor_indices = resolve_sensor_indices(s, cfg.dataset)
    obs = extract_observations_from_fields(y, s, output_vars, cfg.dataset.observation_vars, sensor_indices)
    obs_names = ["x_top", "z_top"]
    for idx in sensor_indices:
        for var in cfg.dataset.observation_vars:
            obs_names.append(f"{var}_idx{int(idx)}")
    save_path = out_dir / cfg.dataset.encoder_dataset_filename
    np.savez_compressed(
        save_path,
        observations=obs,
        c=c,
        mu=mu,
        y=y,
        params=params,
        s=s,
        sensor_indices=sensor_indices.astype(np.int64),
        sensor_s=s[sensor_indices].astype(np.float32),
        obs_names=np.asarray(obs_names),
        output_vars=np.asarray(output_vars),
        observation_vars=np.asarray(cfg.dataset.observation_vars),
        param_names=np.asarray(PARAM_NAMES),
        c_names=np.asarray(C_NAMES),
        mu_names=np.asarray(MU_NAMES),
        source_dataset=str(dataset_path),
        config_json=np.asarray(json.dumps(asdict(cfg), ensure_ascii=False)),
    )
    print("=" * 88)
    print(f"BMN encoder dataset saved: {save_path}")
    print(f"Observations shape : {obs.shape}")
    print(f"Full response shape: {y.shape}")
    print(f"Sensors            : {sensor_indices.tolist()}")
    print("=" * 88)
    return save_path


# =============================================================================
# 6. CLI
# =============================================================================


def make_default_config(path: str | Path) -> None:
    cfg = BMNConfig()
    path = Path(path)
    physics_path = path.parent / "physics_config.json"
    save_config(cfg, path, physics_path=physics_path, physics_ref="physics_config.json")
    print(f"Default para_config.json written to: {path.resolve()}")
    print(f"Default physics_config.json written to: {physics_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BMN-SCR-DD v0.1 Decoder data generation and training")
    parser.add_argument("--config", type=str, default="para_config.json", help="Path to para_config.json")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["make_config", "generate", "train_decoder", "extract_encoder_data", "all"],
        help="Execution mode",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional full-field dataset path")
    args = parser.parse_args()

    if args.mode == "make_config":
        make_default_config(args.config)
        return

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}. Writing a default config first.")
        make_default_config(cfg_path)
        print("Edit the config and rerun.")
        return

    cfg = load_config(cfg_path)
    dataset_path: Optional[Path] = Path(args.dataset) if args.dataset is not None else None

    if args.mode in {"generate", "all"}:
        dataset_path = generate_decoder_dataset(cfg)
    if args.mode in {"train_decoder", "all"}:
        if dataset_path is None:
            dataset_path = Path(cfg.dataset.output_dir) / cfg.dataset.full_dataset_filename
        train_decoder(cfg, dataset_path)
    if args.mode in {"extract_encoder_data", "all"}:
        if dataset_path is None:
            dataset_path = Path(cfg.dataset.output_dir) / cfg.dataset.full_dataset_filename
        build_encoder_dataset(cfg, dataset_path)


if __name__ == "__main__":
    main()
