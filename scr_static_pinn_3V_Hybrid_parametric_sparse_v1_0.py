#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py

Standalone parametric sparse-in-parameter extension of the 3V hybrid PINN.

Key ideas
---------
1. Decoder inputs: [s, Us, Ub, p, Dx, ht]
2. Decoder outputs: theta(s), M(s), T(s)
3. Physics batch: random parameter cases with PDE + BC only
4. Data batch: sparse parameter anchors with full-field hybrid supervision
5. No dependency on `scr_static_pinn_3V_Hybrid_v1_0.py`
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EXACT_SOLVER_IMPORT_ERROR = None
# `exact` 数据监督优先走主 solver，失败时再退回老版本 solver。
# 这里保留两条导入链，主要是为了兼容我之前仓库里的不同 exact 实现。
# for test
# test 2

# test 3

from scr_exact_bvp_solver import (
        solve_scr_exact,
        compute_local_resultants_from_global,
        PhysicalConfig as ExactPhysicalConfig,
        SolverConfig as ExactSolverConfig,
)
EXACT_SOLVER_AVAILABLE = True


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
    def z_bottom(self) -> float:
        return -self.water_depth - self.w_eff / self.k_b

    @property
    def z_bed(self) -> float:
        return -self.water_depth

    @property
    def D_i(self) -> float:
        return self.D_o - 2.0 * self.t

    @property
    def A_s(self) -> float:
        return np.pi / 4.0 * (self.D_o**2 - self.D_i**2)

    @property
    def A_o(self) -> float:
        return np.pi / 4.0 * self.D_o**2

    @property
    def I(self) -> float:
        return np.pi / 64.0 * (self.D_o**4 - self.D_i**4)

    @property
    def EA(self) -> float:
        return self.E_steel * self.A_s

    @property
    def EI(self) -> float:
        return self.E_steel * self.I

    @property
    def w_eff(self) -> float:
        return (self.rho_s * self.A_s - self.rho_w * self.A_o) * self.g


@dataclass
class ScaleConfig:
    T_scale: Optional[float] = None
    q_scale: Optional[float] = None
    M_scale: Optional[float] = None
    Q_scale: Optional[float] = None
    x_scale: Optional[float] = None
    theta_max: float = math.pi

    def build_from_physics(self, phys: PhysicalConfig) -> "ScaleConfig":
        # 统一把各状态量缩放到 O(1) 量级，减小 hybrid 训练时 data/PDE 项的数值落差。
        self.T_scale = max(abs(phys.w_eff) * phys.L, 1.0)
        self.q_scale = max(abs(phys.w_eff), 1.0)
        self.M_scale = max(phys.EI / phys.L, 1.0)
        self.Q_scale = max(self.T_scale, 1.0)
        self.x_scale = phys.L
        return self


@dataclass
class SingleCaseConfig:
    Us: float = 1.5
    Ub: float = 0.3
    p: float = 0.25
    Dx: float = 1800.0
    ht: float = 20.0


@dataclass
class ParameterRanges:
    Us_min: float = 0.5
    Us_max: float = 2.5
    Ub_min: float = 0.0
    Ub_max: float = 0.8
    p_min: float = 1.0 / 7.0
    p_max: float = 1.0 / 3.0
    Dx_min: float = 1700.0
    Dx_max: float = 1900.0
    ht_min: float = -10.0
    ht_max: float = 10.0


@dataclass
class NetworkConfig:
    hidden_dim: int = 256
    num_hidden_layers: int = 5
    activation: str = "tanh"
    theta_scale_factor: float = 1.0
    T0_scale_factor: float = 1.0
    T_res_scale_factor: float = 0.5
    M_scale_factor: float = 1.0


@dataclass
class HybridConfig:
    adam_steps: int = 12000
    lr: float = 5.0e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    data_w_x: float = 0.5
    data_w_z: float = 0.5
    data_w_theta: float = 1.0
    data_w_T: float = 1.0
    data_w_M: float = 0.5
    data_w_Q: float = 0.5
    tdp_data_weight: float = 2.0
    top_data_weight: float = 3.0
    schedule_num_stages: int = 20
    schedule_steps: Optional[List[int]] = None
    schedule_w_data: Optional[List[float]] = None
    schedule_w_pde: Optional[List[float]] = None
    schedule_grad_ratio_target: Optional[List[float]] = None
    grad_balance_every: int = 20
    grad_balance_alpha: float = 0.1
    grad_balance_eps: float = 1.0e-12
    grad_balance_corr_min: float = 0.005
    grad_balance_corr_max: float = 20.0

    def ensure_schedule(self, adam_steps: int) -> None:
        """Build the default staircase schedule from the actual training length.

        If the four schedule lists are all provided explicitly, keep them as-is.
        Otherwise, generate a matched schedule automatically from `adam_steps`.
        """
        explicit = [
            self.schedule_steps,
            self.schedule_w_data,
            self.schedule_w_pde,
            self.schedule_grad_ratio_target,
        ]
        if all(item is not None for item in explicit):
            n_items = len(self.schedule_steps)
            if not (
                n_items == len(self.schedule_w_data)
                == len(self.schedule_w_pde)
                == len(self.schedule_grad_ratio_target)
            ):
                raise ValueError("Explicit hybrid schedules must have the same length.")
            return
        if any(item is not None for item in explicit):
            raise ValueError("Provide either all hybrid schedule lists or none of them.")

        n_stages = max(int(self.schedule_num_stages), 1)
        self.schedule_steps = np.linspace(0, max(int(adam_steps) - 1, 0), n_stages, dtype=int).tolist()
        self.schedule_w_data = np.geomspace(1.0, 0.15, n_stages).tolist()
        self.schedule_w_pde = np.geomspace(0.2, 1.2, n_stages).tolist()
        self.schedule_grad_ratio_target = np.geomspace(10.0, 0.1, n_stages).tolist()


@dataclass
class FullStageConfig:
    pde_w_theta: float = 1.0
    pde_w_T: float = 1.0
    pde_w_M: float = 1.0
    lambda_bc_init: float = 1.0
    anneal_start_step: int = 800
    anneal_every: int = 20
    anneal_alpha: float = 0.1
    anneal_eps: float = 1.0e-12
    lambda_bc_min: float = 1.0
    lambda_bc_max: float = 1.0e4
    bc_w_M0: float = 0.0
    bc_w_ML: float = 0.0
    bc_w_x: float = 1.0
    bc_w_z: float = 1.0
    mesh_region_bounds: List[float] = field(default_factory=lambda: [0.0, 1200.0, 1450.0, 2420.0, 2500.0])
    mesh_region_elem_sizes: List[float] = field(
        default_factory=lambda: [120.0, 250.0 / 48.0, (2420.0 - 1450.0) / 24.0, (2500.0 - 2420.0) / 12.0]
    )
    strong_points_per_elem: int = 5
    gauss_points_per_elem: int = 5


@dataclass
class SparseParametricDataConfig:
    reference_source: str = "exact"  # "exact" or "catenary": how the reference fields for data supervision are generated. "exact" for exact solver，"catenary" for catenary aaproximation.
    num_cases: int = 24    # number of sparse parameter anchors with data supervision. Better be <= 48 for the current parameter ranges to avoid duplicates.
    batch_size: int = 24    # number of sparse parameter-space data cases used in each training step; keep it <= num_cases for sampling without replacement.
    max_sample_attempts: int = 5000


@dataclass
class ParametricTrainingConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_nodes: int = 256
    physics_batch_size: int = 12  # number of random parameter cases used in each PDE/BC training batch.
    adam_steps: int = 20000
    print_every: int = 500
    plot_every: int = 2000
    output_dir: str = "scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def linspace_01(n: int, device: str) -> torch.Tensor:
    return torch.linspace(0.0, 1.0, n, device=device)


def cumulative_trapezoid_nonuniform(y: torch.Tensor, x: torch.Tensor, initial: float = 0.0) -> torch.Tensor:
    out = torch.zeros_like(y)
    out[..., 0] = initial
    dx = x[..., 1:] - x[..., :-1]
    avg = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    out[..., 1:] = torch.cumsum(avg, dim=-1) + initial
    return out


def normalize_to_minus1_plus1(value: torch.Tensor, vmin: float, vmax: float) -> torch.Tensor:
    return 2.0 * (value - vmin) / (vmax - vmin) - 1.0


def build_case_dict(single_case: SingleCaseConfig, device: str) -> Dict[str, torch.Tensor]:
    return {
        "Us": torch.tensor([single_case.Us], device=device),
        "Ub": torch.tensor([single_case.Ub], device=device),
        "p": torch.tensor([single_case.p], device=device),
        "Dx": torch.tensor([single_case.Dx], device=device),
        "ht": torch.tensor([single_case.ht], device=device),
    }


def build_model_inputs(s01: torch.Tensor, case_params: Dict[str, torch.Tensor], ranges: ParameterRanges) -> torch.Tensor:
    # 网络始终吃固定 6 维输入：[s, Us, Ub, p, Dx, ht]。
    # 其中 s 用 [0,1] 网格给出，其余参数按范围归一化到 [-1,1]。
    _, n_pts = s01.shape
    s_feat = 2.0 * s01 - 1.0
    Us_feat = normalize_to_minus1_plus1(case_params["Us"], ranges.Us_min, ranges.Us_max)[:, None].repeat(1, n_pts)
    Ub_feat = normalize_to_minus1_plus1(case_params["Ub"], ranges.Ub_min, ranges.Ub_max)[:, None].repeat(1, n_pts)
    p_feat = normalize_to_minus1_plus1(case_params["p"], ranges.p_min, ranges.p_max)[:, None].repeat(1, n_pts)
    Dx_feat = normalize_to_minus1_plus1(case_params["Dx"], ranges.Dx_min, ranges.Dx_max)[:, None].repeat(1, n_pts)
    ht_feat = normalize_to_minus1_plus1(case_params["ht"], ranges.ht_min, ranges.ht_max)[:, None].repeat(1, n_pts)
    return torch.stack([s_feat, Us_feat, Ub_feat, p_feat, Dx_feat, ht_feat], dim=-1)


def _build_uniform_segment_edges(start: float, end: float, elem_size: float) -> np.ndarray:
    if end <= start + 1.0e-12:
        return np.array([start], dtype=float)
    if elem_size <= 0.0:
        raise ValueError("element size must be positive")
    n_elem = max(int(round((end - start) / elem_size)), 1)
    return np.linspace(start, end, n_elem + 1)


def build_explicit_mesh_edges(phys: PhysicalConfig, full_cfg: FullStageConfig) -> np.ndarray:
    bounds = np.asarray(full_cfg.mesh_region_bounds, dtype=float)
    elem_sizes = np.asarray(full_cfg.mesh_region_elem_sizes, dtype=float)
    if len(elem_sizes) != len(bounds) - 1:
        raise ValueError("mesh_region_elem_sizes must have len(mesh_region_bounds) - 1 entries")
    if abs(bounds[0]) > 1.0e-12 or abs(bounds[-1] - phys.L) > 1.0e-12:
        raise ValueError("mesh_region_bounds must start at 0 and end at L")
    # 这里沿用 hybrid v1_0 的显式分段网格，便于在 TDP 附近加密采样。
    pieces = [_build_uniform_segment_edges(float(a), float(b), float(h)) for a, b, h in zip(bounds[:-1], bounds[1:], elem_sizes)]
    edges = [pieces[0][0]]
    for seg in pieces:
        if seg[0] != edges[-1]:
            edges.append(seg[0])
        edges.extend(seg[1:])
    return np.asarray(edges, dtype=float)


def build_element_midpoints(element_edges: np.ndarray, phys: PhysicalConfig, device: str, n_pts_per_elem: int) -> torch.Tensor:
    n_pts_per_elem = max(int(n_pts_per_elem), 1)
    s_blocks = []
    for a, b in zip(element_edges[:-1], element_edges[1:]):
        ds = (b - a) / float(n_pts_per_elem)
        s_blocks.append(a + (np.arange(n_pts_per_elem, dtype=float) + 0.5) * ds)
    s_all = np.concatenate([blk for blk in s_blocks if blk.size > 0])
    return torch.tensor(s_all / phys.L, dtype=torch.float32, device=device)[None, :]


def solve_catenary_parameter(delta_x: float, delta_z: float, L: float) -> float:
    rhs = math.sqrt(max(L**2 - delta_z**2, 1.0e-12)) / max(delta_x, 1.0e-12)
    if rhs <= 1.0:
        raise ValueError("Invalid catenary geometry")

    def log_two_sinh(x: float) -> float:
        if x < 20.0:
            return math.log(2.0 * math.sinh(x))
        return x + math.log1p(-math.exp(-2.0 * x))

    def f(a: float) -> float:
        return log_two_sinh(delta_x / (2.0 * a)) - math.log(rhs)

    from scipy.optimize import root_scalar

    sol = root_scalar(f, bracket=[1.0e-6, 1.0e6], method="bisect")
    if not sol.converged:
        raise RuntimeError("Failed to solve catenary parameter")
    return sol.root


def generate_catenary_reference(phys: PhysicalConfig, single_case: SingleCaseConfig, n_nodes: int, device: str):
    x0 = phys.x_bottom
    z0 = phys.z_bottom
    x1 = phys.x_bottom + single_case.Dx
    z1 = -single_case.ht
    dx = x1 - x0
    dz = z1 - z0
    straight = math.hypot(dx, dz)
    if phys.L <= straight:
        raise ValueError("Pipe length is shorter than endpoint distance")

    # catenary 只作为可选参考，不承担最终物理真值的角色。
    def free_catenary_reference():
        a = solve_catenary_parameter(dx, dz, phys.L)
        ratio = np.clip(dz / phys.L, -0.999999, 0.999999)
        m = np.arctanh(ratio)
        u1 = m - dx / (2.0 * a)
        u2 = m + dx / (2.0 * a)
        C = z0 - a * np.cosh(u1)
        s = np.linspace(0.0, phys.L, n_nodes)
        u = np.arcsinh(s / a + np.sinh(u1))
        x = x0 + a * (u - u1)
        z = C + a * np.cosh(u)
        theta = np.arctan(np.sinh(u))
        H = np.full_like(s, a * phys.w_eff)
        V = H * np.tan(theta)
        return s, x, z, theta, H, V

    s, x, z, theta, H, V = free_catenary_reference()
    T = H * np.cos(theta) + V * np.sin(theta)
    return {
        "s": torch.tensor(s, dtype=torch.float32, device=device)[None, :],
        "x": torch.tensor(x, dtype=torch.float32, device=device)[None, :],
        "z": torch.tensor(z, dtype=torch.float32, device=device)[None, :],
        "theta": torch.tensor(theta, dtype=torch.float32, device=device)[None, :],
        "T": torch.tensor(T, dtype=torch.float32, device=device)[None, :],
    }


class ActivationFactory:
    @staticmethod
    def make(name: str) -> nn.Module:
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        raise ValueError(f"Unsupported activation: {name}")


class SCRStaticPINNMT(nn.Module):
    def __init__(self, net_cfg: NetworkConfig) -> None:
        super().__init__()
        self.net_cfg = net_cfg
        # T = T0 + T_res(s)；其中 T0 做成全局可学习量，避免网络在整条曲线上重复表达同一常量偏置。
        self.raw_T0 = nn.Parameter(torch.zeros(1))
        act = ActivationFactory.make(net_cfg.activation)
        layers = [nn.Linear(6, net_cfg.hidden_dim), act]
        for _ in range(net_cfg.num_hidden_layers - 1):
            layers.extend([nn.Linear(net_cfg.hidden_dim, net_cfg.hidden_dim), ActivationFactory.make(net_cfg.activation)])
        self.hidden = nn.Sequential(*layers)
        self.output_layer = nn.Linear(net_cfg.hidden_dim, 3)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_pts, n_in = x.shape
        xf = x.reshape(bsz * n_pts, n_in)
        y = self.output_layer(self.hidden(xf))
        return y.reshape(bsz, n_pts, 3)

    def T0(self, scales: ScaleConfig) -> torch.Tensor:
        return self.net_cfg.T0_scale_factor * scales.T_scale * F.softplus(self.raw_T0)[0]


@dataclass
class StateFields:
    theta: torch.Tensor
    M: torch.Tensor
    T: torch.Tensor
    T0: torch.Tensor
    dtheta_ds: torch.Tensor
    dM_ds: torch.Tensor
    d2M_ds2: torch.Tensor
    dT_ds: torch.Tensor
    x: torch.Tensor
    z: torch.Tensor
    dxds: torch.Tensor
    dzds: torch.Tensor
    qx: torch.Tensor
    qz: torch.Tensor
    qt: torch.Tensor
    qn: torch.Tensor
    penetration: torch.Tensor
    Q: torch.Tensor


@dataclass
class InferenceFields:
    theta: torch.Tensor
    M: torch.Tensor
    T: torch.Tensor
    T0: torch.Tensor
    x: torch.Tensor
    z: torch.Tensor
    dxds: torch.Tensor
    dzds: torch.Tensor
    penetration: torch.Tensor
    Q: torch.Tensor


def xi_from_inputs(model_inputs: torch.Tensor) -> torch.Tensor:
    return 0.5 * (model_inputs[..., 0] + 1.0)


def build_endpoint_inputs(model_inputs: torch.Tensor) -> torch.Tensor:
    left = model_inputs[:, :1, :].clone()
    right = model_inputs[:, :1, :].clone()
    left[..., 0] = -1.0
    right[..., 0] = 1.0
    return torch.cat([left, right], dim=1)


def project_moment_with_endpoint_elimination(
    model: SCRStaticPINNMT, raw_m: torch.Tensor, model_inputs: torch.Tensor, net_cfg: NetworkConfig, scales: ScaleConfig
) -> torch.Tensor:
    # 直接消去两端 raw moment，使 M(0)=M(L)=0 精确成立。
    # 这样比单纯乘 bubble 更稳，也更接近我在 single-case 版里想要的约束方式。
    xi = xi_from_inputs(model_inputs)
    M_raw = net_cfg.M_scale_factor * scales.M_scale * raw_m
    endpoint_inputs = build_endpoint_inputs(model_inputs)
    raw_end = model(endpoint_inputs)
    M0_raw = net_cfg.M_scale_factor * scales.M_scale * raw_end[:, 0, 1][:, None]
    ML_raw = net_cfg.M_scale_factor * scales.M_scale * raw_end[:, 1, 1][:, None]
    left_present = torch.isclose(xi[:, 0], torch.zeros_like(xi[:, 0]), atol=1.0e-7, rtol=0.0)
    right_present = torch.isclose(xi[:, -1], torch.ones_like(xi[:, -1]), atol=1.0e-7, rtol=0.0)
    if torch.any(left_present):
        M0_raw[left_present] = M_raw[left_present, 0][:, None]
    if torch.any(right_present):
        ML_raw[right_present] = M_raw[right_present, -1][:, None]
    return M_raw - (1.0 - xi) * M0_raw - xi * ML_raw


def apply_theta_anchor(theta_raw: torch.Tensor, model_inputs: torch.Tensor, net_cfg: NetworkConfig, scales: ScaleConfig):
    return net_cfg.theta_scale_factor * scales.theta_max * xi_from_inputs(model_inputs) * torch.sigmoid(theta_raw)


def convert_raw_outputs(
    model: SCRStaticPINNMT, raw: torch.Tensor, model_inputs: torch.Tensor, net_cfg: NetworkConfig, scales: ScaleConfig
):
    theta = apply_theta_anchor(raw[..., 0], model_inputs, net_cfg, scales)
    M = project_moment_with_endpoint_elimination(model, raw[..., 1], model_inputs, net_cfg, scales)
    T_res = net_cfg.T_res_scale_factor * scales.T_scale * torch.tanh(raw[..., 2])
    return theta, M, T_res


def derivative_wrt_s(model_inputs: torch.Tensor, field: torch.Tensor, phys: PhysicalConfig) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=field,
        inputs=model_inputs,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad[..., 0] * (2.0 / phys.L)


def reconstruct_geometry(theta: torch.Tensor, s_phys: torch.Tensor, phys: PhysicalConfig):
    dxds = torch.cos(theta)
    dzds = torch.sin(theta)
    x = cumulative_trapezoid_nonuniform(dxds, s_phys, initial=phys.x_bottom)
    z = cumulative_trapezoid_nonuniform(dzds, s_phys, initial=phys.z_bottom)
    return x, z, dxds, dzds


def current_velocity(z: torch.Tensor, Us: torch.Tensor, Ub: torch.Tensor, p_exp: torch.Tensor, phys: PhysicalConfig):
    depth_ratio = torch.clamp((z - phys.z_bed) / phys.water_depth, 0.0, 1.0)
    depth_ratio = torch.clamp(depth_ratio, min=1.0e-12)
    return Ub[:, None] + (Us[:, None] - Ub[:, None]) * depth_ratio.pow(p_exp[:, None])


def compute_distributed_loads_global(theta: torch.Tensor, z: torch.Tensor, case_params: Dict[str, torch.Tensor], phys: PhysicalConfig):
    # 分布载荷由三部分叠加：
    # 1. 有效重力
    # 2. 海流法向拖曳
    # 3. 海床 penalty 接触反力
    ct = torch.cos(theta)
    st = torch.sin(theta)
    qx_g = torch.zeros_like(theta)
    qz_g = -phys.w_eff * torch.ones_like(theta)
    Uc = current_velocity(z, case_params["Us"], case_params["Ub"], case_params["p"], phys)
    V_n = Uc * st
    qd_signed = 0.5 * phys.rho_w * phys.C_d * phys.D_o * V_n * torch.abs(V_n)
    qx_d = qd_signed * st
    qz_d = qd_signed * (-ct)
    penetration = torch.clamp(phys.z_bed - z, min=0.0)
    qx_c = torch.zeros_like(theta)
    qz_c = phys.k_b * penetration
    return qx_g + qx_d + qx_c, qz_g + qz_d + qz_c, penetration


def project_global_load_to_local(qx: torch.Tensor, qz: torch.Tensor, theta: torch.Tensor):
    ct = torch.cos(theta)
    st = torch.sin(theta)
    return qx * ct + qz * st, qx * st - qz * ct


def compute_training_fields(
    model: SCRStaticPINNMT,
    model_inputs: torch.Tensor,
    case_params: Dict[str, torch.Tensor],
    phys: PhysicalConfig,
    net_cfg: NetworkConfig,
    scales: ScaleConfig,
) -> StateFields:
    # 训练态下把需要的一阶/二阶导数一次性都算出来，后续 data/PDE/BC 共用同一批字段。
    raw = model(model_inputs)
    theta, M, T_res = convert_raw_outputs(model, raw, model_inputs, net_cfg, scales)
    T0 = model.T0(scales)
    T = T0 + T_res
    dtheta_ds = derivative_wrt_s(model_inputs, theta, phys)
    dM_ds = derivative_wrt_s(model_inputs, M, phys)
    d2M_ds2 = derivative_wrt_s(model_inputs, dM_ds, phys)
    dT_ds = derivative_wrt_s(model_inputs, T, phys)
    s_phys = 0.5 * (model_inputs[..., 0] + 1.0) * phys.L
    x, z, dxds, dzds = reconstruct_geometry(theta, s_phys, phys)
    qx, qz, penetration = compute_distributed_loads_global(theta, z, case_params, phys)
    qt, qn = project_global_load_to_local(qx, qz, theta)
    return StateFields(theta, M, T, T0, dtheta_ds, dM_ds, d2M_ds2, dT_ds, x, z, dxds, dzds, qx, qz, qt, qn, penetration, dM_ds)


@torch.no_grad()
def compute_inference_fields(
    model: SCRStaticPINNMT,
    model_inputs: torch.Tensor,
    case_params: Dict[str, torch.Tensor],
    phys: PhysicalConfig,
    net_cfg: NetworkConfig,
    scales: ScaleConfig,
) -> InferenceFields:
    raw = model(model_inputs)
    theta, M, T_res = convert_raw_outputs(model, raw, model_inputs, net_cfg, scales)
    T0 = model.T0(scales)
    T = T0 + T_res
    s_phys = 0.5 * (model_inputs[..., 0] + 1.0) * phys.L
    x, z, dxds, dzds = reconstruct_geometry(theta, s_phys, phys)
    Q = torch.tensor(np.gradient(M[0].cpu().numpy(), s_phys[0].cpu().numpy()), dtype=theta.dtype, device=theta.device)[None, :]
    return InferenceFields(theta, M, T, T0, x, z, dxds, dzds, torch.clamp(phys.z_bed - z, min=0.0), Q)


@dataclass
class DataLossTerms:
    total: torch.Tensor
    x: torch.Tensor
    z: torch.Tensor
    theta: torch.Tensor
    T: torch.Tensor
    M: torch.Tensor
    Q: torch.Tensor


@dataclass
class FullStageLossTerms:
    pde: torch.Tensor
    bc: torch.Tensor
    pde_theta: torch.Tensor
    pde_M: torch.Tensor
    pde_T: torch.Tensor
    bc_M0: torch.Tensor
    bc_ML: torch.Tensor
    bc_x: torch.Tensor
    bc_z: torch.Tensor


@dataclass
class BoundaryDiagnostics:
    M0_nd: torch.Tensor
    ML_nd: torch.Tensor
    xL_nd: torch.Tensor
    zL_nd: torch.Tensor
    L_bc_M0: torch.Tensor
    L_bc_ML: torch.Tensor
    L_bc_x: torch.Tensor
    L_bc_z: torch.Tensor


def compute_boundary_diagnostics(fields: StateFields, case_params: Dict[str, torch.Tensor], phys: PhysicalConfig) -> BoundaryDiagnostics:
    x_target = phys.x_bottom + case_params["Dx"]
    z_target = -case_params["ht"]
    moment_scale = max(phys.EI / phys.L, 1.0)
    M0_nd = fields.M[:, 0] / moment_scale
    ML_nd = fields.M[:, -1] / moment_scale
    xL_nd = (fields.x[:, -1] - x_target) / phys.L
    zL_nd = (fields.z[:, -1] - z_target) / phys.L
    return BoundaryDiagnostics(M0_nd, ML_nd, xL_nd, zL_nd, torch.mean(M0_nd**2), torch.mean(ML_nd**2), torch.mean(xL_nd**2), torch.mean(zL_nd**2))


def get_hybrid_schedule_weights(step: int, hybrid_cfg: HybridConfig) -> tuple[float, float]:
    idx = 0
    for i, start in enumerate(hybrid_cfg.schedule_steps):
        if step >= start:
            idx = i
        else:
            break
    return float(hybrid_cfg.schedule_w_data[idx]), float(hybrid_cfg.schedule_w_pde[idx])


def get_hybrid_grad_ratio_target(step: int, hybrid_cfg: HybridConfig) -> float:
    idx = 0
    for i, start in enumerate(hybrid_cfg.schedule_steps):
        if step >= start:
            idx = i
        else:
            break
    return float(hybrid_cfg.schedule_grad_ratio_target[idx])


def compute_data_supervision_loss(
    fields: StateFields,
    ref: Dict[str, torch.Tensor],
    scales: ScaleConfig,
    phys: PhysicalConfig,
    full_cfg: FullStageConfig,
    hybrid_cfg: HybridConfig,
    s_phys: torch.Tensor,
) -> DataLossTerms:
    del phys, full_cfg, s_phys
    # parametric 版里的 data supervision 仍然是“整场监督”，
    # 稀疏的是参数点，不是 s 方向观测点。
    L_x = torch.mean(((fields.x - ref["x"]) / scales.x_scale) ** 2)
    L_z = torch.mean(((fields.z - ref["z"]) / scales.x_scale) ** 2)
    L_theta = torch.mean(((fields.theta - ref["theta"]) / scales.theta_max) ** 2)
    L_T = torch.mean(((fields.T - ref["T"]) / scales.T_scale) ** 2)
    L_M = torch.mean(((fields.M - ref["M"]) / scales.M_scale) ** 2)
    L_Q = torch.mean(((fields.Q - ref["Q"]) / scales.Q_scale) ** 2)
    total = (
        hybrid_cfg.data_w_x * L_x +
        hybrid_cfg.data_w_z * L_z +
        hybrid_cfg.data_w_theta * L_theta +
        hybrid_cfg.data_w_T * L_T +
        hybrid_cfg.data_w_M * L_M +
        hybrid_cfg.data_w_Q * L_Q
    )
    return DataLossTerms(total, L_x, L_z, L_theta, L_T, L_M, L_Q)


def compute_fullstage_losses(
    fields_strong: StateFields,
    fields_bc: StateFields,
    case_params: Dict[str, torch.Tensor],
    phys: PhysicalConfig,
    scales: ScaleConfig,
    full_cfg: FullStageConfig,
    weak_m_loss: Optional[torch.Tensor] = None,
) -> FullStageLossTerms:
    theta_grad_scale = max(scales.theta_max / phys.L, 1.0 / phys.L)
    r_theta = fields_strong.dtheta_ds - fields_strong.M / phys.EI
    r_T = fields_strong.dT_ds + fields_strong.dM_ds * fields_strong.dtheta_ds + fields_strong.qt
    L_pde_theta = torch.mean((r_theta / theta_grad_scale) ** 2)
    L_pde_T = torch.mean((r_T / scales.q_scale) ** 2)
    if weak_m_loss is None:
        r_M = fields_strong.d2M_ds2 - fields_strong.T * fields_strong.dtheta_ds + fields_strong.qn
        L_pde_M = torch.mean((r_M / scales.q_scale) ** 2)
    else:
        L_pde_M = weak_m_loss
    L_pde = full_cfg.pde_w_theta * L_pde_theta + full_cfg.pde_w_T * L_pde_T + full_cfg.pde_w_M * L_pde_M
    bc_diag = compute_boundary_diagnostics(fields_bc, case_params, phys)
    L_bc = full_cfg.bc_w_x * bc_diag.L_bc_x + full_cfg.bc_w_z * bc_diag.L_bc_z
    return FullStageLossTerms(L_pde, L_bc, L_pde_theta, L_pde_M, L_pde_T, bc_diag.L_bc_M0, bc_diag.L_bc_ML, bc_diag.L_bc_x, bc_diag.L_bc_z)


class SingleTargetAnnealer:
    def __init__(self, lambda_bc_init: float, alpha: float, eps: float, lambda_bc_min: float, lambda_bc_max: float) -> None:
        self.lambda_bc = float(lambda_bc_init)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.lambda_bc_min = float(lambda_bc_min)
        self.lambda_bc_max = float(lambda_bc_max)

    @staticmethod
    def _grad_stats(loss: torch.Tensor, params: List[nn.Parameter], retain_graph: bool = True):
        grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=False, allow_unused=True)
        vals = [g.detach().abs().reshape(-1) for g in grads if g is not None]
        if not vals:
            return 0.0, 0.0
        vec = torch.cat(vals)
        return float(vec.max().cpu()), float(vec.mean().cpu())

    def update(self, model: nn.Module, ref_loss: torch.Tensor, target_bc_loss: torch.Tensor) -> Dict[str, float]:
        # 让 BC 权重跟随主损失块与 BC 块的梯度量级做缓慢调整，避免某一项长期压制另一项。
        params = [p for p in model.parameters() if p.requires_grad]
        g_ref_max, _ = self._grad_stats(ref_loss, params, retain_graph=True)
        _, g_bc_mean = self._grad_stats(target_bc_loss, params, retain_graph=True)
        lam_hat = g_ref_max / (g_bc_mean + self.eps)
        self.lambda_bc = float(np.clip((1.0 - self.alpha) * self.lambda_bc + self.alpha * lam_hat, self.lambda_bc_min, self.lambda_bc_max))
        return {"lambda_bc": self.lambda_bc}


def compute_block_grad_l2(loss: torch.Tensor, params: List[nn.Parameter], retain_graph: bool = True) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, create_graph=False, allow_unused=True)
    sq_sum = 0.0
    for g in grads:
        if g is not None:
            sq_sum += float(torch.sum(g.detach() ** 2).cpu())
    return math.sqrt(max(sq_sum, 0.0))


def predict_single_case(
    model: SCRStaticPINNMT, single_case: SingleCaseConfig, ranges: ParameterRanges, phys: PhysicalConfig, net_cfg: NetworkConfig, scales: ScaleConfig, n_nodes: int, device: str
) -> Dict[str, np.ndarray]:
    case = build_case_dict(single_case, device=device)
    s01 = linspace_01(n_nodes, device=device)[None, :]
    inf = compute_inference_fields(model, build_model_inputs(s01, case, ranges), case, phys, net_cfg, scales)
    return {
        "s": (s01[0] * phys.L).cpu().numpy(),
        "theta": inf.theta[0].cpu().numpy(),
        "T": inf.T[0].cpu().numpy(),
        "T0": np.array([float(inf.T0.detach().cpu())], dtype=float),
        "N": inf.T[0].cpu().numpy(),
        "Q": inf.Q[0].cpu().numpy(),
        "M": inf.M[0].cpu().numpy(),
        "x": inf.x[0].cpu().numpy(),
        "z": inf.z[0].cpu().numpy(),
        "penetration": inf.penetration[0].cpu().numpy(),
        "x_target": np.array([phys.x_bottom + single_case.Dx], dtype=float),
        "z_target": np.array([-single_case.ht], dtype=float),
    }


def compute_exact_solution(single_case: SingleCaseConfig, phys: PhysicalConfig) -> Optional[Dict[str, np.ndarray]]:
    if not EXACT_SOLVER_AVAILABLE:
        return None
    exact_phys = ExactPhysicalConfig(
        D_o=phys.D_o, t=phys.t, E_steel=phys.E_steel, rho_s=phys.rho_s, rho_w=phys.rho_w, g=phys.g,
        C_d=phys.C_d, L=phys.L, water_depth=phys.water_depth, x_bottom=phys.x_bottom, k_b=phys.k_b,
    )
    solver_cfg = ExactSolverConfig(verbose=False, use_fast_solver_first=True)
    s, y, success, info = solve_scr_exact(
        Us=single_case.Us, Ub=single_case.Ub, p_exp=single_case.p, Dx=single_case.Dx, ht=single_case.ht, phys=exact_phys, solver_cfg=solver_cfg
    )
    if not success:
        return None
    x, z, theta, H, V, M = y
    N, Q = compute_local_resultants_from_global(theta, H, V)
    return {"s": s, "x": x, "z": z, "theta": theta, "N": N, "Q": Q, "M": M, "info": info}


def exact_solution_to_reference(exact: Dict[str, np.ndarray], device: str, s_target: np.ndarray) -> Dict[str, torch.Tensor]:
    s_exact = exact["s"]

    def interp(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.interp(s_target, s_exact, arr), dtype=torch.float32, device=device)[None, :]

    return {
        "s": torch.tensor(s_target, dtype=torch.float32, device=device)[None, :],
        "x": interp(exact["x"]),
        "z": interp(exact["z"]),
        "theta": interp(exact["theta"]),
        "T": interp(exact["N"]),
        "M": interp(exact["M"]),
        "Q": interp(exact["Q"]),
    }


def catenary_reference_to_reference(catenary_ref: Dict[str, torch.Tensor], phys: PhysicalConfig, device: str, s_target: np.ndarray) -> Dict[str, torch.Tensor]:
    s_src = catenary_ref["s"][0].detach().cpu().numpy()

    def interp_tensor(key: str) -> torch.Tensor:
        vals = np.interp(s_target, s_src, catenary_ref[key][0].detach().cpu().numpy())
        return torch.tensor(vals, dtype=torch.float32, device=device)[None, :]

    theta = interp_tensor("theta")
    T = interp_tensor("T")
    x = interp_tensor("x")
    z = interp_tensor("z")
    dtheta_ds = np.gradient(theta[0].detach().cpu().numpy(), s_target)
    d2theta_ds2 = np.gradient(dtheta_ds, s_target)
    M = torch.tensor(phys.EI * dtheta_ds, dtype=torch.float32, device=device)[None, :]
    Q = torch.tensor(phys.EI * d2theta_ds2, dtype=torch.float32, device=device)[None, :]
    return {
        "s": torch.tensor(s_target, dtype=torch.float32, device=device)[None, :],
        "x": x, "z": z, "theta": theta, "T": T, "M": M, "Q": Q,
    }


def save_curve_bundle_npz(save_path: Path, pred: Dict[str, np.ndarray], exact: Optional[Dict[str, np.ndarray]] = None) -> None:
    arrays = {f"pred_{k}": np.asarray(v) for k, v in pred.items()}
    if exact is not None:
        for k, v in exact.items():
            if k != "info":
                arrays[f"exact_{k}"] = np.asarray(v)
    np.savez(save_path, **arrays)


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.semilogy(history["adam_total"], label="total")
    ax1.semilogy(history["adam_data"], label="data")
    ax1.semilogy(history["adam_pde"], label="pde")
    ax1.semilogy(history["adam_bc"], label="bc")
    ax1.grid(True, ls=":")
    ax1.legend()
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.semilogy(history["adam_data_theta"], label="data_theta")
    ax2.semilogy(history["adam_data_T"], label="data_T")
    ax2.semilogy(history["adam_data_M"], label="data_M")
    ax2.semilogy(history["adam_pde_theta"], label="pde_theta")
    ax2.semilogy(history["adam_pde_M"], label="pde_M")
    ax2.semilogy(history["adam_pde_T"], label="pde_T")
    ax2.grid(True, ls=":")
    ax2.legend()
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.semilogy(history["adam_ratio_data_pde"], label="data/pde")
    ax3.semilogy(history["adam_ratio_weighted"], label="weighted data/pde")
    ax3.semilogy(history["adam_grad_ratio_target"], label="grad target")
    ax3.semilogy(history["adam_grad_ratio_weighted"], label="grad weighted")
    ax3.grid(True, ls=":")
    ax3.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_prediction_vs_exact(pred: Dict[str, np.ndarray], exact: Optional[Dict[str, np.ndarray]], save_path: Path) -> None:
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(pred["x"], pred["z"], lw=2, label="PINN")
    ax1.scatter(pred["x_target"][0], pred["z_target"][0], c="r", label="Target")
    if exact is not None:
        ax1.plot(exact["x"], exact["z"], "--", lw=2, label="Exact")
    ax1.grid(True, ls=":")
    ax1.legend()
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(pred["s"], pred["theta"], lw=2, label="PINN")
    if exact is not None:
        ax2.plot(exact["s"], exact["theta"], "--", lw=2, label="Exact")
    ax2.grid(True, ls=":")
    ax2.legend()
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pred["s"], pred["T"] / 1e3, lw=2, label="PINN T")
    ax3.plot(pred["s"], pred["Q"] / 1e3, lw=1.5, label="PINN Q")
    if exact is not None:
        ax3.plot(exact["s"], exact["N"] / 1e3, "--", lw=2, label="Exact N")
        ax3.plot(exact["s"], exact["Q"] / 1e3, "--", lw=2, label="Exact Q")
    ax3.grid(True, ls=":")
    ax3.legend(ncol=2)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(pred["s"], pred["M"] / 1e3, lw=2, label="PINN")
    if exact is not None:
        ax4.plot(exact["s"], exact["M"] / 1e3, "--", lw=2, label="Exact")
    ax4.grid(True, ls=":")
    ax4.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def repeat_s_grid(s01_single: torch.Tensor, batch_size: int) -> torch.Tensor:
    return s01_single.repeat(batch_size, 1)


def case_geometry_is_admissible(dx: float, ht: float, phys: PhysicalConfig) -> bool:
    # 这里只做最基本的几何筛选：
    # 1. 管长至少大于两端点直线距离
    # 2. 对当前单调 2D SCR 构型，L 不应大于 Dx + 垂向跨度
    vertical = phys.water_depth - ht
    return math.hypot(dx, vertical) <= phys.L and phys.L <= dx + vertical


def sample_case_params(batch_size: int, ranges: ParameterRanges, rng: np.random.Generator, device: str, phys: Optional[PhysicalConfig] = None) -> Dict[str, torch.Tensor]:
    # physics batch 在线随机采样；data case bank 也是从同一参数域里抽样，只是后续固定下来反复使用。
    samples: List[SingleCaseConfig] = []
    while len(samples) < batch_size:
        ub = float(rng.uniform(ranges.Ub_min, ranges.Ub_max))
        us = float(rng.uniform(max(ub, ranges.Us_min), ranges.Us_max))
        p = float(rng.uniform(ranges.p_min, ranges.p_max))
        dx = float(rng.uniform(ranges.Dx_min, ranges.Dx_max))
        ht = float(rng.uniform(ranges.ht_min, ranges.ht_max))
        if phys is not None and not case_geometry_is_admissible(dx, ht, phys):
            continue
        samples.append(SingleCaseConfig(Us=us, Ub=ub, p=p, Dx=dx, ht=ht))
    return {
        "Us": torch.tensor([c.Us for c in samples], dtype=torch.float32, device=device),
        "Ub": torch.tensor([c.Ub for c in samples], dtype=torch.float32, device=device),
        "p": torch.tensor([c.p for c in samples], dtype=torch.float32, device=device),
        "Dx": torch.tensor([c.Dx for c in samples], dtype=torch.float32, device=device),
        "ht": torch.tensor([c.ht for c in samples], dtype=torch.float32, device=device),
    }


def case_params_to_single_case(case_params: Dict[str, torch.Tensor], index: int) -> SingleCaseConfig:
    return SingleCaseConfig(
        Us=float(case_params["Us"][index].detach().cpu()),
        Ub=float(case_params["Ub"][index].detach().cpu()),
        p=float(case_params["p"][index].detach().cpu()),
        Dx=float(case_params["Dx"][index].detach().cpu()),
        ht=float(case_params["ht"][index].detach().cpu()),
    )


def build_reference(single_case: SingleCaseConfig, phys: PhysicalConfig, device: str, s_target: np.ndarray, reference_source: str) -> Optional[Dict[str, torch.Tensor]]:
    source = reference_source.lower()
    if source == "exact":
        exact = compute_exact_solution(single_case, phys)
        if exact is None:
            return None
        return exact_solution_to_reference(exact, device=device, s_target=s_target)
    if source == "catenary":
        catenary_ref = generate_catenary_reference(phys, single_case, n_nodes=max(len(s_target), 128), device=device)
        return catenary_reference_to_reference(catenary_ref, phys, device=device, s_target=s_target)
    raise ValueError(f"Unsupported reference source: {reference_source}")


def build_sparse_case_bank(
    sparse_cfg: SparseParametricDataConfig,
    phys: PhysicalConfig,
    ranges: ParameterRanges,
    device: str,
    rng: np.random.Generator,
    n_nodes: int,
) -> List[Dict[str, object]]:
    # data case bank 的“稀疏”体现在参数空间里。
    # 如果 reference_source='exact'，这里会持续重采样直到收集到足够多的 exact-success cases。
    s_data = np.linspace(0.0, phys.L, n_nodes, dtype=float)
    records: List[Dict[str, object]] = []
    attempts = 0
    while len(records) < sparse_cfg.num_cases:
        attempts += 1
        if attempts > sparse_cfg.max_sample_attempts:
            raise RuntimeError(f"Failed to collect {sparse_cfg.num_cases} data cases in {attempts} attempts")
        sampled = sample_case_params(1, ranges, rng, device=device, phys=phys)
        case = case_params_to_single_case(sampled, 0)
        full_ref = build_reference(case, phys, device, s_data, sparse_cfg.reference_source)
        if full_ref is None:
            continue
        exact_full = compute_exact_solution(case, phys) if EXACT_SOLVER_AVAILABLE else None
        records.append({"single_case": case, "case_params": build_case_dict(case, device=device), "full_ref": full_ref, "exact_full": exact_full})
    return records


def collate_sparse_case_batch(records: List[Dict[str, object]], indices: np.ndarray, device: str) -> Dict[str, Dict[str, torch.Tensor]]:
    case_keys = ["Us", "Ub", "p", "Dx", "ht"]
    ref_keys = ["s", "theta", "T", "M", "Q", "x", "z"]
    case_params = {key: torch.cat([records[int(idx)]["case_params"][key] for idx in indices], dim=0).to(device) for key in case_keys}
    full_ref = {key: torch.cat([records[int(idx)]["full_ref"][key] for idx in indices], dim=0).to(device) for key in ref_keys}
    return {"case_params": case_params, "full_ref": full_ref}


def compute_weak_m_balance_loss_batch(
    model: SCRStaticPINNMT,
    element_edges: np.ndarray,
    case_params: Dict[str, torch.Tensor],
    ranges: ParameterRanges,
    phys: PhysicalConfig,
    net_cfg: NetworkConfig,
    scales: ScaleConfig,
    full_cfg: FullStageConfig,
    device: str,
) -> torch.Tensor:
    # M 方程继续用 element-wise weak form，和 single-case hybrid v1_0 保持一致。
    batch_size = int(case_params["Us"].shape[0])
    n_quad = max(int(full_cfg.gauss_points_per_elem), 1)
    s01_q = repeat_s_grid(build_element_midpoints(element_edges, phys, device=device, n_pts_per_elem=n_quad), batch_size)
    model_inputs_q = build_model_inputs(s01_q, case_params, ranges)
    model_inputs_q.requires_grad_(True)
    fields_q = compute_training_fields(model, model_inputs_q, case_params, phys, net_cfg, scales)
    s_q = s01_q * phys.L
    residuals = []
    start = 0
    for a, b in zip(element_edges[:-1], element_edges[1:]):
        h = float(b - a)
        sl = slice(start, start + n_quad)
        xi = 2.0 * (s_q[:, sl] - a) / h - 1.0
        w = 1.0 - xi**2
        w_s = -4.0 * xi / h
        integrand = -fields_q.dM_ds[:, sl] * w_s - (fields_q.T[:, sl] * fields_q.dtheta_ds[:, sl] - fields_q.qn[:, sl]) * w
        residuals.append(torch.mean(integrand, dim=-1))
        start += n_quad
    return torch.mean((torch.stack(residuals, dim=1) / scales.q_scale) ** 2)


def train_hybrid_parametric_sparse(
    phys: PhysicalConfig,
    scales: ScaleConfig,
    ranges: ParameterRanges,
    net_cfg: NetworkConfig,
    hybrid_cfg: HybridConfig,
    full_cfg: FullStageConfig,
    sparse_cfg: SparseParametricDataConfig,
    train_cfg: ParametricTrainingConfig,
):
    set_seed(train_cfg.seed)
    out_dir = ensure_dir(train_cfg.output_dir)
    rng = np.random.default_rng(train_cfg.seed)
    # Keep the hybrid schedule aligned with the actual training length used here.
    hybrid_cfg.adam_steps = int(train_cfg.adam_steps)
    hybrid_cfg.ensure_schedule(train_cfg.adam_steps)
    model = SCRStaticPINNMT(net_cfg).to(train_cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=hybrid_cfg.lr, weight_decay=hybrid_cfg.weight_decay)
    element_edges = build_explicit_mesh_edges(phys, full_cfg)
    s_strong_single = build_element_midpoints(element_edges, phys, device=train_cfg.device, n_pts_per_elem=full_cfg.strong_points_per_elem)
    s_bc_single = linspace_01(train_cfg.n_nodes, device=train_cfg.device)[None, :]
    sparse_records = build_sparse_case_bank(sparse_cfg, phys, ranges, train_cfg.device, rng, train_cfg.n_nodes)
    anchor_case = sparse_records[0]["single_case"]
    anchor_exact = sparse_records[0]["exact_full"]

    history = {k: [] for k in [
        "adam_total","adam_data","adam_pde","adam_bc","adam_data_x","adam_data_z","adam_data_theta","adam_data_T",
        "adam_data_M","adam_data_Q","adam_pde_theta","adam_pde_M","adam_pde_T","adam_bc_M0","adam_bc_ML",
        "adam_bc_x","adam_bc_z","w_data","w_pde","lambda_bc","adam_ratio_data_pde","adam_ratio_weighted",
        "adam_grad_ratio_target","adam_grad_ratio_weighted","adam_data_pde_corr","grad_adam_steps",
        "grad_adam_data","grad_adam_pde","grad_adam_bc","lr","adam_T0","adam_T_min","adam_T_max",
        "adam_Tres_min","adam_Tres_max"
    ]}

    annealer = SingleTargetAnnealer(full_cfg.lambda_bc_init, full_cfg.anneal_alpha, full_cfg.anneal_eps, full_cfg.lambda_bc_min, full_cfg.lambda_bc_max)
    grad_params = [p for p in model.parameters() if p.requires_grad]
    data_pde_corr = 1.0
    t0 = time.time()

    print("=" * 88)
    print("Starting parametric sparse-in-parameter 3V hybrid PINN training")
    print(f"Device          : {train_cfg.device}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Data cases      : {sparse_cfg.num_cases} total | batch={sparse_cfg.batch_size}")
    print(f"Physics batch   : {train_cfg.physics_batch_size}")
    print(f"Data source     : {sparse_cfg.reference_source}")
    print("=" * 88)

    for step in range(1, train_cfg.adam_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # PDE/BC 在连续参数域里通过在线随机采样施加。
        physics_case_params = sample_case_params(train_cfg.physics_batch_size, ranges, rng, train_cfg.device, phys=phys)
        model_inputs_strong = build_model_inputs(repeat_s_grid(s_strong_single, train_cfg.physics_batch_size), physics_case_params, ranges)
        model_inputs_strong.requires_grad_(True)
        fields_strong = compute_training_fields(model, model_inputs_strong, physics_case_params, phys, net_cfg, scales)

        model_inputs_bc = build_model_inputs(repeat_s_grid(s_bc_single, train_cfg.physics_batch_size), physics_case_params, ranges)
        model_inputs_bc.requires_grad_(True)
        fields_bc = compute_training_fields(model, model_inputs_bc, physics_case_params, phys, net_cfg, scales)
        weak_m_loss = compute_weak_m_balance_loss_batch(model, element_edges, physics_case_params, ranges, phys, net_cfg, scales, full_cfg, train_cfg.device)
        phys_losses = compute_fullstage_losses(fields_strong, fields_bc, physics_case_params, phys, scales, full_cfg, weak_m_loss=weak_m_loss)

        # data loss 只作用在参数空间里少量离散 anchors 上，但每个 anchor 仍然是整场监督。
        data_indices = rng.choice(len(sparse_records), size=min(sparse_cfg.batch_size, len(sparse_records)), replace=False)
        data_batch = collate_sparse_case_batch(sparse_records, data_indices, train_cfg.device)
        model_inputs_data = build_model_inputs(data_batch["full_ref"]["s"] / phys.L, data_batch["case_params"], ranges)
        model_inputs_data.requires_grad_(True)
        fields_data = compute_training_fields(model, model_inputs_data, data_batch["case_params"], phys, net_cfg, scales)
        s_phys_data = 0.5 * (model_inputs_data[..., 0] + 1.0) * phys.L
        data_losses = compute_data_supervision_loss(fields_data, data_batch["full_ref"], scales, phys, full_cfg, hybrid_cfg, s_phys_data)

        base_w_data, base_w_pde = get_hybrid_schedule_weights(step, hybrid_cfg)
        target_grad_ratio = get_hybrid_grad_ratio_target(step, hybrid_cfg)
        correction_now = step == 1 or step % hybrid_cfg.grad_balance_every == 0
        grad_log_now = step == 1 or step % train_cfg.print_every == 0
        g_data = g_pde = g_bc = None
        if correction_now or grad_log_now:
            g_data = compute_block_grad_l2(data_losses.total, grad_params, retain_graph=True)
            g_pde = compute_block_grad_l2(phys_losses.pde, grad_params, retain_graph=True)
            if grad_log_now:
                g_bc = compute_block_grad_l2(phys_losses.bc, grad_params, retain_graph=True)
        if correction_now:
            corr_hat = target_grad_ratio * base_w_pde * g_pde / max(base_w_data * g_data, hybrid_cfg.grad_balance_eps)
            data_pde_corr = float(np.clip((1.0 - hybrid_cfg.grad_balance_alpha) * data_pde_corr + hybrid_cfg.grad_balance_alpha * corr_hat, hybrid_cfg.grad_balance_corr_min, hybrid_cfg.grad_balance_corr_max))

        w_data = base_w_data * data_pde_corr
        w_pde = base_w_pde
        core_loss = w_data * data_losses.total + w_pde * phys_losses.pde
        if step >= full_cfg.anneal_start_step and (step - full_cfg.anneal_start_step) % full_cfg.anneal_every == 0:
            annealer.update(model=model, ref_loss=core_loss, target_bc_loss=phys_losses.bc)
        total_loss = core_loss + annealer.lambda_bc * phys_losses.bc
        total_loss.backward()
        if hybrid_cfg.grad_clip is not None and hybrid_cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hybrid_cfg.grad_clip)
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        history["adam_total"].append(float(total_loss.detach().cpu()))
        history["adam_data"].append(float(data_losses.total.detach().cpu()))
        history["adam_pde"].append(float(phys_losses.pde.detach().cpu()))
        history["adam_bc"].append(float(phys_losses.bc.detach().cpu()))
        history["adam_data_x"].append(float(data_losses.x.detach().cpu()))
        history["adam_data_z"].append(float(data_losses.z.detach().cpu()))
        history["adam_data_theta"].append(float(data_losses.theta.detach().cpu()))
        history["adam_data_T"].append(float(data_losses.T.detach().cpu()))
        history["adam_data_M"].append(float(data_losses.M.detach().cpu()))
        history["adam_data_Q"].append(float(data_losses.Q.detach().cpu()))
        history["adam_pde_theta"].append(float(phys_losses.pde_theta.detach().cpu()))
        history["adam_pde_M"].append(float(phys_losses.pde_M.detach().cpu()))
        history["adam_pde_T"].append(float(phys_losses.pde_T.detach().cpu()))
        history["adam_bc_M0"].append(float(phys_losses.bc_M0.detach().cpu()))
        history["adam_bc_ML"].append(float(phys_losses.bc_ML.detach().cpu()))
        history["adam_bc_x"].append(float(phys_losses.bc_x.detach().cpu()))
        history["adam_bc_z"].append(float(phys_losses.bc_z.detach().cpu()))
        history["w_data"].append(float(w_data))
        history["w_pde"].append(float(w_pde))
        history["lambda_bc"].append(float(annealer.lambda_bc))
        history["lr"].append(float(lr))
        pde_val = history["adam_pde"][-1]
        data_val = history["adam_data"][-1]
        history["adam_ratio_data_pde"].append(data_val / max(pde_val, 1.0e-12))
        history["adam_ratio_weighted"].append((w_data * data_val) / max(w_pde * pde_val, 1.0e-12))
        grad_ratio_weighted = (w_data * float(g_data)) / max(w_pde * float(g_pde), hybrid_cfg.grad_balance_eps) if g_data is not None and g_pde is not None else float("nan")
        history["adam_grad_ratio_target"].append(float(target_grad_ratio))
        history["adam_grad_ratio_weighted"].append(float(grad_ratio_weighted))
        history["adam_data_pde_corr"].append(float(data_pde_corr))
        T_res_now = (fields_data.T - fields_data.T0).detach()
        history["adam_T0"].append(float(fields_data.T0.detach().cpu()))
        history["adam_T_min"].append(float(fields_data.T.detach().min().cpu()))
        history["adam_T_max"].append(float(fields_data.T.detach().max().cpu()))
        history["adam_Tres_min"].append(float(T_res_now.min().cpu()))
        history["adam_Tres_max"].append(float(T_res_now.max().cpu()))

        if grad_log_now:
            history["grad_adam_steps"].append(step)
            history["grad_adam_data"].append(g_data)
            history["grad_adam_pde"].append(g_pde)
            history["grad_adam_bc"].append(g_bc)
            print(
                f"[Adam] step={step:6d} | lr={lr:.2e} | L={history['adam_total'][-1]:.3e} | "
                f"Ldata={history['adam_data'][-1]:.3e} | Lpde={history['adam_pde'][-1]:.3e} | "
                f"Lbc={history['adam_bc'][-1]:.3e} | data(th,T,M,Q)=({history['adam_data_theta'][-1]:.2e}, "
                f"{history['adam_data_T'][-1]:.2e}, {history['adam_data_M'][-1]:.2e}, {history['adam_data_Q'][-1]:.2e}) | "
                f"pde(th,M,T)=({history['adam_pde_theta'][-1]:.2e}, {history['adam_pde_M'][-1]:.2e}, {history['adam_pde_T'][-1]:.2e}) | "
                f"ratio(data/pde,weighted)=({history['adam_ratio_data_pde'][-1]:.2e}, {history['adam_ratio_weighted'][-1]:.2e}) | "
                f"elapsed={time.time() - t0:.1f}s"
            )

        if step % train_cfg.plot_every == 0:
            pred = predict_single_case(model, anchor_case, ranges, phys, net_cfg, scales, train_cfg.n_nodes, train_cfg.device)
            plot_prediction_vs_exact(pred, anchor_exact, out_dir / f"prediction_adam_step_{step:06d}.png")

    pred_anchor = predict_single_case(model, anchor_case, ranges, phys, net_cfg, scales, train_cfg.n_nodes, train_cfg.device)
    plot_training_history(history, out_dir / "training_history.png")
    plot_prediction_vs_exact(pred_anchor, anchor_exact, out_dir / "final_prediction_anchor_case.png")
    torch.save(model.state_dict(), out_dir / "scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.pth")
    with open(out_dir / "history_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "physical": asdict(phys),
            "scales": asdict(scales),
            "ranges": asdict(ranges),
            "network": asdict(net_cfg),
            "hybrid": asdict(hybrid_cfg),
            "full_stage": asdict(full_cfg),
            "sparse_parametric_data": asdict(sparse_cfg),
            "training": asdict(train_cfg),
            "anchor_case": asdict(anchor_case),
            "exact_solver_available": EXACT_SOLVER_AVAILABLE,
            "exact_solver_import_error": EXACT_SOLVER_IMPORT_ERROR,
        }, f, indent=2)
    with open(out_dir / "sparse_case_bank.json", "w", encoding="utf-8") as f:
        json.dump([asdict(rec["single_case"]) for rec in sparse_records], f, indent=2)
    save_curve_bundle_npz(out_dir / "final_anchor_curves.npz", pred_anchor, exact=anchor_exact)
    print("=" * 88)
    print("Parametric sparse-in-parameter hybrid training finished.")
    print(f"Artifacts saved in: {out_dir.resolve()}")
    print("=" * 88)
    return model, history, sparse_records


def main() -> None:
    phys = PhysicalConfig()
    scales = ScaleConfig().build_from_physics(phys)
    ranges = ParameterRanges()
    net_cfg = NetworkConfig()
    hybrid_cfg = HybridConfig()
    full_cfg = FullStageConfig()
    sparse_cfg = SparseParametricDataConfig()
    train_cfg = ParametricTrainingConfig()
    train_hybrid_parametric_sparse(phys, scales, ranges, net_cfg, hybrid_cfg, full_cfg, sparse_cfg, train_cfg)


if __name__ == "__main__":
    main()
