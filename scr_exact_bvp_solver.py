#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scr_exact_bvp_solver.py

Numerical reference solver for the static 2D SCR configuration (inextensible version).

Compared with the uploaded scr_solver.py, this version:
1. uses the variable upper-end geometry inputs (Dx, ht),
2. fixes the lower-end x-coordinate explicitly and fixes the lower-end z-coordinate
   at the local vertical spring-settlement equilibrium depth z = z_bed - w_eff / k_b,
3. uses a catenary-based gravity-consistent initial guess,
4. uses staged continuation:
       Stage 0: gravity only
       Stage 1: gravity + drag ramp-up
       Stage 2: gravity + drag + contact ramp-up
5. uses a unilateral contact law
        penetration = 0  -->  contact reaction = 0
6. neglects axial extension in the centerline kinematics, matching the PINN model
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from scipy.integrate import solve_bvp
from scipy.optimize import root_scalar


# =============================================================================
# 1. Physical and solver configuration
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
    def z_bottom(self) -> float:
        return -self.water_depth

    @property
    def z_bed(self) -> float:
        return -self.water_depth

    @property
    def z_lower_equilibrium(self) -> float:
        """
        Lower-end vertical position when the seabed spring locally balances the
        effective submerged self-weight:
            k_b * (z_bed - z_eq) = w_eff
        i.e.
            z_eq = z_bed - w_eff / k_b
        """
        return self.z_bed - self.w_eff / self.k_b

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
class SolverConfig:
    n_init: int = 300
    n_eval: int = 256

    use_fast_solver_first: bool = True

    tol_stage0: float = 1.0e-4
    tol_stage1: float = 3.0e-4
    tol_stage2: float = 3.0e-4

    max_nodes: int = 50000
    max_nodes_fast: int = 12000

    drag_factors_stage1: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0)
    contact_factors_stage2: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.0)
    current_factors_fast: Tuple[float, ...] = (0.20, 0.40, 0.60, 0.80, 1.0)

    continuation_regrid_nodes: int = 400

    verbose: bool = True


# =============================================================================
# 2. Contact and current model
# =============================================================================

def positive_part(p: np.ndarray) -> np.ndarray:
    return np.maximum(p, 0.0)


def contact_reaction(z: np.ndarray, phys: PhysicalConfig, contact_factor: float) -> np.ndarray:
    penetration = phys.z_bed - z
    p_pos = positive_part(penetration)
    return contact_factor * phys.k_b * p_pos


def current_velocity(z: np.ndarray, Us: float, Ub: float, p_exp: float, phys: PhysicalConfig) -> np.ndarray:
    depth_ratio = (z - phys.z_bed) / phys.water_depth
    depth_ratio = np.clip(depth_ratio, 0.0, 1.0)
    depth_ratio = np.clip(depth_ratio, 1.0e-12, 1.0)
    return Ub + (Us - Ub) * depth_ratio**p_exp


# =============================================================================
# 3. Strong-form ODE system (global-force formulation)
# =============================================================================

def ode_system(
    s: np.ndarray,
    y: np.ndarray,
    phys: PhysicalConfig,
    Us: float,
    Ub: float,
    p_exp: float,
    drag_factor: float,
    contact_factor: float,
) -> np.ndarray:
    """
    State:
        y = [x, z, theta, H, V, M]
    """
    x, z, theta, H, V, M = y

    T = H * np.cos(theta) + V * np.sin(theta)

    Uc = current_velocity(z, Us, Ub, p_exp, phys)
    V_n = Uc * np.sin(theta)

    f_Dn = drag_factor * 0.5 * phys.rho_w * phys.C_d * phys.D_o * V_n * np.abs(V_n)
    f_Dx = f_Dn * np.sin(theta)
    f_Dz = -f_Dn * np.cos(theta)

    f_cont = contact_reaction(z, phys=phys, contact_factor=contact_factor)

    # Inextensible centerline kinematics to match the reduced PINN model.
    dx_ds = np.cos(theta)
    dz_ds = np.sin(theta)
    dtheta_ds = M / phys.EI

    dH_ds = -f_Dx
    dV_ds = phys.w_eff - f_Dz - f_cont

    dM_ds = H * np.sin(theta) - V * np.cos(theta)

    return np.vstack((dx_ds, dz_ds, dtheta_ds, dH_ds, dV_ds, dM_ds))


# =============================================================================
# 4. Boundary conditions
# =============================================================================

def boundary_conditions(
    ya: np.ndarray,
    yb: np.ndarray,
    x_bottom: float,
    z_lower: float,
    x_top: float,
    z_top: float,
) -> np.ndarray:
    """
    Pinned-pinned SCR with fixed end positions:
        x(0)=x_bottom, z(0)=z_lower, M(0)=0
        x(L)=x_top,    z(L)=z_top,   M(L)=0

    Here z_lower is *not* the seabed elevation itself, but the local spring-
    settlement equilibrium depth:
        z_lower = z_bed - w_eff / k_b
    so that the contact spring at the lower end balances the effective
    submerged self-weight in the gravity-only horizontal laydown state.
    """
    return np.array([
        ya[0] - x_bottom,
        ya[1] - z_lower,
        ya[5],
        yb[0] - x_top,
        yb[1] - z_top,
        yb[5],
    ])


# =============================================================================
# 5. Catenary-based initial guess
# =============================================================================

def solve_catenary_parameter(delta_x: float, delta_z: float, L: float) -> float:
    """
    Solve:
        sqrt(L^2 - delta_z^2) = 2 a sinh(delta_x / (2a))
    """
    rhs = math.sqrt(max(L**2 - delta_z**2, 1.0e-12))
    if rhs <= 0.0:
        raise ValueError("Invalid catenary geometry: rhs must be positive.")

    def log_two_sinh(x: float) -> float:
        if x <= 0.0:
            return float("-inf")
        if x > 20.0:
            return x + math.log1p(-math.exp(-2.0 * x))
        return math.log(2.0 * math.sinh(x))

    log_rhs = math.log(rhs)

    def f(a: float) -> float:
        x = delta_x / (2.0 * a)
        return math.log(a) + log_two_sinh(x) - log_rhs

    res = root_scalar(f, bracket=[1.0, 1.0e7], method="bisect")
    if not res.converged:
        raise RuntimeError("Failed to solve for catenary parameter a.")
    return float(res.root)


def generate_catenary_initial_guess(
    phys: PhysicalConfig,
    Dx: float,
    ht: float,
    n_init: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gravity-consistent catenary-like initial guess between:
        lower end: (x_bottom, z_bottom)
        upper end: (x_bottom + Dx, -ht)
    """
    x0 = phys.x_bottom
    z0 = phys.z_lower_equilibrium
    x1 = phys.x_bottom + Dx
    z1 = -ht

    delta_x = x1 - x0
    delta_z = z1 - z0

    straight = math.sqrt(delta_x**2 + delta_z**2)
    if phys.L <= straight:
        raise ValueError(
            f"Invalid geometry: L={phys.L:.3f} <= straight distance={straight:.3f}"
        )

    a = solve_catenary_parameter(delta_x, delta_z, phys.L)

    d = delta_x / (2.0 * a)
    ratio = np.clip(delta_z / phys.L, -0.999999, 0.999999)
    m = np.arctanh(ratio)

    u1 = m - d
    u2 = m + d

    c = (x0 + x1) / 2.0 - a * m
    C = z0 - a * np.cosh(u1)

    s_init = np.linspace(0.0, phys.L, n_init)

    u = np.arcsinh(s_init / a + np.sinh(u1))

    x = c + a * u
    z = C + a * np.cosh(u)

    theta = np.arctan(np.sinh(u))
    H = np.full_like(s_init, a * phys.w_eff)
    V = H * np.tan(theta)
    M = np.zeros_like(s_init)

    y_init = np.vstack((x, z, theta, H, V, M))
    return s_init, y_init


def generate_touchdown_initial_guess(
    phys: PhysicalConfig,
    Dx: float,
    ht: float,
    n_init: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast initial guess inspired by `scr_solver.py`:
    - a horizontal seabed laydown segment
    - a suspended catenary segment with zero slope at touchdown
    """
    s = np.linspace(0.0, phys.L, n_init)
    z_rest = phys.z_bed - phys.w_eff / phys.k_b
    x_top = phys.x_bottom + Dx
    z_top = -ht
    delta_z = z_top - z_rest

    if delta_z <= 0.0:
        raise ValueError("Invalid touchdown initial guess geometry: top must be above touchdown level.")

    def horizontal_mismatch(Lb: float) -> float:
        Ls = phys.L - Lb
        if Ls <= delta_z:
            return -x_top
        a = (Ls * Ls - delta_z * delta_z) / (2.0 * delta_z)
        if a <= 0.0:
            return -x_top
        x_span = a * np.arcsinh(Ls / a)
        return Lb + x_span - x_top

    Lb_hi = min(x_top, phys.L - delta_z - 1.0e-6)
    if Lb_hi <= 1.0e-6:
        return generate_catenary_initial_guess(phys, Dx, ht, n_init)

    f_lo = horizontal_mismatch(0.0)
    f_hi = horizontal_mismatch(Lb_hi)
    if f_lo * f_hi > 0.0:
        return generate_catenary_initial_guess(phys, Dx, ht, n_init)

    res = root_scalar(horizontal_mismatch, bracket=[0.0, Lb_hi], method="bisect")
    if (not res.converged) or res.root is None:
        return generate_catenary_initial_guess(phys, Dx, ht, n_init)

    Lb = float(res.root)
    Ls = phys.L - Lb
    a = (Ls * Ls - delta_z * delta_z) / (2.0 * delta_z)
    H0 = a * phys.w_eff

    y_init = np.zeros((6, n_init))
    for i, si in enumerate(s):
        if si <= Lb:
            y_init[:, i] = [phys.x_bottom + si, z_rest, 0.0, H0, 0.0, 0.0]
        else:
            s_prime = si - Lb
            x_i = phys.x_bottom + Lb + a * np.arcsinh(s_prime / a)
            z_i = z_rest + a * (np.sqrt(1.0 + (s_prime / a) ** 2) - 1.0)
            theta_i = np.arctan(s_prime / a)
            y_init[:, i] = [x_i, z_i, theta_i, H0, phys.w_eff * s_prime, 0.0]

    return s, y_init


def regrid_solution(sol, L: float, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-interpolate a converged BVP solution onto a fresh uniform mesh.
    This keeps continuation stages from inheriting an excessively refined mesh.
    """
    s_new = np.linspace(0.0, L, n_nodes)
    y_new = sol.sol(s_new)
    return s_new, y_new


def solve_scr_exact_fast(
    Us: float,
    Ub: float,
    p_exp: float,
    Dx: float,
    ht: float,
    phys: PhysicalConfig,
    solver_cfg: SolverConfig,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
    """
    Faster continuation strategy modeled after `scr_solver.py`.
    """
    x_bottom = phys.x_bottom
    z_lower = phys.z_lower_equilibrium
    x_top = phys.x_bottom + Dx
    z_top = -ht

    try:
        s_current, y_current = generate_touchdown_initial_guess(
            phys=phys, Dx=Dx, ht=ht, n_init=solver_cfg.n_init
        )
    except Exception as exc:
        return None, None, False, {
            "stage": "fast_initial_guess",
            "message": f"Fast initial guess generation failed: {exc}",
        }

    def bc_wrap(ya, yb):
        return boundary_conditions(
            ya, yb,
            x_bottom=x_bottom,
            z_lower=z_lower,
            x_top=x_top,
            z_top=z_top,
        )

    if solver_cfg.verbose:
        print("=" * 80)
        print("SCR exact solver (fast mode, inextensible)")
        print(f"Target geometry: Dx={Dx:.3f} m, ht={ht:.3f} m")
        print(f"Lower-end z    : z_lower={z_lower:.6f} m (spring-settlement equilibrium)")
        print(f"Current params : Us={Us:.3f}, Ub={Ub:.3f}, p={p_exp:.4f}")
        print("=" * 80)
        print("Fast continuation: current ramp-up with full contact law active")

    sol = None
    for lf in solver_cfg.current_factors_fast:
        def ode_wrap(s, y, lf_local=float(lf)):
            return ode_system(
                s=s,
                y=y,
                phys=phys,
                Us=Us * lf_local,
                Ub=Ub * lf_local,
                p_exp=p_exp,
                drag_factor=1.0,
                contact_factor=1.0,
            )

        sol = solve_bvp(
            ode_wrap,
            bc_wrap,
            s_current,
            y_current,
            tol=solver_cfg.tol_stage1,
            max_nodes=solver_cfg.max_nodes_fast,
            verbose=0,
        )
        if not sol.success:
            return None, None, False, {
                "stage": "fast_stage",
                "message": f"Fast continuation failed at current factor {lf}: {sol.message}",
            }
        s_current, y_current = regrid_solution(sol, phys.L, solver_cfg.continuation_regrid_nodes)
        if solver_cfg.verbose:
            print(f"  current factor = {lf:.2f}  --> success")

    s_eval = np.linspace(0.0, phys.L, solver_cfg.n_eval)
    y_eval = sol.sol(s_eval)
    return s_eval, y_eval, True, {
        "stage": "done_fast",
        "message": "Success (fast solver)",
        "solver_message": sol.message,
        "phys": asdict(phys),
        "solver_cfg": asdict(solver_cfg),
        "target_geometry": {
            "Dx": Dx,
            "ht": ht,
            "x_top": x_top,
            "z_top": z_top,
            "z_lower": z_lower,
            "z_lower": z_lower,
        },
        "current": {
            "Us": Us,
            "Ub": Ub,
            "p_exp": p_exp,
        },
    }


# =============================================================================
# 6. Robust continuation solver
# =============================================================================

def solve_scr_exact(
    Us: float,
    Ub: float,
    p_exp: float,
    Dx: float,
    ht: float,
    phys: Optional[PhysicalConfig] = None,
    solver_cfg: Optional[SolverConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, object]]:
    phys = PhysicalConfig() if phys is None else phys
    solver_cfg = SolverConfig() if solver_cfg is None else solver_cfg

    if solver_cfg.use_fast_solver_first:
        s_fast, y_fast, ok_fast, info_fast = solve_scr_exact_fast(
            Us=Us,
            Ub=Ub,
            p_exp=p_exp,
            Dx=Dx,
            ht=ht,
            phys=phys,
            solver_cfg=solver_cfg,
        )
        if ok_fast:
            return s_fast, y_fast, ok_fast, info_fast
        if solver_cfg.verbose:
            print("Fast solver failed, falling back to staged robust solver.")
            print(f"Reason: {info_fast.get('message', 'unknown error')}")

    x_bottom = phys.x_bottom
    z_lower = phys.z_lower_equilibrium
    x_top = phys.x_bottom + Dx
    z_top = -ht

    try:
        s_current, y_current = generate_catenary_initial_guess(
            phys=phys, Dx=Dx, ht=ht, n_init=solver_cfg.n_init
        )
    except Exception as exc:
        return None, None, False, {
            "stage": "initial_guess",
            "message": f"Initial guess generation failed: {exc}",
        }

    def bc_wrap(ya, yb):
        return boundary_conditions(
            ya, yb,
            x_bottom=x_bottom,
            z_lower=z_lower,
            x_top=x_top,
            z_top=z_top,
        )

    def make_ode(drag_factor: float, contact_factor: float):
        return lambda s, y: ode_system(
            s=s, y=y,
            phys=phys,
            Us=Us, Ub=Ub, p_exp=p_exp,
            drag_factor=drag_factor,
            contact_factor=contact_factor,
        )

    if solver_cfg.verbose:
        print("=" * 80)
        print("SCR exact solver (inextensible)")
        print(f"Target geometry: Dx={Dx:.3f} m, ht={ht:.3f} m")
        print(f"Lower-end z    : z_lower={z_lower:.6f} m (spring-settlement equilibrium)")
        print(f"Current params : Us={Us:.3f}, Ub={Ub:.3f}, p={p_exp:.4f}")
        print("=" * 80)
        print("Stage 0: gravity only")

    sol = solve_bvp(
        make_ode(drag_factor=0.0, contact_factor=0.0),
        bc_wrap,
        s_current,
        y_current,
        tol=solver_cfg.tol_stage0,
        max_nodes=solver_cfg.max_nodes,
        verbose=0,
    )
    if not sol.success:
        return None, None, False, {
            "stage": "stage0",
            "message": f"Stage 0 failed: {sol.message}",
        }

    s_current, y_current = regrid_solution(sol, phys.L, solver_cfg.continuation_regrid_nodes)

    if solver_cfg.verbose:
        print("Stage 1: gravity + drag ramp-up, contact off")
    for lf in solver_cfg.drag_factors_stage1:
        sol = solve_bvp(
            make_ode(drag_factor=float(lf), contact_factor=0.0),
            bc_wrap,
            s_current,
            y_current,
            tol=solver_cfg.tol_stage1,
            max_nodes=solver_cfg.max_nodes,
            verbose=0,
        )
        if not sol.success:
            return None, None, False, {
                "stage": "stage1",
                "message": f"Stage 1 failed at drag factor {lf}: {sol.message}",
            }
        s_current, y_current = regrid_solution(sol, phys.L, solver_cfg.continuation_regrid_nodes)
        if solver_cfg.verbose:
            print(f"  drag factor = {lf:.2f}  --> success")

    if solver_cfg.verbose:
        print("Stage 2: gravity + full drag + contact ramp-up")
    for cf in solver_cfg.contact_factors_stage2:
        sol = solve_bvp(
            make_ode(drag_factor=1.0, contact_factor=float(cf)),
            bc_wrap,
            s_current,
            y_current,
            tol=solver_cfg.tol_stage2,
            max_nodes=solver_cfg.max_nodes,
            verbose=0,
        )
        if not sol.success:
            return None, None, False, {
                "stage": "stage2",
                "message": f"Stage 2 failed at contact factor {cf}: {sol.message}",
            }
        s_current, y_current = regrid_solution(sol, phys.L, solver_cfg.continuation_regrid_nodes)
        if solver_cfg.verbose:
            print(f"  contact factor = {cf:.2f}  --> success")

    s_eval = np.linspace(0.0, phys.L, solver_cfg.n_eval)
    y_eval = sol.sol(s_eval)

    info = {
        "stage": "done",
        "message": "Success",
        "solver_message": sol.message,
        "phys": asdict(phys),
        "solver_cfg": asdict(solver_cfg),
        "target_geometry": {
            "Dx": Dx,
            "ht": ht,
            "x_top": x_top,
            "z_top": z_top,
        },
        "current": {
            "Us": Us,
            "Ub": Ub,
            "p_exp": p_exp,
        },
    }
    return s_eval, y_eval, True, info


# =============================================================================
# 7. Post-processing
# =============================================================================

def compute_local_resultants_from_global(theta: np.ndarray, H: np.ndarray, V: np.ndarray):
    """
    Convert global force components into local resultants:
        N = H cos(theta) + V sin(theta)
        Q = H sin(theta) - V cos(theta)
    """
    N = H * np.cos(theta) + V * np.sin(theta)
    Q = H * np.sin(theta) - V * np.cos(theta)
    return N, Q


def save_solution_npz(save_path: str | Path, s: np.ndarray, y: np.ndarray, info: Dict[str, object]) -> None:
    x, z, theta, H, V, M = y
    N, Q = compute_local_resultants_from_global(theta, H, V)

    np.savez_compressed(
        save_path,
        s=s,
        x=x,
        z=z,
        theta=theta,
        H=H,
        V=V,
        M=M,
        N=N,
        Q=Q,
        info_json=json.dumps(info, ensure_ascii=False, indent=2),
    )


def plot_solution(s: np.ndarray, y: np.ndarray, phys: PhysicalConfig, save_path: Optional[str | Path] = None) -> None:
    if plt is None:
        raise ImportError("matplotlib is required for plot_solution, but it is not available in this environment.")

    x, z, theta, H, V, M = y
    N, Q = compute_local_resultants_from_global(theta, H, V)

    fig = plt.figure(figsize=(11, 9))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(x, z, lw=2)
    ax1.axhline(0.0, ls="--")
    ax1.axhline(phys.z_bed, ls="--")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_title("SCR geometry")
    ax1.grid(True, ls=":")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(s, theta, lw=2)
    ax2.set_xlabel("s (m)")
    ax2.set_ylabel(r"$\theta$ (rad)")
    ax2.set_title("Tangent angle")
    ax2.grid(True, ls=":")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(s, N / 1e3, lw=2, label="N")
    ax3.plot(s, Q / 1e3, lw=2, label="Q")
    ax3.set_xlabel("s (m)")
    ax3.set_ylabel("Force (kN)")
    ax3.set_title("Local resultants")
    ax3.grid(True, ls=":")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(s, M / 1e3, lw=2)
    ax4.set_xlabel("s (m)")
    ax4.set_ylabel("Moment (kN·m)")
    ax4.set_title("Bending moment")
    ax4.grid(True, ls=":")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=220)
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# 8. Example main
# =============================================================================

def main() -> None:
    phys = PhysicalConfig()
    solver_cfg = SolverConfig()

    # Example case
    Us = 1.5
    Ub = 0.3
    p_exp = 0.25
    Dx = 1800.0
    ht = 20.0

    s, y, success, info = solve_scr_exact(
        Us=Us, Ub=Ub, p_exp=p_exp,
        Dx=Dx, ht=ht,
        phys=phys,
        solver_cfg=solver_cfg,
    )

    if not success:
        print("Solver failed:")
        print(info)
        return

    out_dir = Path("scr_exact_solver_outputs_inextensible")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_solution_npz(out_dir / "scr_exact_solution.npz", s, y, info)
    plot_solution(s, y, phys, save_path=out_dir / "scr_exact_solution.png")

    print("=" * 80)
    print("Numerical SCR solution finished successfully.")
    print(f"Outputs saved in: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
