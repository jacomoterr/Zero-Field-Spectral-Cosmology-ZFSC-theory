# geometry.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.4
# Базовая геометрия + сэмплинг гиперпараметров (umbrella, lotus, cycles, knot, dual, snail)

import numpy as np
from typing import Dict, Any


# -------------------------- ЛАПЛАСИАН -------------------------- #

def laplacian_1d(N: int, edge_kind: str) -> np.ndarray:
    """
    1D лапласиан (симметричный, вещественный):
      - 'hard' => периодические ГУ (кольцо)
      - прочее => открытые ГУ (цепочка)
    """
    L = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        L[i, i] = 2.0
        if i - 1 >= 0:
            L[i, i - 1] = -1.0
        if i + 1 < N:
            L[i, i + 1] = -1.0
    if str(edge_kind).lower() == "hard":  # periodic
        L[0, N - 1] = -1.0
        L[N - 1, 0] = -1.0
    return L


# ---------------------- СЭМПЛИНГ ПАРАМЕТРОВ --------------------- #

def _uniform_from(rng: np.random.Generator, val):
    """Возвращает число из интервала [lo, hi] или float(val), если не список/кортеж."""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        lo, hi = float(val[0]), float(val[1])
        return rng.uniform(lo, hi)
    return float(val)


def sample_hyperparams(rng: np.random.Generator, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Сэмплинг гиперпараметров из словаря cfg.
    ВСЕГДА возвращает ключи: umbrella, lotus, cycles, knot, dual, snail.
    """
    # --- umbrella ---
    umb = cfg.get("umbrella", {})
    alpha = _uniform_from(rng, umb.get("alpha", [0.1, 1.0]))
    kappa = _uniform_from(rng, umb.get("kappa", [0.1, 1.0]))

    # steps
    steps_cfg = umb.get("steps", {})
    N_steps_choices = steps_cfg.get("N_steps", [0])
    if not isinstance(N_steps_choices, (list, tuple)):
        N_steps_choices = [int(N_steps_choices)]
    n_steps = int(rng.choice(N_steps_choices)) if len(N_steps_choices) > 0 else 0

    r_list_full = steps_cfg.get("r_over_R", [])
    if not isinstance(r_list_full, (list, tuple)):
        r_list_full = []
    r_list_full = [float(r) for r in r_list_full]

    if n_steps > 0 and len(r_list_full) > 0:
        pick = min(n_steps, len(r_list_full))
        r_steps = sorted(rng.choice(r_list_full, size=pick, replace=False))
    else:
        r_steps = []

    h_lo, h_hi = steps_cfg.get("h_range", [0.1, 1.0])
    w_lo, w_hi = steps_cfg.get("w_range", [0.01, 0.1])
    steps = [{"r_over_R": float(r),
              "h": rng.uniform(float(h_lo), float(h_hi)),
              "w": rng.uniform(float(w_lo), float(w_hi))} for r in r_steps]

    # anisotropy
    an = umb.get("anisotropy", {})
    m_choices = an.get("m_choices", [1, 2, 3, 4])
    if not isinstance(m_choices, (list, tuple)):
        m_choices = [int(m_choices)]
    m = int(rng.choice(m_choices)) if len(m_choices) > 0 else 1

    eps = _uniform_from(rng, an.get("epsilon", [0.0, 0.2]))
    alpha_theta = float(an.get("alpha_theta", 1.0))

    # edge
    edge_choices = umb.get("edge", ["hard"])
    if isinstance(edge_choices, (list, tuple)) and len(edge_choices) > 0:
        edge = str(rng.choice(edge_choices))
    else:
        edge = str(edge_choices)

    # --- lotus_fix -> lotus (ключ 'lotus' для раннера) ---
    lotus_src = cfg.get("lotus_fix", {})
    gamma = _uniform_from(rng, lotus_src.get("gamma", [0.0, 0.5]))
    beta  = _uniform_from(rng, lotus_src.get("beta",  [0.0, 0.5]))

    # --- cycles ---
    cyc = cfg.get("cycles", {})
    delta = _uniform_from(rng, cyc.get("delta", [0.0, 0.2]))
    P_choices = cyc.get("P_choices", [8, 16, 32, 64])
    if not isinstance(P_choices, (list, tuple)) or len(P_choices) == 0:
        P_choices = [8]
    P = int(rng.choice(P_choices))

    # --- knot (mixing) ---
    knot = cfg.get("knot", {})
    eta = _uniform_from(rng, knot.get("eta", [0.0, 0.3]))
    g_choices = knot.get("locality_g", [1, 2, 3])
    if not isinstance(g_choices, (list, tuple)) or len(g_choices) == 0:
        g_choices = [1]
    g_local = int(rng.choice(g_choices))

    # --- dual (рыбы) ---
    dual_cfg = cfg.get("dual", {})
    dual_enabled = bool(dual_cfg.get("enabled", True))
    kappa_lr = _uniform_from(rng, dual_cfg.get("kappa_lr", [0.0, 0.0]))
    epsilon_asym = _uniform_from(rng, dual_cfg.get("epsilon_asym", [0.0, 0.0]))
    phase = _uniform_from(rng, dual_cfg.get("phase", [0.0, 0.0]))

    # --- snail (улитка) ---
    snail_cfg = cfg.get("snail", {})
    rho = _uniform_from(rng, snail_cfg.get("rho", [0.0, 0.0]))
    sn_m_choices = snail_cfg.get("m_choices", [1, 2, 3, 4, 6])
    if not isinstance(sn_m_choices, (list, tuple)) or len(sn_m_choices) == 0:
        sn_m_choices = [1]
    m_s = int(rng.choice(sn_m_choices))
    phi_s = _uniform_from(rng, snail_cfg.get("phi", [0.0, 0.0]))
    p_s = _uniform_from(rng, snail_cfg.get("p", [1.0, 1.0]))

    return {
        "umbrella": {
            "alpha": float(alpha),
            "kappa": float(kappa),
            "steps": steps,
            "anisotropy": {"m": int(m), "epsilon": float(eps), "alpha_theta": float(alpha_theta)},
            "edge": edge,
        },
        # ключ 'lotus' — так ожидает runner.py
        "lotus": {"gamma": float(gamma), "beta": float(beta)},
        "cycles": {"delta": float(delta), "P": int(P)},
        "knot": {"eta": float(eta), "g": int(g_local)},
        "dual": {
            "enabled": bool(dual_enabled),
            "kappa_lr": float(kappa_lr),
            "epsilon_asym": float(epsilon_asym),
            "phase": float(phase),
        },
        # ГАРАНТИРОВАНО присутствует:
        "snail": {
            "rho": float(rho),
            "m": int(m_s),
            "phi": float(phi_s),
            "p": float(p_s),
        },
    }


# --------------------- ПОТЕНЦИАЛ ЗОНТИКА ---------------------- #

def umbrella_diagonal(N: int, params: Dict[str, Any]) -> np.ndarray:
    """
    Диагональный потенциал V(r,θ) для зонтика:
    V = α·r + κ·r^2 + sum(step) + α_θ·ε·cos(m·θ)
    где r = i/(N-1), θ = 2π i / N
    """
    i = np.arange(N, dtype=np.float64)
    r = i / (N - 1 if N > 1 else 1.0)
    theta = 2.0 * np.pi * i / max(N, 1)

    umb = params["umbrella"]
    V = umb["alpha"] * r + umb["kappa"] * r**2

    for st in umb["steps"]:
        rj, h, w = float(st["r_over_R"]), float(st["h"]), float(st["w"])
        V += h * 0.5 * (1.0 + np.tanh((r - rj) / max(w, 1e-6)))

    m = int(umb["anisotropy"]["m"])
    eps = float(umb["anisotropy"]["epsilon"])
    alpha_theta = float(umb["anisotropy"]["alpha_theta"])
    V += alpha_theta * eps * np.cos(m * theta)
    return V


# ------------------------ СБОРКА H_eff ------------------------ #

def build_H_eff(N: int, params: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """
    Минимальный H_eff: (1+γ)·L + diag(V) + симметричный слабый шум.
    """
    edge = params["umbrella"]["edge"]
    L = laplacian_1d(N, edge_kind=edge)
    V_diag = umbrella_diagonal(N, params)
    gamma = float(params["lotus"]["gamma"])

    H = (1.0 + gamma) * L + np.diag(V_diag)

    # симметричный малый шум для разлипание вырожденностей
    noise = 1e-6 * rng.standard_normal((N, N))
    H = H + (noise + noise.T) * 0.5
    return H
