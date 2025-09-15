# viz_surface.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.6
# Визуализация предгеометрического потенциала V(r,θ) в виде 2D-карты

import os
import numpy as np
import matplotlib.pyplot as plt


def V_surface(Nr=200, Nθ=200,
              alpha=0.5, kappa=0.3,
              steps=None,
              gamma=0.2, beta=0.1,
              rho=0.05, m=3, phi=0.0, p=1.0):
    if steps is None:
        steps = [{"r_over_R": 0.5, "h": 0.5, "w": 0.05}]

    r = np.linspace(0, 1.0, Nr)
    θ = np.linspace(0, 2*np.pi, Nθ)
    R, Θ = np.meshgrid(r, θ, indexing="ij")

    V = alpha * R + kappa * R**2

    for st in steps:
        rj, h, w = st["r_over_R"], st["h"], st["w"]
        V += h * 0.5 * (1.0 + np.tanh((R - rj)/max(w,1e-6)))

    if beta != 0.0:
        V += gamma * np.sin(2*np.pi*R/max(beta, 1e-6))

    V += rho * (R**p) * np.cos(m*Θ + phi)

    return R, Θ, V


def save_surface(params: dict, save_path: str):
    """Сохраняет 2D-карту потенциала V(r,θ)"""
    umb = params.get("umbrella", {})
    lotus = params.get("lotus", {})
    snail = params.get("snail", {})

    alpha = float(umb.get("alpha", 0.5))
    kappa = float(umb.get("kappa", 0.3))
    steps = umb.get("steps", [])

    gamma = float(lotus.get("gamma", 0.2))
    beta  = float(lotus.get("beta", 0.1))

    rho = float(snail.get("rho", 0.05))
    m   = int(snail.get("m", 3))
    phi = float(snail.get("phi", 0.0))
    p   = float(snail.get("p", 1.0))

    R, Θ, V = V_surface(alpha=alpha, kappa=kappa, steps=steps,
                        gamma=gamma, beta=beta,
                        rho=rho, m=m, phi=phi, p=p)

    fig, ax = plt.subplots(figsize=(8,6))
    c = ax.imshow(V, origin="lower", aspect="auto",
                  extent=[0, 2*np.pi, 0, 1],
                  cmap="viridis")
    fig.colorbar(c, ax=ax, label="V(r,θ)")
    ax.set_xlabel("θ (угол)")
    ax.set_ylabel("r (радиус)")
    ax.set_title("2D-карта предгеометрического потенциала V(r,θ)")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ 2D-визуализация сохранена: {save_path}")
