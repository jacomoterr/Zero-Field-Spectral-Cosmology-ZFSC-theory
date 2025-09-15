# snail.py
# Спиральная анизотропия ("улитка") для ZFSC v3.1.4

import numpy as np

def apply_snail(H: np.ndarray, params: dict) -> np.ndarray:
    """
    Диагональный спиральный потенциал:
      V_snail(i) = ρ · (r^p) · cos(m_s · θ(i) + φ_s)
    где r = i/(N-1), θ = 2π i/N.  ρ>=0, p>=1 управляют ростом к краю.
    """
    sn = params.get("snail", None)
    if not sn or float(sn.get("rho", 0.0)) == 0.0:
        return H

    rho = float(sn["rho"])
    m_s = int(sn.get("m", 1))
    phi = float(sn.get("phi", 0.0))
    p   = float(sn.get("p", 1.0))

    N = H.shape[0]
    i = np.arange(N, dtype=np.float64)
    r = i / max(N - 1, 1.0)
    theta = 2.0 * np.pi * i / max(N, 1)

    V_snail = rho * (r ** p) * np.cos(m_s * theta + phi)
    return H + np.diag(V_snail)
