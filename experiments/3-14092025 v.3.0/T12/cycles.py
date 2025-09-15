# cycles.py
# Циклическая геометрия для ZFSC v3.1.2

import numpy as np

def apply_cycles(H: np.ndarray, params: dict) -> np.ndarray:
    """
    Вносим циклический потенциал:
      V_cycles(i) = δ · cos(2π P i / N)
    """
    delta = params["cycles"]["delta"]
    P = params["cycles"]["P"]

    if delta == 0.0 or P <= 0:
        return H

    N = H.shape[0]
    i = np.arange(N, dtype=np.float64)
    V_cycles = delta * np.cos(2.0 * np.pi * P * i / N)

    return H + np.diag(V_cycles)
