# normalization.py
# Нормировочный сдвиг для ZFSC v3.1.3
# Добавляет -μI ко всему гамильтониану

import numpy as np

def apply_normalization(H: np.ndarray, params: dict) -> np.ndarray:
    """
    Вносим нормировочный сдвиг:
      H' = H - μ I
    """
    mu = float(params.get("normalization", {}).get("mu", 0.0))
    if mu == 0.0:
        return H

    N = H.shape[0]
    return H - mu * np.eye(N)
