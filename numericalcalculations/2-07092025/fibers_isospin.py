import numpy as np
ENABLE_ISOSPIN = True
EPS_ISOSPIN = 0.001

def isospin_fiber_base(M: np.ndarray) -> np.ndarray:
    if not ENABLE_ISOSPIN:
        return M
    N = M.shape[0]
    H = M.copy()
    for i in range(0, N-1, 2):
        H[i, i+1] = H[i+1, i] = H[i, i+1] + EPS_ISOSPIN
    return H
