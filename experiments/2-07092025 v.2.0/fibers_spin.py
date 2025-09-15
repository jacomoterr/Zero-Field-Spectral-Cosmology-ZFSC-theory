import numpy as np
ENABLE_SPIN = True
EPS_SPIN = 0.01

def spin_fiber_expand(M: np.ndarray) -> np.ndarray:
    if not ENABLE_SPIN:
        return M
    N = M.shape[0]
    H = np.kron(M, np.eye(2))
    for i in range(N):
        a, b = 2*i, 2*i+1
        H[a, b] = H[b, a] = H[a, b] + EPS_SPIN
    return H
