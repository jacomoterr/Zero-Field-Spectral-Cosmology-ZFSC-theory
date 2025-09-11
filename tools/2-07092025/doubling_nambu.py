import numpy as np
ENABLE_NAMBU = True
DELTA_COUPLE = 0.001

def nambu_double(M: np.ndarray) -> np.ndarray:
    if not ENABLE_NAMBU:
        return M
    N = M.shape[0]
    H = np.zeros((2*N, 2*N), dtype=float)
    H[:N,:N] =  M
    H[N:,N:] = -M
    for i in range(N):
        H[i, i+N] = H[i+N, i] = H[i, i+N] + DELTA_COUPLE
    return H
