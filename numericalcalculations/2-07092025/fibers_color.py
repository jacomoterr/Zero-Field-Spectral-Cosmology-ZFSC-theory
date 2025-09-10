import numpy as np
ENABLE_COLOR = True
COLOR_INCLUDE_ANTICOLOR = True
EPS_COLOR = 0.001

def color_fiber_expand(M: np.ndarray) -> np.ndarray:
    if not ENABLE_COLOR:
        return M
    color_dim = 6 if COLOR_INCLUDE_ANTICOLOR else 3
    H = np.kron(M, np.eye(color_dim))
    oldN = M.shape[0]
    for base_idx in range(oldN):
        base_start = base_idx * color_dim
        for c1 in range(color_dim):
            for c2 in range(c1+1, color_dim):
                i = base_start + c1
                j = base_start + c2
                H[i,j] = H[j,i] = H[i,j] + EPS_COLOR
    return H
