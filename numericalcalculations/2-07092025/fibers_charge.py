import numpy as np
ENABLE_CHARGE = True
CHARGE_MODE = "odd"
CHARGE_MASK = None
Q_SHIFT = 0.005

def charge_shift_base(M: np.ndarray) -> np.ndarray:
    if not ENABLE_CHARGE:
        return M
    N = M.shape[0]
    H = M.copy()
    if CHARGE_MODE == "odd":
        charged = [(i % 2 == 1) for i in range(N)]
    elif CHARGE_MODE == "mask" and CHARGE_MASK is not None and len(CHARGE_MASK) == N:
        charged = [bool(x) for x in CHARGE_MASK]
    else:
        charged = [False] * N
    for i, is_charged in enumerate(charged):
        if is_charged:
            H[i,i] += Q_SHIFT
    return H
