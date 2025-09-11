import numpy as np
from fibers_spin import spin_fiber_expand
from fibers_color import color_fiber_expand
from fibers_isospin import isospin_fiber_base
from fibers_charge import charge_shift_base
from fibers_baryon import baryon_fiber_expand
from doubling_nambu import nambu_double

# ===============================
# Размеры блоков
# ===============================
N_NU   = 12
N_LEP  = 12
N_UQ   = 18
N_DQ   = 18
N_BOS  = 10
N_DARK = 3
N_GRAV = 3

# ===============================
# Флаги
# ===============================
ENABLE_SPIN    = True
ENABLE_COLOR   = True
ENABLE_ISOSPIN = True
ENABLE_CHARGE  = True
ENABLE_BARYON  = True
ENABLE_NAMBU   = True
DEBUG_GEOMETRY = True   # <<< включаем отладочный вывод

# ===============================
# Параметры
# ===============================
EPS_NESTED   = 0.05
BETA_RADIAL  = 0.02

LINK_GRAV_VISIBLE = 0.05
LINK_VISIBLE_DARK = 0.05

# ===============================
# Луковичный блок
# ===============================
def build_onion(size: int, delta: float, r: float, nested: bool = True, name="block") -> np.ndarray:
    if size <= 0:
        return np.zeros((0,0))
    N = size
    M = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        M[i, i+1] = M[i+1, i] = r
    center = 0.5 * (N - 1)
    if nested:
        for i in range(N):
            for j in range(i+1, N):
                if M[i, j] != 0.0:
                    depth = int(min(abs(i - center), abs(j - center)))
                    if depth > 0:
                        M[i, j] *= (EPS_NESTED ** depth)
                        M[j, i]  =  M[i, j]
    np.fill_diagonal(M, 0.0)
    M[N // 2, N // 2] = delta
    for i in range(N):
        M[i,i] += BETA_RADIAL * (i - center)**2

    if DEBUG_GEOMETRY:
        print(f"[DEBUG] build_onion {name}: size={N}, min={M.min():.4g}, max={M.max():.4g}, mean={M.mean():.4g}")
    return M

# ===============================
# Сборка Вселенной
# ===============================
def compose_matrix(size: int, delta: float, r: float):
    nu   = build_onion(N_NU,   delta, r, name="nu")
    lep  = build_onion(N_LEP,  delta, r, name="lep")
    uq   = build_onion(N_UQ,   delta, r, name="u")
    dq   = build_onion(N_DQ,   delta, r, name="d")
    bos  = build_onion(N_BOS,  delta, r, name="bos")
    dark = build_onion(N_DARK, delta, r, name="dark")
    grav = build_onion(N_GRAV, delta, r, name="grav")

    blocks = {"nu": nu, "lep": lep, "u": uq, "d": dq, "bos": bos, "dark": dark, "grav": grav}
    offsets = {}
    cur = 0
    for k,v in blocks.items():
        offsets[k] = (cur, cur+v.shape[0])
        cur += v.shape[0]

    N_total = cur
    M = np.zeros((N_total, N_total), dtype=float)
    for k,(i0,i1) in offsets.items():
        M[i0:i1, i0:i1] = blocks[k]

    # связи
    if grav.shape[0] > 0 and lep.shape[0] > 0:
        gi0,gi1 = offsets["grav"]
        li0,li1 = offsets["lep"]
        M[gi1-1, li0] = M[li0, gi1-1] = LINK_GRAV_VISIBLE
    if dark.shape[0] > 0 and dq.shape[0] > 0:
        di0,di1 = offsets["dark"]
        dqi0,dqi1 = offsets["d"]
        M[di0, dqi1-1] = M[dqi1-1, di0] = LINK_VISIBLE_DARK

    # фибры
    if ENABLE_CHARGE: M = charge_shift_base(M)
    if ENABLE_ISOSPIN: M = isospin_fiber_base(M)
    if ENABLE_SPIN: M = spin_fiber_expand(M)
    if ENABLE_COLOR: M = color_fiber_expand(M)
    if ENABLE_BARYON: M = baryon_fiber_expand(M)
    if ENABLE_NAMBU: M = nambu_double(M)

    if DEBUG_GEOMETRY:
        print(f"[DEBUG] compose_matrix: total size={M.shape[0]}, dtype={M.dtype}, min={M.min():.4g}, max={M.max():.4g}, mean={M.mean():.4g}")

    return M, offsets


