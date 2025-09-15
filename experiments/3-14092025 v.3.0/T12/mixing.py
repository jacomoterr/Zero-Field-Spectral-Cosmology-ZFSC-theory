# mixing.py
# Узел (локальные смешивания) для ZFSC v3.1.2

import numpy as np

def random_unitary(g: int, rng: np.random.Generator) -> np.ndarray:
    """Генерируем случайную унитарную матрицу размера g×g (Haar)."""
    Z = rng.standard_normal((g, g)) + 1j * rng.standard_normal((g, g))
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph

def apply_mixing(H: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """
    Вносим поправку узла:
      ΔH = (W ⊗ I) H (W† ⊗ I) - H
      H' = H + η ΔH
    """
    eta = params["knot"]["eta"]
    g = params["knot"]["g"]

    if g <= 1 or eta == 0.0:
        return H

    N = H.shape[0]
    if N % g != 0:
        return H  # только если N делится на g

    # строим блочную унитарную матрицу
    Wg = random_unitary(g, rng)
    W = np.kron(Wg, np.eye(N // g))

    H_new = W @ H @ W.conj().T
    delta_H = H_new - H
    return H + eta * delta_H.real
