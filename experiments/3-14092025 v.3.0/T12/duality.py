# duality.py
# Двойное поле (φ ⊕ \bar{φ}) для ZFSC v3.1.4
# Строит расширенный гамильтониан:
#   H_ext = [[ H_R,    Δ ],
#            [ Δ† , H_L+ε ]]
# где H_L — зеркальная (левохиральная) копия H_R, Δ — связь между ветвями.

import numpy as np

def _flip_operator(N: int) -> np.ndarray:
    """J — оператор зеркального отражения индексов (антидиагональная единичная)."""
    J = np.eye(N, dtype=np.float64)[::-1]
    return J

def _random_unitary(N: int, rng: np.random.Generator) -> np.ndarray:
    """Случайная унитарная (Haar) матрица размера N×N."""
    Z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph

def apply_duality(H: np.ndarray, rng: np.random.Generator, params: dict):
    """
    Возвращает (H_ext, doubled_flag).
    Если dual.enabled=False или kappa_lr=0 → возвращает исходный H и False.
    """
    dual = params.get("dual", {})
    enabled = bool(dual.get("enabled", True))
    if not enabled:
        return H, False

    kappa_lr = float(dual.get("kappa_lr", 0.0))       # сила связи L↔R
    epsilon_asym = float(dual.get("epsilon_asym", 0.0))# малая асимметрия между ветвями
    phi0 = float(dual.get("phase", 0.0))               # глобальная фаза связи

    if kappa_lr == 0.0:
        return H, False

    N = H.shape[0]
    J = _flip_operator(N)
    # Зеркальная левая ветвь как H_L = J H J^T (инверсия индексов/«поворот»)
    H_L = J @ H @ J.T

    # Унитарная структура связи Δ (со случайными фазами вокруг глобальной фазы phi0)
    U = _random_unitary(N, rng)
    phases = np.exp(1j * (phi0 + 2.0 * np.pi * rng.random(N)))
    Delta = kappa_lr * (U @ np.diag(phases) @ U.conj().T)

    # Сборка блочного гамильтониана (эрмитова матрица 2N×2N)
    H_ext = np.zeros((2 * N, 2 * N), dtype=np.complex128)
    H_ext[:N, :N] = H
    H_ext[N:, N:] = H_L + epsilon_asym * np.eye(N)
    H_ext[:N, N:] = Delta
    H_ext[N:, :N] = Delta.conj().T

    # Строго гермитизуем и возвращаем вещественную часть (наш спектр — вещественный)
    H_ext = (H_ext + H_ext.conj().T) * 0.5
    return H_ext.real, True
