# reservoir.py
# Вакуумный резервуар для ZFSC v3.1.3
# Добавляет подпитку каналов gr/em/wk/st в H_eff

import numpy as np

def apply_reservoir(H: np.ndarray, rng: np.random.Generator, params: dict) -> np.ndarray:
    """
    Добавляем вклад вакуумного резервуара:
      H' = H + sum_k κ_k Π_k
    где Π_k — диагональные «канальные» профили (гравитация, электромагнетизм,
    слабое и сильное взаимодействия).
    """
    res_cfg = params.get("reservoir", {})
    kappas = res_cfg.get("kappas", [0.0, 0.0, 0.0, 0.0])  # [κ_gr, κ_em, κ_wk, κ_st]

    if not any(abs(k) > 0 for k in kappas):
        return H

    N = H.shape[0]
    diag_add = np.zeros(N, dtype=np.float64)

    # Каналы: простые разные профили
    i = np.arange(N, dtype=np.float64) / max(N - 1, 1)

    # 1. Гравитация → равномерный фон
    diag_add += kappas[0] * np.ones_like(i)

    # 2. Электромагнетизм → линейный рост
    diag_add += kappas[1] * i

    # 3. Слабое взаимодействие → колоколообразный профиль
    diag_add += kappas[2] * np.exp(-((i - 0.5) ** 2) / 0.02)

    # 4. Сильное взаимодействие → стоячая волна (синус)
    diag_add += kappas[3] * np.sin(2.0 * np.pi * i)

    return H + np.diag(diag_add)
