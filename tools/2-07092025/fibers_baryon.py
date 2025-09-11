import numpy as np

# Глобальные настройки
ENABLE_BARYON = True

BARYON_CLUSTER_SIZE = 3   # три узла = барион
EPS_BARYON = 0.1          # сила связи внутри кластера (сильная!)

# Маска: True = кварк (барион), False = лептон (одиночный узел)
# Допустим, первые 1/4 узлов = лептоны, остальные = кварки
LEPTON_FRACTION = 0.25

def baryon_fiber_expand(M: np.ndarray) -> np.ndarray:
    """
    Расширяет матрицу:
    - узлы-лептоны остаются одиночными,
    - узлы-кварки становятся барионными кластерами (3 узла).
    """
    if not ENABLE_BARYON:
        return M

    N = M.shape[0]
    n_leptons = int(N * LEPTON_FRACTION)

    # считаем новую размерность
    newN = n_leptons + (N - n_leptons) * BARYON_CLUSTER_SIZE
    H = np.zeros((newN, newN), dtype=float)

    # отображение индексов: старый → список новых индексов
    index_map = {}

    cur = 0
    for i in range(N):
        if i < n_leptons:
            # лептон = одиночный узел
            index_map[i] = [cur]
            cur += 1
        else:
            # кварк = барионный кластер
            indices = list(range(cur, cur + BARYON_CLUSTER_SIZE))
            index_map[i] = indices
            cur += BARYON_CLUSTER_SIZE
            # внутри кластера — полный граф
            for a in indices:
                for b in indices:
                    if a != b:
                        H[a, b] = EPS_BARYON

    # перенос связей
    for i in range(N):
        for j in range(i+1, N):
            if M[i, j] != 0.0:
                for a in index_map[i]:
                    for b in index_map[j]:
                        H[a, b] = H[b, a] = M[i, j]

    return H
