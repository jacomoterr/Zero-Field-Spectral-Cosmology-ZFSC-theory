# stability.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.6
# Фильтр устойчивости плато и оценка persistence

import numpy as np

def evaluate_stability(eigs, plateaus, H=None, rng=None, cfg=None):
    """
    Возвращает список устойчивых плато и метрики:
      - stable_count
      - stable_width_avg/max
      - plateau_persistence (если n_probe>0)
    """
    if not plateaus:
        return [], {
            "stable_count": 0,
            "stable_width_avg": None,
            "stable_width_max": None,
            "plateau_persistence": None
        }

    st_cfg = cfg.get("stability", {})
    q_min = float(st_cfg.get("q_min", 3.0))
    n_probe = int(st_cfg.get("n_probe", 0))
    sigma = float(st_cfg.get("sigma", 1e-5))

    stable, widths = [], []

    for cl in plateaus:
        if len(cl) < 2:
            continue
        gaps_intra = np.diff(cl)
        gap_intra = np.mean(gaps_intra) if gaps_intra.size > 0 else 0.0

        left_gap = cl[0] - max([e for e in eigs if e < cl[0]], default=cl[0])
        right_gap = min([e for e in eigs if e > cl[-1]], default=cl[-1]) - cl[-1]

        q = min(left_gap, right_gap) / max(gap_intra, 1e-12)
        if q >= q_min:
            stable.append(cl)
            widths.append(len(cl))

    persistence = None
    if n_probe > 0 and H is not None and rng is not None and stable:
        means0 = [np.mean(cl) for cl in stable]
        survive_counts = []
        for _ in range(n_probe):
            noise = sigma * rng.standard_normal(H.shape)
            noise = (noise + noise.T) * 0.5
            try:
                ev = np.linalg.eigvalsh(H + noise)
                ev.sort()
                means_probe = ev  # приближенно сравниваем центры
                survived = sum(1 for m in means0
                               if any(abs(m - mp) < 1e-3 for mp in means_probe))
                survive_counts.append(survived / len(stable))
            except np.linalg.LinAlgError:
                continue
        if survive_counts:
            persistence = float(np.mean(survive_counts))

    return stable, {
        "stable_count": len(stable),
        "stable_width_avg": float(np.mean(widths)) if widths else None,
        "stable_width_max": int(np.max(widths)) if widths else None,
        "plateau_persistence": persistence
    }
