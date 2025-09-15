# shells.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.6
# Кластеризация спектра в оболочки (3–4–7)

import numpy as np

def segment_shells(eigs, cfg):
    """
    Разбивает спектр на оболочки по крупным разрывам.
    Возвращает список кластеров и метрики:
      - shell_count
      - shell_sizes
      - shell_gap_min/max
      - shell_purity
    """
    if eigs.size < 2:
        return [], {
            "shell_count": 0,
            "shell_sizes": [],
            "shell_gap_min": None,
            "shell_gap_max": None,
            "shell_purity": None
        }

    sh_cfg = cfg.get("shells", {})
    method = sh_cfg.get("method", "mult_median")
    shell_min_size = int(sh_cfg.get("shell_min_size", 2))
    mult = float(sh_cfg.get("mult", 2.5))
    percentile = float(sh_cfg.get("percentile", 90.0))

    diffs = np.diff(eigs)
    med = np.median(diffs) if diffs.size > 0 else 0.0
    thr = med * mult if method == "mult_median" else np.percentile(diffs, percentile)

    clusters, cur = [], [eigs[0]]
    for j in range(1, eigs.size):
        if diffs[j-1] > thr and len(cur) >= shell_min_size:
            clusters.append(cur)
            cur = [eigs[j]]
        else:
            cur.append(eigs[j])
    if len(cur) >= shell_min_size:
        clusters.append(cur)

    gaps = [clusters[i][0] - clusters[i-1][-1] for i in range(1, len(clusters))]
    intra = [np.mean(np.diff(cl)) if len(cl) > 1 else 0 for cl in clusters]

    purity = min(gaps)/max(intra) if gaps and intra and max(intra) > 0 else None

    return clusters, {
        "shell_count": len(clusters),
        "shell_sizes": [len(cl) for cl in clusters],
        "shell_gap_min": float(np.min(gaps)) if gaps else None,
        "shell_gap_max": float(np.max(gaps)) if gaps else None,
        "shell_purity": float(purity) if purity is not None else None
    }
