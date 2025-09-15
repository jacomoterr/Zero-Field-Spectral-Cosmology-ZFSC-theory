import numpy as np

def robust_plateaus(eigs: np.ndarray, min_width: int, tau: float = 0.35):
    if eigs.size < 2:
        return []
    diffs = np.diff(eigs)
    med = np.median(diffs) if (diffs.size > 0 and np.all(np.isfinite(diffs))) else 0.0
    thr = tau * med if med > 0 else np.inf
    plateaus = []
    cluster = [eigs[0]]
    for j in range(1, eigs.size):
        if abs(eigs[j] - eigs[j - 1]) <= thr:
            cluster.append(eigs[j])
        else:
            if len(cluster) >= min_width:
                plateaus.append(cluster)
            cluster = [eigs[j]]
    if len(cluster) >= min_width:
        plateaus.append(cluster)
    return plateaus
