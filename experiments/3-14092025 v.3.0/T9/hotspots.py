# hotspots.py
# Детектор и обработка hotspot'ов для adaptive runner

import json
import os


def detect_hotspots(summary_rows, cfg):
    """Выбирает точки, которые проходят по критериям hotspots"""
    hs_cfg = cfg.get("hotspots", {})
    gap_thr = hs_cfg.get("gap_ratio_min", 100.0)
    purity_thr = hs_cfg.get("shell_purity_min", 2.0)
    persist_thr = hs_cfg.get("plateau_persistence_min", 0.35)

    hotspots = []
    for r in summary_rows:
        if (
            (r.get("gap_ratio") is not None and r["gap_ratio"] >= gap_thr) or
            (r.get("shell_purity") is not None and r["shell_purity"] >= purity_thr) or
            (r.get("plateau_persistence") is not None and r["plateau_persistence"] >= persist_thr)
        ):
            hotspots.append({
                "N": r["N"],
                "seed": r["seed"],
                "point": r["point"],
                "edge": r["edge"],
                "ablation_id": r.get("ablation_id", 0),
                "gap_ratio": r.get("gap_ratio"),
                "stable_count": r.get("stable_count"),
                "shell_purity": r.get("shell_purity"),
                "plateau_persistence": r.get("plateau_persistence"),
            })
    return hotspots


def save_hotspots(hotspots, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hotspots, f, ensure_ascii=False, indent=2)


def load_hotspots(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
