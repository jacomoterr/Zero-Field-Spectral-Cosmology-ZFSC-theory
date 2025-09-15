# stats.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.5
# Агрегация и сохранение статистики (расширенный вариант)

import os
import csv
import json
import numpy as np
from collections import defaultdict


def make_summary_stats(summary_rows):
    """
    Собирает агрегированную статистику по каждой группе (N, edge).
    Метрики:
      - num_plateaus (avg/min/max)
      - gap_ratio (avg/min/max, p90/p95/p99)
      - частоты резонансов (>=10, >=100, >=1000)
      - plateau_width (avg, max)
    """
    groups = defaultdict(list)
    for row in summary_rows:
        key = (row["N"], row["edge"])
        groups[key].append(row)

    stats = []
    for (N, edge), rows in groups.items():
        num_plateaus = [r["num_plateaus"] for r in rows if r["num_plateaus"] is not None]
        gap_ratio = [r["gap_ratio"] for r in rows if r["gap_ratio"] is not None]

        plateau_widths = [r["plateau_width"] for r in rows if r["plateau_width"] is not None]
        plateau_width_maxes = [r["plateau_width_max"] for r in rows if r["plateau_width_max"] is not None]

        res10 = sum(1 for g in gap_ratio if g >= 10)
        res100 = sum(1 for g in gap_ratio if g >= 100)
        res1000 = sum(1 for g in gap_ratio if g >= 1000)

        stat = {
            "N": N,
            "edge": edge,
            "count": len(rows),
            # базовые метрики
            "num_plateaus_avg": float(np.mean(num_plateaus)) if num_plateaus else None,
            "num_plateaus_min": int(np.min(num_plateaus)) if num_plateaus else None,
            "num_plateaus_max": int(np.max(num_plateaus)) if num_plateaus else None,
            "gap_ratio_avg": float(np.mean(gap_ratio)) if gap_ratio else None,
            "gap_ratio_min": float(np.min(gap_ratio)) if gap_ratio else None,
            "gap_ratio_max": float(np.max(gap_ratio)) if gap_ratio else None,
            # хвосты распределений
            "gap_ratio_p90": float(np.percentile(gap_ratio, 90)) if len(gap_ratio) >= 10 else None,
            "gap_ratio_p95": float(np.percentile(gap_ratio, 95)) if len(gap_ratio) >= 20 else None,
            "gap_ratio_p99": float(np.percentile(gap_ratio, 99)) if len(gap_ratio) >= 50 else None,
            # частоты резонансов
            "res10_frac": res10 / len(gap_ratio) if gap_ratio else None,
            "res100_frac": res100 / len(gap_ratio) if gap_ratio else None,
            "res1000_frac": res1000 / len(gap_ratio) if gap_ratio else None,
            # ширины плато
            "plateau_width_avg": float(np.mean(plateau_widths)) if plateau_widths else None,
            "plateau_width_max": float(np.max(plateau_width_maxes)) if plateau_width_maxes else None,
        }
        stats.append(stat)

    return stats


def write_summary_stats(stats, root):
    """Записывает агрегированную статистику в JSON и CSV"""
    json_path = os.path.join(root, "summary_stats.json")
    csv_path = os.path.join(root, "summary_stats.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if stats:
        keys = list(stats[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(stats)

    return json_path, csv_path
