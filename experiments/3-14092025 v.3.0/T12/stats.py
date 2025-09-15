# stats.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.6
# Агрегация и сохранение статистики (расширенный вариант, с абляциями)

import os
import csv
import json
import numpy as np
from collections import defaultdict


def _clean(values):
    """Фильтрует None и оставляет только числа"""
    return [v for v in values if v is not None and np.isfinite(v)]


def make_summary_stats(summary_rows):
    """
    Собирает агрегированную статистику по каждой группе (N, edge, ablation_id).
    Метрики:
      - num_plateaus (avg/min/max)
      - gap_ratio (avg/min/max, p90/p95/p99)
      - частоты резонансов (>=10, >=100, >=1000)
      - plateau_width (avg, max)
      - stable_count (avg)
      - shell_count (avg)
      - shell_purity (avg)
      - plateau_persistence (avg)
    Дополнительно фиксируются флаги абляции (abl_mixing и т.д.).
    """
    groups = defaultdict(list)
    for row in summary_rows:
        key = (row["N"], row["edge"], row.get("ablation_id", 0))
        groups[key].append(row)

    stats = []
    for (N, edge, ablation_id), rows in groups.items():
        num_plateaus = _clean([r.get("num_plateaus") for r in rows])
        gap_ratio = _clean([r.get("gap_ratio") for r in rows])

        plateau_widths = _clean([r.get("plateau_width") for r in rows])
        plateau_width_maxes = _clean([r.get("plateau_width_max") for r in rows])

        # --- новые метрики ---
        stable_counts = _clean([r.get("stable_count") for r in rows])
        stable_width_avg = _clean([r.get("stable_width_avg") for r in rows])
        stable_width_max = _clean([r.get("stable_width_max") for r in rows])

        shell_counts = _clean([r.get("shell_count") for r in rows])
        shell_purities = _clean([r.get("shell_purity") for r in rows])
        plateau_persist = _clean([r.get("plateau_persistence") for r in rows])

        res10 = sum(1 for g in gap_ratio if g >= 10)
        res100 = sum(1 for g in gap_ratio if g >= 100)
        res1000 = sum(1 for g in gap_ratio if g >= 1000)

        # берём флаги абляции из первой строки (они одинаковые для всех строк группы)
        abl_flags = {}
        if rows:
            abl_flags = {
                "abl_mixing": rows[0].get("abl_mixing", True),
                "abl_cycles": rows[0].get("abl_cycles", True),
                "abl_snail": rows[0].get("abl_snail", True),
                "abl_reservoir": rows[0].get("abl_reservoir", True),
                "abl_dual": rows[0].get("abl_dual", True),
            }

        stat = {
            "N": N,
            "edge": edge,
            "ablation_id": ablation_id,
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
            # устойчивые плато
            "stable_count_avg": float(np.mean(stable_counts)) if stable_counts else None,
            "stable_width_avg": float(np.mean(stable_width_avg)) if stable_width_avg else None,
            "stable_width_max": float(np.mean(stable_width_max)) if stable_width_max else None,
            # оболочки
            "shell_count_avg": float(np.mean(shell_counts)) if shell_counts else None,
            "shell_purity_avg": float(np.mean(shell_purities)) if shell_purities else None,
            # persistence
            "plateau_persistence_avg": float(np.mean(plateau_persist)) if plateau_persist else None,
        }

        # добавляем флаги абляции
        stat.update(abl_flags)
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
