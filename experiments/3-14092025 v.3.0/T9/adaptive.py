# adaptive.py
# Управление лестницей уровней (авто-стоп, подуровни, финальный stress-test)

import numpy as np
from level import run_level
from stress import stress_test


def adaptive_run(cfg, root):
    hotspots = []
    level = 0
    prev_metrics = None
    max_levels = cfg["adaptive"].get("max_levels", 15)

    while level < max_levels:
        hotspots, metrics = run_level(level, hotspots, cfg, root)
        median_persistence, avg_purity, median_gap = metrics

        # --- Сводный отчёт ---
        print(f"📊 Summary L{level}:")
        print(f"   median persistence = {median_persistence:.3f}")
        print(f"   avg purity         = {avg_purity:.3f}")
        print(f"   median gap_ratio   = {median_gap:.3f}")

        # --- Авто-стоп ---
        if not hotspots or len(hotspots) < cfg["hotspots"].get("K_min", 20):
            print(f"⏹️ Auto-stop: мало точек ({len(hotspots)})")
            break

        if prev_metrics:
            dp = abs(median_persistence - prev_metrics[0])
            dq = abs(avg_purity - prev_metrics[1])
            if dp < 0.02 and dq < 0.02:
                print(f"⏹️ Auto-stop: стабилизация (Δpers={dp:.3f}, Δpur={dq:.3f})")
                break

        prev_metrics = (median_persistence, avg_purity, median_gap)

        # --- Подуровни ---
        if len(hotspots) > 2000:
            print(f"↳ Sub-level triggered at Level {level}")
            hotspots, metrics = run_level(level, hotspots, cfg, root, sublevel="a")
            median_persistence, avg_purity, median_gap = metrics
            print(f"📊 Summary L{level}a:")
            print(f"   median persistence = {median_persistence:.3f}")
            print(f"   avg purity         = {avg_purity:.3f}")
            print(f"   median gap_ratio   = {median_gap:.3f}")
            prev_metrics = metrics

        level += 1

    # --- Финальный stress-test (если включён в конфиг) ---
    stress_cfg = cfg.get("stress", {})
    if stress_cfg.get("enabled", False):
        print("\n🚨 Запуск финального stress-test ...")
        try:
            stress_test(hotspots, cfg, root, level_tag=f"L{level}")
        except Exception as e:
            print(f"[stress] ошибка: {e}")


def stress_only_run(cfg, root):
    """Режим: только stress-test для заданных вручную узлов"""
    hotspots = cfg.get("hotspots_manual", [])
    stress_test(hotspots, cfg, root, level_tag="manual")
