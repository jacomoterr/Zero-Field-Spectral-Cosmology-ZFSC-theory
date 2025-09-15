# level.py
# Один уровень адаптивного прогона

import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from core.single_run import single_run
from hotspots import detect_hotspots, save_hotspots
from stats import make_summary_stats, write_summary_stats
from outputs import save_summary
from viz_metrics import build_viz


def _score(r):
    """Ранжирование hotspot по persistence + purity + gap_ratio"""
    import math
    gr = max(0.0, float(r.get("gap_ratio") or 0.0))
    p_pers = max(0.0, float(r.get("plateau_persistence") or 0.0))
    p_pur = max(0.0, float(r.get("shell_purity") or 0.0))
    GR_MAX = 1000.0
    return 0.5 * p_pers + 0.35 * p_pur + 0.15 * math.log1p(gr) / math.log1p(GR_MAX)


def run_level(level, hotspots, cfg, root, sublevel=""):
    """Запускает один уровень adaptive"""
    print(f"\n▶ Starting Level {level}{sublevel}")
    print(f"  Hotspots in: {len(hotspots)}")

    points = int(cfg["sampling"]["points"] * (cfg["adaptive"]["points_mult"] ** level))
    N_step = int(cfg["adaptive"]["N_step"])
    n_probe_base = cfg["stability"].get("n_probe", 1)
    n_probe = int(n_probe_base + level * cfg["adaptive"]["n_probe_step"])
    sigma_base = float(cfg["stability"].get("sigma", 1e-5))
    sigma = sigma_base * (cfg["adaptive"]["sigma_mult"] ** level)

    N_JOBS = int(cfg.get("parallel", {}).get("N_JOBS", 8))
    modes = cfg.get("ablation", {}).get("modes", [{}])
    if not modes:
        modes = [{}]

    stab_params = {"q_min": cfg["stability"].get("q_min", 3.0),
                   "n_probe": n_probe,
                   "sigma": sigma}

    # --- Формируем список задач ---
    if level == 0 and sublevel == "":
        jobs = [(N, s, i, a_id, mode, stab_params, level, cfg)
                for N in cfg["sizes"]
                for s in cfg["seeds"]
                for i in range(points)
                for a_id, mode in enumerate(modes)]
    else:
        jobs = [(hp["N"] + level * N_step,
                 hp["seed"],
                 hp["point"],
                 hp["ablation_id"],
                 modes[hp["ablation_id"]],
                 stab_params,
                 level,
                 cfg)
                for hp in hotspots]

    # ⚡ Сначала печатаем информацию
    print(f"  Jobs prepared: {len(jobs)} (points={points}, n_probe={n_probe}, sigma={sigma})")

    level_root = os.path.join(root, f"level{level}{sublevel}")
    os.makedirs(level_root, exist_ok=True)

    # --- Основной прогон ---
    results = Parallel(n_jobs=N_JOBS)(
        delayed(single_run)(*job) for job in tqdm(
            jobs,
            desc=f"Level {level}{sublevel} compute",
            total=len(jobs),
            unit="task",
            dynamic_ncols=True,
            leave=True
        )
    )

    summary_rows = [r for (r, p) in results if r is not None]
    save_summary(summary_rows, level_root)
    summary_stats = make_summary_stats(summary_rows)
    write_summary_stats(summary_stats, level_root)
    print(f"  Saved summary & stats for Level {level}{sublevel}")

    # --- Визуализация метрик (опционально через cfg['viz']) ---
    try:
        build_viz(summary_rows, level_root, cfg)
    except Exception as e:
        print(f"  [viz] skipped due to error: {e}")

    # --- Hotspots обработка ---
    print(f"  Processing hotspots for Level {level}{sublevel} ...")
    hotspots_next = []
    for r in tqdm(summary_rows, desc="Hotspot filter", unit="row", dynamic_ncols=True):
        hs = detect_hotspots([r], cfg)
        if hs:
            hotspots_next.extend(hs)

    # --- Ранжирование и top-K ---
    hotspots_next.sort(key=_score, reverse=True)
    topk_cfg = cfg.get("hotspots", {}).get("topk_level", [])
    if level < len(topk_cfg):
        cap = topk_cfg[level]
        if cap and len(hotspots_next) > cap:
            hotspots_next = hotspots_next[:cap]

    save_hotspots(hotspots_next, os.path.join(level_root, f"hotspots_level{level}{sublevel}.json"))

    # --- Эффективность ---
    tasks_total = len(jobs)
    hotspots_found = len(hotspots_next)
    efficiency = (hotspots_found / tasks_total * 100) if tasks_total > 0 else 0
    speedup = (tasks_total / hotspots_found) if hotspots_found > 0 else float("inf")

    print(f"✅ Level {level}{sublevel} complete.")
    print(f"  tasks = {tasks_total}, hotspots = {hotspots_found}")
    print(f"  efficiency = {efficiency:.2f}%")
    print(f"  speedup vs global = ×{speedup:.1f}")

    # --- Метрики для авто-стопа и отчётов ---
    pers_vals = [r.get("plateau_persistence") for r in hotspots_next if r.get("plateau_persistence") is not None]
    median_persistence = np.median(pers_vals) if pers_vals else 0

    pur_vals = [r.get("shell_purity") for r in hotspots_next if r.get("shell_purity") is not None]
    avg_purity = np.mean(pur_vals) if pur_vals else 0

    gap_vals = [r.get("gap_ratio") for r in hotspots_next if r.get("gap_ratio") is not None]
    median_gap = np.median(gap_vals) if gap_vals else 0

    return hotspots_next, (median_persistence, avg_purity, median_gap)
