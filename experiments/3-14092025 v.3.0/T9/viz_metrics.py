# viz_metrics.py
# ZFSC v3.3 — Визуализация и агрегаты метрик (без seaborn; один график на фигуру)

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _parse_shell_sizes(val):
    # summary_rows["shell_sizes"] хранится как строка "3|4|7" (см. single_run.py)
    if val is None:
        return []
    if isinstance(val, str):
        return [int(s) for s in val.split("|") if s.strip().isdigit()]
    if isinstance(val, (list, tuple)):
        return list(val)
    return []

def _agg(values, mode="mean"):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    mode = str(mode).lower()
    if mode == "p95":
        return float(np.percentile(arr, 95))
    if mode == "p90":
        return float(np.percentile(arr, 90))
    if mode == "median":
        return float(np.median(arr))
    return float(np.mean(arr))  # default mean

def _save_csv_matrix(matrix, xs, ys, path):
    # xs — ось X (в PNG это будет горизонтальная), ys — ось Y (вертикальная)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Y\\X"] + list(xs))
        for j, y in enumerate(ys):
            row = [y] + [("" if (matrix[j, i] is None) else matrix[j, i]) for i, _ in enumerate(xs)]
            w.writerow(row)

def resonance_map(summary_rows, out_dir, cfg):
    """
    Теплокарта агрегированного gap_ratio в координатах (N, P).
    По умолчанию агрегирование 'p95' — хвост резонансов.
    """
    _ensure_dir(out_dir)
    viz_cfg = cfg.get("viz", {})
    agg_mode = viz_cfg.get("resonance", {}).get("agg", "p95")

    # Сгруппируем по (N, P)
    data = {}
    Ns, Ps = set(), set()
    for r in summary_rows:
        N = r.get("N")
        P = r.get("P")
        g = r.get("gap_ratio")
        if (N is None) or (P is None):
            continue
        g = _safe_float(g)
        if g is None:
            continue
        Ns.add(N); Ps.add(P)
        data.setdefault((N, P), []).append(g)

    if not data:
        return  # нечего рисовать

    xs = sorted(Ns)
    ys = sorted(Ps)
    mat = np.full((len(ys), len(xs)), np.nan, dtype=float)

    for j, p in enumerate(ys):
        for i, n in enumerate(xs):
            vals = data.get((n, p), [])
            agg = _agg(vals, agg_mode)
            mat[j, i] = np.nan if (agg is None) else agg

    # CSV
    _save_csv_matrix(mat, xs, ys, os.path.join(out_dir, "resonance_map.csv"))

    # PNG
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(mat, origin="lower", aspect="auto",
                   extent=[min(xs)-0.5, max(xs)+0.5, min(ys)-0.5, max(ys)+0.5])
    fig.colorbar(im, ax=ax, label=f"gap_ratio ({agg_mode})")
    ax.set_xlabel("N")
    ax.set_ylabel("P (cycles)")
    ax.set_title("Resonance map: gap_ratio vs (N, P)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "resonance_map.png"), dpi=200)
    plt.close(fig)

def shell_hist(summary_rows, out_dir, cfg):
    """
    Две гистограммы:
      1) по числу оболочек (shell_count) на прогон,
      2) по размерам оболочек (объединяем все 'shell_sizes').
    Пишем PNG + CSV.
    """
    _ensure_dir(out_dir)

    # 1) Гисто по shell_count
    counts = []
    for r in summary_rows:
        v = r.get("shell_count")
        if v is not None:
            try:
                counts.append(int(v))
            except Exception:
                pass

    if counts:
        # CSV
        csv_path = os.path.join(out_dir, "shell_count_hist.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shell_count"])
            for c in counts:
                w.writerow([c])

        # PNG
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.hist(counts, bins=range(min(counts), max(counts)+2))
        ax1.set_xlabel("Число оболочек на прогон")
        ax1.set_ylabel("Частота")
        ax1.set_title("Histogram: shell_count")
        plt.tight_layout()
        fig1.savefig(os.path.join(out_dir, "shell_count_hist.png"), dpi=200)
        plt.close(fig1)

    # 2) Гисто по размерам оболочек
    sizes = []
    for r in summary_rows:
        sizes.extend(_parse_shell_sizes(r.get("shell_sizes")))

    if sizes:
        csv_path = os.path.join(out_dir, "shell_size_hist.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shell_size"])
            for s in sizes:
                w.writerow([s])

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.hist(sizes, bins=range(min(sizes), max(sizes)+2))
        ax2.set_xlabel("Размер оболочки (число λ в кластере)")
        ax2.set_ylabel("Частота")
        ax2.set_title("Histogram: shell_size")
        plt.tight_layout()
        fig2.savefig(os.path.join(out_dir, "shell_size_hist.png"), dpi=200)
        plt.close(fig2)

def scaling_num_plateaus(summary_rows, out_dir, cfg):
    """
    Кривая ⟨num_plateaus⟩ по N (с барами дисперсии).
    Пишем PNG + CSV.
    """
    _ensure_dir(out_dir)
    # группируем по N
    byN = {}
    for r in summary_rows:
        N = r.get("N")
        k = r.get("num_plateaus")
        if (N is None) or (k is None):
            continue
        try:
            N = int(N)
            k = float(k)
        except Exception:
            continue
        byN.setdefault(N, []).append(k)

    if not byN:
        return

    xs = sorted(byN.keys())
    means = [float(np.mean(byN[n])) for n in xs]
    stds  = [float(np.std(byN[n]))  for n in xs]

    # CSV
    csv_path = os.path.join(out_dir, "scaling_num_plateaus.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["N", "mean_num_plateaus", "std_num_plateaus", "count"])
        for n in xs:
            w.writerow([n, float(np.mean(byN[n])), float(np.std(byN[n])), len(byN[n])])

    # PNG
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(xs, means, yerr=stds, fmt="o-")
    ax.set_xlabel("N")
    ax.set_ylabel("⟨num_plateaus⟩")
    ax.set_title("Scaling: ⟨num_plateaus⟩ vs N")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "scaling_num_plateaus.png"), dpi=200)
    plt.close(fig)

def build_viz(summary_rows, level_root, cfg):
    """
    Единая точка входа: строит все три артефакта в level_root/viz/.
    Управляется флагом cfg['viz'].
    """
    viz_cfg = cfg.get("viz", {})
    if not viz_cfg or not bool(viz_cfg.get("enabled", False)):
        return

    out = os.path.join(level_root, "viz")
    _ensure_dir(out)

    # Включаем/выключаем отдельные графики по флагам
    if viz_cfg.get("resonance_map", True):
        resonance_map(summary_rows, os.path.join(out, "resonance"), cfg)

    if viz_cfg.get("shell_hist", True):
        shell_hist(summary_rows, os.path.join(out, "shell_hist"), cfg)

    if viz_cfg.get("scaling_num_plateaus", True):
        scaling_num_plateaus(summary_rows, os.path.join(out, "scaling"), cfg)
