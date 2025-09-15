# universe_runner.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.1
# "Скрипт Вселенной" — минимально честная версия (зонтик + лотос + край)
# Напарник: этот каркас уже честно учитывает влияние зонтика/лотоса на спектр
# и сохраняет воспроизводимые логи (версии, хэши, копию конфига, summary.csv/txt).

import os
import csv
import json
import yaml
import hashlib
import shutil
import datetime
import numpy as np
from tqdm import tqdm

# --- Версия и хэши (жёстко) ---
VERSION = "3.1.1"
PROGRAM_HASH = "T3C9-EXAMPLE-0001"  # фиксированный хэш кода

# --- Утилиты ---
def sha256_of_file(path: str) -> str:
    """SHA-256 хэш файла блоками (без загрузки всего в память)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def timestamp() -> str:
    """Строковый идентификатор запуска."""
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

def robust_plateaus(eigs: np.ndarray, min_width: int, tau: float = 0.35):
    """
    Кластеризация собственных значений в плато по относительному порогу.
    Порог разрыва между соседями: thr = tau * median(diff).
    Масштаб-инвариантность за счёт использования медианы.
    """
    if eigs.size < 2:
        return []
    diffs = np.diff(eigs)
    med = np.median(diffs) if (diffs.size > 0 and np.all(np.isfinite(diffs))) else 0.0
    thr = tau * med if med > 0 else np.inf  # если спектр вырожден — не режем
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

# --- Геометрия/операторы ---
def laplacian_1d(N: int, edge_kind: str) -> np.ndarray:
    """
    1D лапласиан (реальный симметричный):
      - 'hard'  => периодические ГУ (кольцо)
      - иные    => открытые ГУ (цепочка)
    """
    L = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        L[i, i] = 2.0
        if i - 1 >= 0:
            L[i, i - 1] = -1.0
        if i + 1 < N:
            L[i, i + 1] = -1.0
    if edge_kind == "hard":  # periodic
        L[0, N - 1] = -1.0
        L[N - 1, 0] = -1.0
    # open-край уже отражён в матрице (нет замыкания)
    return L

def sample_hyperparams(rng: np.random.Generator, cfg: dict) -> dict:
    """Простой равномерный сэмплинг по диапазонам cfg (без внешних пакетов)."""
    umb = cfg["umbrella"]
    lotus = cfg["lotus_fix"]
    cycles = cfg["cycles"]
    knot = cfg["knot"]
    edge_choices = umb["edge"]

    # зонтик
    alpha = rng.uniform(*umb["alpha"])
    kappa = rng.uniform(*umb["kappa"])

    # ступени (до 3 фиксированных радиусов, случайные высоты/ширины)
    n_steps = int(rng.choice(umb["steps"]["N_steps"]))
    r_list_full = umb["steps"]["r_over_R"]
    if n_steps > 0:
        pick = min(n_steps, len(r_list_full))
        r_steps = sorted(rng.choice(r_list_full, size=pick, replace=False))
    else:
        r_steps = []
    h_lo, h_hi = umb["steps"]["h_range"]
    w_lo, w_hi = umb["steps"]["w_range"]
    steps = [{"r_over_R": float(r), "h": rng.uniform(h_lo, h_hi), "w": rng.uniform(w_lo, w_hi)} for r in r_steps]

    # анизотропия
    m = int(rng.choice(umb["anisotropy"]["m_choices"]))
    eps = rng.uniform(*umb["anisotropy"]["epsilon"])
    # ВАЖНО: берём alpha_theta из блока anisotropy (исправлено)
    alpha_theta = float(umb["anisotropy"].get("alpha_theta", 1.0))

    # край
    edge = str(rng.choice(edge_choices))

    # лотос/фиксация
    gamma = rng.uniform(*lotus["gamma"]) if isinstance(lotus["gamma"], list) else float(lotus["gamma"])
    beta  = rng.uniform(*lotus["beta"])  if isinstance(lotus["beta"], list)  else float(lotus["beta"])

    # циклы
    delta = rng.uniform(*cycles["delta"]) if isinstance(cycles["delta"], list) else float(cycles["delta"])
    P = int(rng.choice(cycles["P_choices"]))

    # узел
    eta = rng.uniform(*knot["eta"]) if isinstance(knot["eta"], list) else float(knot["eta"])
    g_local = int(rng.choice(knot["locality_g"]))

    return {
        "umbrella": {
            "alpha": alpha,
            "kappa": kappa,
            "steps": steps,
            "anisotropy": {"m": m, "epsilon": eps, "alpha_theta": alpha_theta},
            "edge": edge,
        },
        "lotus": {"gamma": gamma, "beta": beta},
        "cycles": {"delta": delta, "P": P},
        "knot": {"eta": eta, "g": g_local},
    }

def umbrella_diagonal(N: int, params: dict) -> np.ndarray:
    """
    Диагональный потенциал V(r,theta) на 1D кольце/цепочке:
      r ∈ [0,1], theta = 2π i/N
    """
    i = np.arange(N, dtype=np.float64)
    r = i / (N - 1 if N > 1 else 1.0)
    theta = 2.0 * np.pi * i / max(N, 1)

    umb = params["umbrella"]
    V = umb["alpha"] * r + umb["kappa"] * r**2

    # ступени (сигмоиды)
    for st in umb["steps"]:
        rj, h, w = st["r_over_R"], st["h"], st["w"]
        V += h * 0.5 * (1.0 + np.tanh((r - rj) / max(w, 1e-6)))

    # анизотропия
    m = umb["anisotropy"]["m"]
    eps = umb["anisotropy"]["epsilon"]
    alpha_theta = umb["anisotropy"]["alpha_theta"]
    V += alpha_theta * eps * np.cos(m * theta)

    return V

def build_H_eff(N: int, params: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Минимальный H_eff: (1+gamma)*L + diag(V) + слабый симм. шум.
    Это честная зависимость спектра от зонтика/лотоса/края.
    """
    edge = params["umbrella"]["edge"]
    L = laplacian_1d(N, edge_kind=edge)
    V_diag = umbrella_diagonal(N, params)
    gamma = params["lotus"]["gamma"]

    # базовый кинетический член + "лотос" как сглаживатель
    H = (1.0 + gamma) * L + np.diag(V_diag)

    # мягкий стабилизирующий шум (симметризованный), чтобы избегать редких вырождений
    noise = 1e-6 * rng.standard_normal((N, N))
    H = H + (noise + noise.T) * 0.5
    return H

# --- Загрузка конфига ---
import os
CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "config",
    "config_full.yaml"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

CONFIG_HASH_DECLARED = cfg.get("CONFIG_HASH", "")
CONFIG_VERSION = cfg.get("CONFIG_VERSION", "unknown")

# --- Проверка согласованности версий/хэшей ---
if cfg.get("VERSION") != VERSION or cfg.get("PROGRAM_HASH") != PROGRAM_HASH:
    raise RuntimeError("Config/Program mismatch (VERSION or PROGRAM_HASH)!")

CONFIG_FILE_SHA256 = sha256_of_file(CONFIG_PATH)
CONFIG_HASH_MATCH = (CONFIG_FILE_SHA256 == CONFIG_HASH_DECLARED)

# --- Папка запуска ---
RUN_ID = timestamp()
ROOT = os.path.join(cfg["runs_root"], RUN_ID)
os.makedirs(ROOT, exist_ok=True)

# Сохраняем копию конфига внутрь папки запуска (воспроизводимость)
try:
    shutil.copy2(CONFIG_PATH, os.path.join(ROOT, os.path.basename(CONFIG_PATH)))
except Exception:
    # не фейлим прогон из-за проблем копирования; просто идём дальше
    pass

# --- Лог мета ---
meta = {
    "VERSION": VERSION,
    "PROGRAM_HASH": PROGRAM_HASH,
    "CONFIG_VERSION": CONFIG_VERSION,
    "CONFIG_HASH_declared": CONFIG_HASH_DECLARED,
    "CONFIG_FILE_sha256": CONFIG_FILE_SHA256,
    "config_hash_match": CONFIG_HASH_MATCH,
    "time_start": RUN_ID,
    "sizes": cfg["sizes"],
    "seeds": cfg["seeds"],
    "sampling": cfg["sampling"],
}
with open(os.path.join(ROOT, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# --- Основной цикл ---
sizes = list(cfg["sizes"])
seeds = list(cfg["seeds"])
points = int(cfg["sampling"]["points"])
minw = int(cfg["filters"]["plateau_min_width"])
# позволим через конфиг переопределять tau (иначе дефолт 0.35)
plateau_tau = float(cfg.get("filters", {}).get("plateau_tau", 0.35))

summary_rows = []
total_iter = len(sizes) * len(seeds) * points
pbar = tqdm(total=total_iter, desc="Universe run", ncols=100)

for N in sizes:
    for s in seeds:
        for i in range(points):
            # независимый генератор на (seed, run)
            rng = np.random.default_rng(np.random.SeedSequence([s, i]))

            # сэмплинг гиперпараметров
            params = sample_hyperparams(rng, cfg)

            # построение H_eff
            H = build_H_eff(N, params, rng)

            # собственные значения
            try:
                eigvals = np.linalg.eigvalsh(H)
            except np.linalg.LinAlgError:
                # пропускаем редкий неустойчивый случай
                pbar.update(1)
                continue

            eigvals.sort(kind="mergesort")  # стабильная сортировка

            # плато
            plateaus = robust_plateaus(eigvals, min_width=minw, tau=plateau_tau)

            # gap ratio g3
            g3 = None
            if len(plateaus) >= 3:
                l2 = float(np.mean(plateaus[1]))
                l3 = float(np.mean(plateaus[2]))
                if len(plateaus) > 3:
                    l4 = float(np.mean(plateaus[3]))
                    denom = (l3 - l2)
                    if denom != 0 and np.isfinite(denom):
                        g3 = (l4 - l3) / denom

            summary_rows.append({
                "run_id": RUN_ID,
                "N": N,
                "seed": s,
                "point": i,
                "num_plateaus": len(plateaus),
                "gap_ratio": (None if g3 is None else float(g3)),
                "edge": params["umbrella"]["edge"],
                "alpha": float(params["umbrella"]["alpha"]),
                "kappa": float(params["umbrella"]["kappa"]),
                "gamma": float(params["lotus"]["gamma"]),
                "m": int(params["umbrella"]["anisotropy"]["m"]),
                "eps": float(params["umbrella"]["anisotropy"]["epsilon"]),
            })

            pbar.update(1)

pbar.close()
# --- Агрегация выжимки ---
import statistics

stats = {}
for row in summary_rows:
    N = row["N"]
    edge = row["edge"]
    stats.setdefault(N, {}).setdefault(edge, {"plateaus": [], "gaps": []})
    stats[N][edge]["plateaus"].append(row["num_plateaus"])
    if row["gap_ratio"] is not None:
        stats[N][edge]["gaps"].append(row["gap_ratio"])

# Формируем компактные данные
summary_stats = {}
for N, edges in stats.items():
    summary_stats[N] = {}
    for edge, vals in edges.items():
        p = vals["plateaus"]
        g = vals["gaps"]
        summary_stats[N][edge] = {
            "num_plateaus_avg": statistics.mean(p) if p else None,
            "num_plateaus_min": min(p) if p else None,
            "num_plateaus_max": max(p) if p else None,
            "gap_ratio_avg": statistics.mean(g) if g else None,
            "gap_ratio_min": min(g) if g else None,
            "gap_ratio_max": max(g) if g else None,
            "count": len(p),
        }

# Сохраняем в JSON
stats_path = os.path.join(ROOT, "summary_stats.json")
with open(stats_path, "w", encoding="utf-8") as fstats:
    json.dump(summary_stats, fstats, indent=2, ensure_ascii=False)

# И в CSV для удобства
stats_csv_path = os.path.join(ROOT, "summary_stats.csv")
with open(stats_csv_path, "w", newline="", encoding="utf-8") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["N", "edge", "num_plateaus_avg", "num_plateaus_min", "num_plateaus_max",
                     "gap_ratio_avg", "gap_ratio_min", "gap_ratio_max", "count"])
    for N, edges in summary_stats.items():
        for edge, vals in edges.items():
            writer.writerow([
                N, edge,
                vals["num_plateaus_avg"],
                vals["num_plateaus_min"],
                vals["num_plateaus_max"],
                vals["gap_ratio_avg"],
                vals["gap_ratio_min"],
                vals["gap_ratio_max"],
                vals["count"],
            ])


# --- Сохраняем summary (CSV и TXT) ---
csv_path = os.path.join(ROOT, "summary.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
    if summary_rows:
        writer = csv.DictWriter(fcsv, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    else:
        # всё равно запишем заголовок с базовыми полями
        writer = csv.DictWriter(fcsv, fieldnames=[
            "run_id", "N", "seed", "point", "num_plateaus",
            "gap_ratio", "edge", "alpha", "kappa", "gamma", "m", "eps"
        ])
        writer.writeheader()

txt_path = os.path.join(ROOT, "summary.txt")
with open(txt_path, "w", encoding="utf-8") as ftxt:
    for row in summary_rows:
        ftxt.write(str(row) + "\n")
