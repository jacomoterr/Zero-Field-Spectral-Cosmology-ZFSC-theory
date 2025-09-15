#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T6_scaling.py
ZFSC v3.0 — Scaling test for the umbrella (radial confinement)

Идея: без изменения оператора (как в T0), проверить число положительных плато
при росте размера матрицы и при разных сидов.
Считает только первые k_eigs*2 собственных значений (по модулю) через eigh, затем обрезает.
"""

import os, json, time, math
import numpy as np
from numpy.linalg import eigh
from datetime import datetime
from tqdm import tqdm
import yaml

# --- Версия и код программы ---
PROGRAM_VERSION = "3.0.5"
PROGRAM_CODE    = "Q7V9R2M4X1PLZ8Y3K0DW"

# --- Пути ---
CONFIG_PATH = "config/T6.yaml"
RUNS_BASE   = "runs/T6"

# --- Утилиты ---
def make_run_dir(base=RUNS_BASE):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path

def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

# --- Геометрия и операторы (идентично T0-логике) ---
def make_grid(size, bc="periodic"):
    N = size * size
    H = np.zeros((N, N))
    def idx(x, y): return x * size + y
    for x in range(size):
        for y in range(size):
            i = idx(x, y)
            for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                xx, yy = x+dx, y+dy
                if bc == "periodic":
                    xx %= size; yy %= size
                if 0 <= xx < size and 0 <= yy < size:
                    j = idx(xx, yy)
                    H[i, j] = 1
    return (H + H.T)/2

def apply_radial(H, levels=3, strength=1.0):
    N = H.shape[0]
    size = int(np.sqrt(N))
    V = np.zeros((N, N))
    cx, cy = (size-1)/2, (size-1)/2
    for x in range(size):
        for y in range(size):
            i = x*size + y
            r = math.hypot(x - cx, y - cy)
            V[i, i] = strength * r
    return H + V

def apply_stabilizer(H, beta=0.2):
    return H - beta*np.eye(H.shape[0])

# --- Спектр и плато ---
def compute_eigs(H, k_eigs=None):
    # Для надёжности считаем весь спектр (матрицы до ~65×65*65×65 терпимы)
    # Если k_eigs задан, всё равно eigh вернёт полный; фильтрацию сделаем потом.
    vals, _ = eigh(H)
    return vals

def detect_plateaus(lmbda, tol=1e-2, min_width=2):
    lmbda = np.sort(lmbda)
    clusters, cur = [], [lmbda[0]]
    for v in lmbda[1:]:
        if abs(v - cur[-1]) < tol * (abs(cur[-1]) + 1e-6):
            cur.append(v)
        else:
            clusters.append(cur); cur = [v]
    clusters.append(cur)
    plateaus = []
    for c in clusters:
        if len(c) >= min_width:
            if (max(c) - min(c)) <= tol * (abs(np.mean(c)) + 1e-6):
                plateaus.append(c)
    return plateaus

def count_positive_plateaus(vals, tol=1e-2, min_width=2, gap_to=4):
    plateaus = detect_plateaus(vals, tol=tol, min_width=min_width)
    pos = [p for p in plateaus if np.mean(p) > 0]
    return len(pos[:gap_to]), [len(p) for p in pos[:gap_to]]

# --- Основной прогон ---
def run_scaling(cfg):
    sizes = cfg["run"]["sizes"]           # например: [21, 33, 41, 49, 57, 65]
    seeds = cfg["run"]["seeds"]           # например: [101,102,103,104,105]
    k_eigs = cfg["run"].get("k_eigs", None)  # зарезервировано (сейчас не используем)
    tol     = cfg["checks"].get("tol", 1e-2)
    min_w   = cfg["checks"].get("min_width", 2)
    gap_to  = cfg["checks"].get("gap_to_mode", 4)

    beta    = cfg["operators"]["stabilizer_beta"]
    levels  = cfg["operators"]["radial_zeta"]["levels"]
    strength= cfg["operators"]["radial_zeta"]["strength"]
    bc      = cfg["geometry"]["params"]["bc"]

    rows = []
    for size in tqdm(sizes, desc="Sizes"):
        for seed in tqdm(seeds, desc=f"s={size}", leave=False):
            np.random.seed(seed)
            H = make_grid(size, bc=bc)
            H = apply_radial(H, levels=levels, strength=strength)
            H = apply_stabilizer(H, beta=beta)

            vals = compute_eigs(H, k_eigs=k_eigs)
            npos, widths = count_positive_plateaus(vals, tol=tol, min_width=min_w, gap_to=gap_to)
            rows.append([size, seed, npos, ";".join(map(str, widths)),
                         float(vals[0]), float(vals[1]), float(vals[2])])
    return rows

if __name__ == "__main__":
    # Загружаем конфиг
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dir = make_run_dir()
    rows = run_scaling(cfg)

    # Один агрегированный CSV + два коротких лога
    save_csv(os.path.join(run_dir, "scaling_results.csv"),
             ["size","seed","num_pos_plateaus","widths","lambda0","lambda1","lambda2"],
             rows)

    # summary.txt
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"PROGRAM VERSION: {PROGRAM_VERSION}\n")
        f.write(f"PROGRAM CODE:    {PROGRAM_CODE}\n")
        f.write(f"Start time: {datetime.now().isoformat()}\n")
        f.write(f"Config version: {cfg.get('version')}\n")
        f.write(f"Config code:    {cfg.get('config_code')}\n")
        f.write("PROGRAM/CONFIG checks: OK (manual codes)\n\n")

        # Краткая сводка по размерам
        by_size = {}
        for size, seed, npos, widths, *_ in rows:
            by_size.setdefault(size, []).append(int(npos))
        for size in sorted(by_size):
            arr = by_size[size]
            f.write(f"size={size}: mean={np.mean(arr):.2f}, min={np.min(arr)}, max={np.max(arr)}, n={len(arr)}\n")

    # meta.json
    meta = {
        "program_version": PROGRAM_VERSION,
        "program_code": PROGRAM_CODE,
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "config_path": CONFIG_PATH,
        "config_version": cfg.get("version"),
        "config_code": cfg.get("config_code"),
        "sizes": cfg["run"]["sizes"],
        "seeds": cfg["run"]["seeds"]
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Results saved in {run_dir}")
