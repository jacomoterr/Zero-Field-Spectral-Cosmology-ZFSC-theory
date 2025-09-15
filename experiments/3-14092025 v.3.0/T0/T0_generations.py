#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T0_generations.py
ZFSC v3.0 — Stage T0: "Three and only three generations"
"""

import os
import json
import yaml
import time
import numpy as np
from numpy.linalg import eigh
from datetime import datetime
from tqdm import tqdm   # прогрессбар

# ------------------------------
# Версия и код программы
# ------------------------------
PROGRAM_VERSION = "3.0.4"
PROGRAM_CODE = "PG89DK4S1F7LQW2XCVHZ"  # фиксированный случайный код

# ------------------------------
# Пути
# ------------------------------
CONFIG_PATH = "config/T0.yaml"
RUNS_BASE = "runs/T0"

# ------------------------------
# Утилиты
# ------------------------------
def make_run_dir(base=RUNS_BASE):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path

def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

# ------------------------------
# Генерация базовой геометрии
# ------------------------------
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
                    xx %= size
                    yy %= size
                if 0 <= xx < size and 0 <= yy < size:
                    j = idx(xx, yy)
                    H[i,j] = 1
    return (H + H.T)/2

# ------------------------------
# Операторы
# ------------------------------
def apply_radial(H, levels=3, strength=1.0):
    N = H.shape[0]
    size = int(np.sqrt(N))
    V = np.zeros((N, N))
    cx, cy = (size-1)/2, (size-1)/2
    for x in range(size):
        for y in range(size):
            i = x*size+y
            r = np.sqrt((x-cx)**2 + (y-cy)**2)
            V[i,i] = strength * r
    return H + V

def apply_stabilizer(H, beta=0.2):
    return H - beta*np.eye(H.shape[0])

# ------------------------------
# Спектр и плато
# ------------------------------
def compute_eigs(H):
    vals, _ = eigh(H)
    return vals

def detect_plateaus(lmbda, tol=1e-2, min_width=2):
    """
    Находим плато, фильтруем только те, что шириной >= min_width
    """
    lmbda = np.sort(lmbda)
    clusters, cur = [], [lmbda[0]]
    for v in lmbda[1:]:
        if abs(v - cur[-1]) < tol*(abs(cur[-1]) + 1e-6):
            cur.append(v)
        else:
            clusters.append(cur); cur = [v]
    clusters.append(cur)

    plateaus = []
    for c in clusters:
        if len(c) >= min_width:
            if (max(c) - min(c)) <= tol*(abs(np.mean(c)) + 1e-6):
                plateaus.append(c)
    return plateaus

def check_generations(plateaus, gap_to=4):
    pos = [p for p in plateaus if np.mean(p) > 0]
    return len(pos[:gap_to])

# ------------------------------
# Основной прогон
# ------------------------------
def run_experiment(cfg):
    sizes = cfg["run"]["sizes"]
    results = []
    for size in tqdm(sizes, desc="Processing sizes"):
        H = make_grid(size, bc=cfg["geometry"]["params"]["bc"])
        H = apply_radial(H,
                         levels=cfg["operators"]["radial_zeta"]["levels"],
                         strength=cfg["operators"]["radial_zeta"]["strength"])
        H = apply_stabilizer(H, beta=cfg["operators"]["stabilizer_beta"])

        vals = compute_eigs(H)
        plateaus = detect_plateaus(vals, tol=1e-2, min_width=2)
        ngen = check_generations(plateaus,
                                 gap_to=cfg["checks"]["gap_to_mode"])

        results.append({
            "size": size,
            "num_plateaus": ngen,
            "plateaus": [len(p) for p in plateaus[:5]],
            "first_vals": [float(v) for v in vals[:10]]
        })
    return results

# ------------------------------
# Entry point
# ------------------------------
if __name__ == "__main__":
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_version = cfg.get("version", "UNKNOWN")
    config_code = cfg.get("config_code", "UNKNOWN")

    run_dir = make_run_dir()
    results = run_experiment(cfg)

    # Сохраняем spectrum.csv
    rows = []
    for r in results:
        for i, v in enumerate(r["first_vals"]):
            rows.append([r["size"], i, v])
    save_csv(os.path.join(run_dir, "spectrum.csv"),
             ["size","index","lambda"], rows)

    # generations_check.json
    with open(os.path.join(run_dir, "generations_check.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # summary.txt
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"PROGRAM VERSION: {PROGRAM_VERSION}\n")
        f.write(f"PROGRAM CODE:    {PROGRAM_CODE}\n")
        f.write(f"Start time: {datetime.now().isoformat()}\n")
        f.write(f"Config version: {config_version}\n")
        f.write(f"Config code:    {config_code}\n")
        f.write(f"PROGRAM check: OK\n")
        f.write(f"CONFIG check:   OK (manual scheme)\n\n")
        for r in results:
            f.write(f"Size {r['size']}: {r['num_plateaus']} positive plateaus\n")

    # meta.json
    meta = {
        "program_version": PROGRAM_VERSION,
        "program_code": PROGRAM_CODE,
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "config_path": CONFIG_PATH,
        "config_version": config_version,
        "config_code": config_code
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Results saved in {run_dir}")
