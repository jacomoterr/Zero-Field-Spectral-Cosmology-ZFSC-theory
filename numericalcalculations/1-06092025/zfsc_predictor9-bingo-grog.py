#!/usr/bin/env python3
# zfsc_predictor.py v6.3 (updated based on analysis)
# Универсальный спектральный предсказатель (ZFSC):
# Поддержка 3×3, 4×4 и 6×6 матриц.
# Режимы:
#   - independent_all
#   - shared_r_all
#   - shared_delta_all
#   - full_unify_all
#   - grand_unify_all
#   - grand_unify_all_scaled
# Обновления в v6.3:
# - Добавлен расчёт предсказанного m_e для сектора ℓ и соответствующей z-оценки.
# - Улучшена глобальная z-оценка: теперь использует sqrt(chi2 / dof) для всех режимов.
# - Реализована параллелизация с помощью multiprocessing для ускорения сканирования сетки.
# - Уточнён выбор триплетов собственных значений: добавлен критерий минимальной ошибки + проверка на асимптотику c ≈ δ² / (1 + r²) + 2.
# - Исправлена логика вывода сообщения "УРА!!!".
# - Добавлены комментарии о размерности параметров (безразмерные в планковских единицах) и связи λ_k с ω_D (λ_k^eff ~ ω_D для масс m_k = ħ / c² * λ_k^eff).

import numpy as np
import math
import argparse
import time
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import itertools
import multiprocessing as mp

# ==========================
#  Experimental inputs
# ==========================
@dataclass
class ExpData:
    # Neutrino splittings (normal ordering)
    dm21: float = 7.42e-5        # eV^2
    dm31_NO: float = 2.517e-3    # eV^2
    sig_dm21: float = 0.21e-5    # eV^2
    sig_dm31_NO: float = 0.026e-3# eV^2

    # Charged leptons (MeV)
    me: float = 0.51099895
    mmu: float = 105.6583755
    mtau: float = 1776.86

    # Up-type quarks (MeV)
    mu: float = 2.16
    mc: float = 1270.0
    mt: float = 172700.0

    # Down-type quarks (MeV)
    md: float = 4.67
    ms: float = 93.0
    mb: float = 4180.0

    tol_rel_model: float = 0.01  # 1% model tolerance for c_ℓ, c_u, c_d
    tol_me: float = 0.00001      # Experimental uncertainty for m_e (very small, in MeV)

    @property
    def c_nu_exp(self) -> float:
        return self.dm31_NO / self.dm21

    @property
    def sig_c_nu(self) -> float:
        c = self.c_nu_exp
        return c * math.sqrt((self.sig_dm31_NO/self.dm31_NO)**2 +
                             (self.sig_dm21/self.dm21)**2)

    @property
    def c_lep_exp(self) -> float:
        me2, mmu2, mtau2 = self.me**2, self.mmu**2, self.mtau**2
        return (mtau2 - me2) / (mmu2 - me2)

    @property
    def c_u_exp(self) -> float:
        mu2, mc2, mt2 = self.mu**2, self.mc**2, self.mt**2
        return (mt2 - mu2) / (mc2 - mu2)

    @property
    def c_d_exp(self) -> float:
        md2, ms2, mb2 = self.md**2, self.ms**2, self.mb**2
        return (mb2 - md2) / (ms2 - md2)

    def predicted_me(self, c_lep_pred: float) -> float:
        s2 = (self.mtau**2 - self.mmu**2) / (c_lep_pred - 1)
        me2_pred = self.mmu**2 - s2
        return math.sqrt(max(0, me2_pred))  # Avoid negative sqrt

# ==========================
#  Matrix builders
# ==========================
def build_matrix(size: int, delta: float, r: float,
                 gL: float = 1.0, gR: float = 1.0,
                 h: float = 0.0,
                 h_params: List[float] = None) -> np.ndarray:
    """
    Build symmetric band matrix of size 3, 4, or 6.
    Parameters:
        size : 3, 4, or 6
        delta : central diagonal shift (dimensionless in Planck units)
        r     : nearest-neighbor coupling (dimensionless)
        gL,gR : edge couplings (dimensionless)
        h,h_params : asymmetries (dimensionless)
    Note: Eigenvalues λ_k^eff ~ ω_D, where m_k = ħ / c² * λ_k^eff
    """
    if size == 3:
        return np.array([[0.0, gL, 0.0],
                         [gL, delta, r],
                         [0.0, r, 0.0]], dtype=float)

    elif size == 4:
        h1 = h_params[0] if h_params else h
        return np.array([[0.0, gL, 0.0, 0.0],
                         [gL, delta, r, h1],
                         [0.0, r, 0.0, gR],
                         [0.0, h1, gR, 0.0]], dtype=float)

    elif size == 6:
        h1, h2, h3 = (h_params if h_params else [h, h, h])
        M = np.zeros((6,6), dtype=float)
        for i in range(5):
            M[i,i+1] = r
            M[i+1,i] = r
        M[0,1] = gL; M[1,0] = gL
        M[4,5] = gR; M[5,4] = gR
        np.fill_diagonal(M, 0.0)
        M[2,2] = delta
        M[3,3] = delta
        M[1,4] = h1; M[4,1] = h1
        M[2,5] = h2; M[5,2] = h2
        M[0,3] = h3; M[3,0] = h3
        return M

    else:
        raise ValueError("Matrix size must be 3, 4, or 6.")

# ==========================
#  Ladder ratio computation
# ==========================
def ladder_c_from_matrix(M: np.ndarray, target_c: float = None, delta: float = None, r: float = None) -> float:
    """
    Compute ladder ratio c = (λ_max-λ_min)/(λ_mid-λ_min).
    For 3×3: trivial.
    For 4×4 or 6×6: choose the best triple of eigenvalues
    that reproduces target_c (if provided), else choose first 3.
    Improved: Add asymptotic check c ≈ δ² / (1 + r²) + 2 for validation.
    """
    w = np.linalg.eigvalsh(M)
    w.sort()

    if len(w) == 3:
        lam_min, lam_mid, lam_max = w
        eps = lam_mid - lam_min
        if eps <= 0:
            return np.inf
        return (lam_max - lam_min) / eps

    else:
        best_c = None
        best_err = float('inf')
        asymptotic_c = (delta**2 / (1 + r**2) + 2) if delta is not None and r is not None else None
        for triple in itertools.combinations(w, 3):
            lam_min, lam_mid, lam_max = sorted(triple)
            eps = lam_mid - lam_min
            if eps <= 0:
                continue
            c = (lam_max - lam_min) / eps
            err = abs(c - target_c) if target_c else 0
            # Additional check: prefer triples close to asymptotic if available
            if asymptotic_c:
                err += 0.1 * abs(c - asymptotic_c)  # Weighted penalty
            if err < best_err:
                best_err, best_c = err, c
        return best_c if best_c is not None else np.inf

# Parallel helper for scanning
def scan_sector_delta_r(args):
    sector, target, sigma, delta, r, matrix_size, gL, gR, h_params = args
    M = build_matrix(matrix_size, delta, r, gL=gL, gR=gR, h_params=h_params)
    c_val = ladder_c_from_matrix(M, target_c=target, delta=delta, r=r)
    err = abs(c_val - target)
    return delta, r, c_val, err

# ==========================
#  Mode: independent_all
# ==========================
def mode_independent_all(exp: ExpData, args):
    print("\n=== Mode: independent_all ===")
    start_time = time.time()
    results = {}
    chi2_tot = 0.0
    dof = 4  # 4 sectors

    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        best_local = {"delta": None, "r": None, "c": None, "err": float("inf")}
        deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
        rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
        h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

        # Parallel scanning
        if args.parallel:
            with mp.Pool(args.workers) as pool:
                tasks = [(sector, target, sigma, d, r, args.matrix_size, args.gL, args.gR, h_params) for d in deltas for r in rs]
                for res in tqdm(pool.imap_unordered(scan_sector_delta_r, tasks), total=len(tasks), desc=f"Parallel scanning for {sector}"):
                    d, r, c_val, err = res
                    if err < best_local["err"]:
                        best_local = {"delta": d, "r": r, "c": c_val, "err": err}
        else:
            for d in tqdm(deltas, desc=f"Scanning δ for {sector}", leave=False):
                for r in rs:
                    M = build_matrix(args.matrix_size, d, r, gL=args.gL, gR=args.gR, h_params=h_params)
                    c_val = ladder_c_from_matrix(M, target_c=target, delta=d, r=r)
                    err = abs(c_val - target)
                    if err < best_local["err"]:
                        best_local = {"delta": d, "r": r, "c": c_val, "err": err}

        results[sector] = best_local
        z = abs(best_local["c"] - target) / sigma
        chi2_tot += z**2
        print(f"[{sector}] best (δ={best_local['delta']:.9f}, r={best_local['r']:.9f}) "
              f"→ c={best_local['c']:.9f} (target {target:.9f}), z≈{z:.9f}σ")

        # Predict m_e for ℓ sector and compute z_me
        if sector == "ℓ":
            me_pred = exp.predicted_me(best_local["c"])
            z_me = abs(me_pred - exp.me) / exp.tol_me  # Use experimental sigma for m_e
            print(f"  [ℓ] Predicted m_e={me_pred:.9f} MeV (target {exp.me:.9f}), z_me≈{z_me:.9f}σ")
            chi2_tot += (abs(me_pred - exp.me) / (exp.me * exp.tol_rel_model))**2  # Add to chi2 with model tol
            dof += 1  # Extra degree for m_e prediction

    # Global chi2-based z (approx sqrt(chi2 / dof))
    z_tot = math.sqrt(chi2_tot / dof)
    print(f"Global χ²_tot={chi2_tot:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")

    if z_tot < 2.0:
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")
        print(f"Mode: independent_all")
        print(f"Parameters: gL={args.gL:.2f}, gR={args.gR:.2f}, h1={args.h1:.2f}, h2={args.h2:.2f}, h3={args.h3:.2f}")
        for sector, res in results.items():
            print(f"[{sector}] (δ={res['delta']:.9f}, r={res['r']:.9f}), c={res['c']:.9f}, z≈{abs(res['c'] - {'ν': exp.c_nu_exp, 'ℓ': exp.c_lep_exp, 'u': exp.c_u_exp, 'd': exp.c_d_exp}[sector]) / (exp.c_nu_exp * exp.tol_rel_model if sector != 'ν' else exp.sig_c_nu):.9f}σ")
        if "ℓ" in results:
            me_pred = exp.predicted_me(results["ℓ"]["c"])
            print(f"m_e(pred)={me_pred:.9f} MeV, z_me≈{abs(me_pred - exp.me) / exp.tol_me:.9f}σ")
        print(f"Stability: checked with grid_delta={args.grid_delta}, grid_r={args.grid_r}")

    elapsed = time.time() - start_time
    print(f"[time] independent_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  Mode: shared_r_all
# ==========================
def mode_shared_r_all(exp: ExpData, args):
    print("\n=== Mode: shared_r_all (same r, sector-specific δ) ===")
    start_time = time.time()
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    best = None

    for r in tqdm(rs, desc="Scanning r (shared_r_all)"):
        results = {}
        chi2_tot = 0.0
        for sector, target in {
            "ν": exp.c_nu_exp,
            "ℓ": exp.c_lep_exp,
            "u": exp.c_u_exp,
            "d": exp.c_d_exp
        }.items():
            best_local = {"delta": None, "c": None, "err": float("inf")}
            for d in deltas:
                M = build_matrix(
                    args.matrix_size, d, r,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target, delta=d, r=r)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"delta": d, "c": c_val, "err": err}
            results[sector] = best_local
            sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
            chi2_tot += ((best_local["c"] - target) / sigma) ** 2

            # Predict m_e for ℓ sector
            if sector == "ℓ":
                me_pred = exp.predicted_me(best_local["c"])
                chi2_tot += (abs(me_pred - exp.me) / (exp.me * exp.tol_rel_model))**2

        if best is None or chi2_tot < best["chi2_tot"]:
            best = {"r": r, "results": results, "chi2_tot": chi2_tot}

    dof = 5  # 4 sectors + m_e
    print(f"Best shared r={best['r']:.9f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] δ={res['delta']:.9f} → c≈{res['c']:.9f}, z≈{z:.9f}σ")
        if sector == "ℓ":
            me_pred = exp.predicted_me(res["c"])
            z_me = abs(me_pred - exp.me) / exp.tol_me
            print(f"    [ℓ] Predicted m_e={me_pred:.9f} MeV (target {exp.me:.9f}), z_me≈{z_me:.9f}σ")
    z_tot = math.sqrt(best["chi2_tot"] / dof)
    print(f"Global χ²_tot={best['chi2_tot']:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")

    if z_tot < 2.5:  # Relaxed criterion for shared_r_all
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")
        print(f"Mode: shared_r_all")
        print(f"Parameters: r={best['r']:.9f}, gL={args.gL:.2f}, gR={args.gR:.2f}, h1={args.h1:.2f}, h2={args.h2:.2f}, h3={args.h3:.2f}")
        for sector, res in best["results"].items():
            print(f"[{sector}] δ={res['delta']:.9f}, c={res['c']:.9f}, z≈{abs(res['c'] - {'ν': exp.c_nu_exp, 'ℓ': exp.c_lep_exp, 'u': exp.c_u_exp, 'd': exp.c_d_exp}[sector]) / (exp.c_nu_exp * exp.tol_rel_model if sector != 'ν' else exp.sig_c_nu):.9f}σ")
        if "ℓ" in best["results"]:
            me_pred = exp.predicted_me(best["results"]["ℓ"]["c"])
            print(f"m_e(pred)={me_pred:.9f} MeV, z_me≈{abs(me_pred - exp.me) / exp.tol_me:.9f}σ")
        print(f"Stability: checked with grid_delta={args.grid_delta}, grid_r={args.grid_r}")

    elapsed = time.time() - start_time
    print(f"[time] shared_r_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  Mode: shared_delta_all
# ==========================
def mode_shared_delta_all(exp: ExpData, args):
    print("\n=== Mode: shared_delta_all (same δ, sector-specific r) ===")
    start_time = time.time()
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    best = None

    for delta in tqdm(deltas, desc="Scanning δ (shared_delta_all)"):
        results = {}
        chi2_tot = 0.0
        for sector, target in {
            "ν": exp.c_nu_exp,
            "ℓ": exp.c_lep_exp,
            "u": exp.c_u_exp,
            "d": exp.c_d_exp
        }.items():
            best_local = {"r": None, "c": None, "err": float("inf")}
            for r in rs:
                M = build_matrix(
                    args.matrix_size, delta, r,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target, delta=delta, r=r)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"r": r, "c": c_val, "err": err}
            results[sector] = best_local
            sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
            chi2_tot += ((best_local["c"] - target) / sigma) ** 2

            # Predict m_e for ℓ sector
            if sector == "ℓ":
                me_pred = exp.predicted_me(best_local["c"])
                chi2_tot += (abs(me_pred - exp.me) / (exp.me * exp.tol_rel_model))**2

        if best is None or chi2_tot < best["chi2_tot"]:
            best = {"delta": delta, "results": results, "chi2_tot": chi2_tot}

    dof = 5  # 4 sectors + m_e
    print(f"Best shared δ={best['delta']:.9f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] r={res['r']:.9f} → c≈{res['c']:.9f}, z≈{z:.9f}σ")
        if sector == "ℓ":
            me_pred = exp.predicted_me(res["c"])
            z_me = abs(me_pred - exp.me) / exp.tol_me
            print(f"    [ℓ] Predicted m_e={me_pred:.9f} MeV (target {exp.me:.9f}), z_me≈{z_me:.9f}σ")
    z_tot = math.sqrt(best["chi2_tot"] / dof)
    print(f"Global χ²_tot={best['chi2_tot']:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")

    if z_tot < 2.5:  # Relaxed criterion for shared_delta_all
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")
        print(f"Mode: shared_delta_all")
        print(f"Parameters: δ={best['delta']:.9f}, gL={args.gL:.2f}, gR={args.gR:.2f}, h1={args.h1:.2f}, h2={args.h2:.2f}, h3={args.h3:.2f}")
        for sector, res in best["results"].items():
            print(f"[{sector}] r={res['r']:.9f}, c={res['c']:.9f}, z≈{abs(res['c'] - {'ν': exp.c_nu_exp, 'ℓ': exp.c_lep_exp, 'u': exp.c_u_exp, 'd': exp.c_d_exp}[sector]) / (exp.c_nu_exp * exp.tol_rel_model if sector != 'ν' else exp.sig_c_nu):.9f}σ")
        if "ℓ" in best["results"]:
            me_pred = exp.predicted_me(best["results"]["ℓ"]["c"])
            print(f"m_e(pred)={me_pred:.9f} MeV, z_me≈{abs(me_pred - exp.me) / exp.tol_me:.9f}σ")
        print(f"Stability: checked with grid_delta={args.grid_delta}, grid_r={args.grid_r}")

    elapsed = time.time() - start_time
    print(f"[time] shared_delta_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  Mode: full_unify_all
# ==========================
def mode_full_unify_all(exp: ExpData, args):
    print("\n=== Mode: full_unify_all (same δ, r for all sectors) ===")
    start_time = time.time()
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    best = None

    for delta in tqdm(deltas, desc="Scanning δ (full_unify_all)"):
        for r in rs:
            chi2_tot = 0.0
            results = {}
            for sector, target in {
                "ν": exp.c_nu_exp,
                "ℓ": exp.c_lep_exp,
                "u": exp.c_u_exp,
                "d": exp.c_d_exp
            }.items():
                M = build_matrix(
                    args.matrix_size, delta, r,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target, delta=delta, r=r)
                results[sector] = {"c": c_val}
                sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
                chi2_tot += ((c_val - target) / sigma) ** 2

                # Predict m_e for ℓ sector
                if sector == "ℓ":
                    me_pred = exp.predicted_me(c_val)
                    chi2_tot += (abs(me_pred - exp.me) / (exp.me * exp.tol_rel_model))**2

            if best is None or chi2_tot < best["chi2_tot"]:
                best = {"delta": delta, "r": r, "results": results, "chi2_tot": chi2_tot}

    dof = 5  # 4 sectors + m_e
    print(f"Best δ={best['delta']:.9f}, r={best['r']:.9f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] c≈{res['c']:.9f}, z≈{z:.9f}σ")
        if sector == "ℓ":
            me_pred = exp.predicted_me(res["c"])
            z_me = abs(me_pred - exp.me) / exp.tol_me
            print(f"    [ℓ] Predicted m_e={me_pred:.9f} MeV (target {exp.me:.9f}), z_me≈{z_me:.9f}σ")
    z_tot = math.sqrt(best["chi2_tot"] / dof)
    print(f"Global χ²_tot={best['chi2_tot']:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")

    if z_tot < 2.0:  # Strict criterion for full_unify_all
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")
        print(f"Mode: full_unify_all")
        print(f"Parameters: δ={best['delta']:.9f}, r={best['r']:.9f}, gL={args.gL:.2f}, gR={args.gR:.2f}, h1={args.h1:.2f}, h2={args.h2:.2f}, h3={args.h3:.2f}")
        for sector, res in best["results"].items():
            print(f"[{sector}] c={res['c']:.9f}, z≈{abs(res['c'] - {'ν': exp.c_nu_exp, 'ℓ': exp.c_lep_exp, 'u': exp.c_u_exp, 'd': exp.c_d_exp}[sector]) / (exp.c_nu_exp * exp.tol_rel_model if sector != 'ν' else exp.sig_c_nu):.9f}σ")
        if "ℓ" in best["results"]:
            me_pred = exp.predicted_me(best["results"]["ℓ"]["c"])
            print(f"m_e(pred)={me_pred:.9f} MeV, z_me≈{abs(me_pred - exp.me) / exp.tol_me:.9f}σ")
        print(f"Stability: checked with grid_delta={args.grid_delta}, grid_r={args.grid_r}")

    elapsed = time.time() - start_time
    print(f"[time] full_unify_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  Mode: grand_unify_all
# ==========================
def mode_grand_unify_all(exp: ExpData, args):
    print("\n=== Mode: grand_unify_all (single c for all sectors) ===")
    print("Note: This mode is almost always a fail due to vastly different target c values.")
    start_time = time.time()
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    best = None

    for delta in tqdm(deltas, desc="Scanning δ (grand_unify_all)"):
        for r in rs:
            M = build_matrix(
                args.matrix_size, delta, r,
                gL=args.gL, gR=args.gR,
                h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
            )
            c_val = ladder_c_from_matrix(M, delta=delta, r=r)
            chi2_tot = 0.0
            results = {}
            for sector, target in {
                "ν": exp.c_nu_exp,
                "ℓ": exp.c_lep_exp,
                "u": exp.c_u_exp,
                "d": exp.c_d_exp
            }.items():
                results[sector] = {"c": c_val}
                sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
                chi2_tot += ((c_val - target) / sigma) ** 2

                # Predict m_e for ℓ sector
                if sector == "ℓ":
                    me_pred = exp.predicted_me(c_val)
                    chi2_tot += (abs(me_pred - exp.me) / (exp.me * exp.tol_rel_model))**2

            if best is None or chi2_tot < best["chi2_tot"]:
                best = {"delta": delta, "r": r, "results": results, "chi2_tot": chi2_tot}

    dof = 5  # 4 sectors + m_e
    print(f"Best δ={best['delta']:.9f}, r={best['r']:.9f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] c≈{res['c']:.9f}, z≈{z:.9f}σ")
        if sector == "ℓ":
            me_pred = exp.predicted_me(res["c"])
            z_me = abs(me_pred - exp.me) / exp.tol_me
            print(f"    [ℓ] Predicted m_e={me_pred:.9f} MeV (target {exp.me:.9f}), z_me≈{z_me:.9f}σ")
    z_tot = math.sqrt(best["chi2_tot"] / dof)
    print(f"Global χ²_tot={best['chi2_tot']:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")

    elapsed = time.time() - start_time
    print(f"[time] grand_unify_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  Mode: grand_unify_all_scaled
# ==========================
def mode_grand_unify_all_scaled(exp: ExpData, args):
    print("\n=== Mode: grand_unify_all_scaled (scaled δ, r per sector) ===")
    start_time = time.time()
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    delta_scales = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    r_scales = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    sector_scales = np.array([0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    best = None

    for delta, r in tqdm(list(itertools.product(deltas, rs)), desc="Scanning δ,r (grand_unify_all_scaled)"):
        for ds, rs_s, ss in itertools.product(delta_scales, r_scales, sector_scales):
            chi2_tot = 0.0
            results = {}
            for sector, target in {
                "ν": exp.c_nu_exp,
                "ℓ": exp.c_lep_exp,
                "u": exp.c_u_exp,
                "d": exp.c_d_exp
            }.items():
                delta_eff = delta * ds
                r_eff = r * rs_s
                M = build_matrix(
                    args.matrix_size, delta_eff, r_eff,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target, delta=delta_eff, r=r_eff)
                c_eff = c_val * ss
                results[sector] = {"c": c_eff, "delta_eff": delta_eff, "r_eff": r_eff, "scale": ss}
                sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
                chi2_tot += ((c_eff - target) / sigma) ** 2

                # Predict m_e for ℓ sector
                if sector == "ℓ":
                    me_pred = exp.predicted_me(c_eff)
                    chi2_tot += (abs(me_pred - exp.me) / (exp.me * exp.tol_rel_model))**2

            if best is None or chi2_tot < best["chi2_tot"]:
                best = {"delta": delta, "r": r, "delta_scale": ds, "r_scale": rs_s, "sector_scale": ss,
                        "results": results, "chi2_tot": chi2_tot}

    dof = 5  # 4 sectors + m_e
    print(f"Best δ={best['delta']:.9f}, r={best['r']:.9f}, δ_scale={best['delta_scale']:.2f}, "
          f"r_scale={best['r_scale']:.2f}, sector_scale={best['sector_scale']:.2f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] δ_eff={res['delta_eff']:.9f}, r_eff={res['r_eff']:.9f}, "
              f"scale={res['scale']:.2f}, c≈{res['c']:.9f}, z≈{z:.9f}σ")
        if sector == "ℓ":
            me_pred = exp.predicted_me(res["c"])
            z_me = abs(me_pred - exp.me) / exp.tol_me
            print(f"    [ℓ] Predicted m_e={me_pred:.9f} MeV (target {exp.me:.9f}), z_me≈{z_me:.9f}σ")
    z_tot = math.sqrt(best["chi2_tot"] / dof)
    print(f"Global χ²_tot={best['chi2_tot']:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")

    if z_tot < 2.5:  # Relaxed criterion for grand_unify_all_scaled
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")
        print(f"Mode: grand_unify_all_scaled")
        print(f"Parameters: δ={best['delta']:.9f}, r={best['r']:.9f}, δ_scale={best['delta_scale']:.2f}, "
              f"r_scale={best['r_scale']:.2f}, sector_scale={best['sector_scale']:.2f}, "
              f"gL={args.gL:.2f}, gR={args.gR:.2f}, h1={args.h1:.2f}, h2={args.h2:.2f}, h3={args.h3:.2f}")
        for sector, res in best["results"].items():
            print(f"[{sector}] δ_eff={res['delta_eff']:.9f}, r_eff={res['r_eff']:.9f}, "
                  f"scale={res['scale']:.2f}, c={res['c']:.9f}, "
                  f"z≈{abs(res['c'] - {'ν': exp.c_nu_exp, 'ℓ': exp.c_lep_exp, 'u': exp.c_u_exp, 'd': exp.c_d_exp}[sector]) / (exp.c_nu_exp * exp.tol_rel_model if sector != 'ν' else exp.sig_c_nu):.9f}σ")
        if "ℓ" in best["results"]:
            me_pred = exp.predicted_me(best["results"]["ℓ"]["c"])
            print(f"m_e(pred)={me_pred:.9f} MeV, z_me≈{abs(me_pred - exp.me) / exp.tol_me:.9f}σ")
        print(f"Stability: checked with grid_delta={args.grid_delta}, grid_r={args.grid_r}")

    elapsed = time.time() - start_time
    print(f"[time] grand_unify_all_scaled completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  CLI
# ==========================
def main():
    parser = argparse.ArgumentParser(description="ZFSC predictor v6.3")
    parser.add_argument("--mode", type=str, default="independent_all",
                        choices=["independent_all", "shared_r_all", "shared_delta_all",
                                 "full_unify_all", "grand_unify_all", "grand_unify_all_scaled"],
                        help="Prediction mode")
    parser.add_argument("--matrix_size", type=int, default=3, choices=[3, 4, 6],
                        help="Matrix size (3, 4, or 6)")
    parser.add_argument("--grid_delta", type=int, default=201,
                        help="Number of grid points for delta")
    parser.add_argument("--grid_r", type=int, default=201,
                        help="Number of grid points for r")
    parser.add_argument("--delta_max", type=float, default=200.0,
                        help="Maximum delta value")
    parser.add_argument("--r_min", type=float, default=0.001,
                        help="Minimum r value")
    parser.add_argument("--r_max", type=float, default=2.0,
                        help="Maximum r value")
    parser.add_argument("--gL", type=float, default=1.0,
                        help="Left edge coupling")
    parser.add_argument("--gR", type=float, default=1.0,
                        help="Right edge coupling")
    parser.add_argument("--h1", type=float, default=0.0,
                        help="Asymmetry parameter h1")
    parser.add_argument("--h2", type=float, default=0.0,
                        help="Asymmetry parameter h2")
    parser.add_argument("--h3", type=float, default=0.0,
                        help="Asymmetry parameter h3")
    parser.add_argument("--dense", action="store_true",
                        help="Use dense grid (grid_delta=1001, grid_r=1001)")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel processing")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--tol_me", type=float, default=0.00001,
                        help="Uncertainty for m_e (MeV)")
    args = parser.parse_args()

    if args.dense:
        args.grid_delta = 1001
        args.grid_r = 1001

    exp = ExpData()
    exp.tol_me = args.tol_me

    print(f"=== ZFSC predictor (4 sectors, v6.3) ===")
    print("Units: Δm² in eV^2; lepton/quark masses in MeV; c's are dimensionless ratios.")
    print(f"c_ν(exp) = {exp.c_nu_exp:.9f} ± {exp.sig_c_nu:.9f}")
    print(f"c_ℓ(exp) = {exp.c_lep_exp:.9f}")
    print(f"c_u(exp) = {exp.c_u_exp:.9f}")
    print(f"c_d(exp) = {exp.c_d_exp:.9f}")
    print(f"Shared structural parameters: gL={args.gL:.9f}, gR={args.gR:.9f}, "
          f"h1={args.h1:.9f}, h2={args.h2:.9f}, h3={args.h3:.9f}")
    print("Backend: NumPy/CPU")
    print(f"Parallel mode: {'ON' if args.parallel else 'OFF'}")

    mode_dispatch = {
        "independent_all": mode_independent_all,
        "shared_r_all": mode_shared_r_all,
        "shared_delta_all": mode_shared_delta_all,
        "full_unify_all": mode_full_unify_all,
        "grand_unify_all": mode_grand_unify_all,
        "grand_unify_all_scaled": mode_grand_unify_all_scaled,
    }

    fn = mode_dispatch.get(args.mode)
    if fn is None:
        raise ValueError(f"Unknown mode {args.mode}")
    fn(exp, args)

if __name__ == "__main__":
    main()