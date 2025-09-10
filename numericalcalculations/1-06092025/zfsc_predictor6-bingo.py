#!/usr/bin/env python3
# zfsc_predictor.py v6.0
# Универсальный спектральный предсказатель (ZFSC):
# Поддержка 3×3, 4×4 и 6×6 матриц.
# Выбор активного триплета уровней из спектра.
# Встроенный прогрессбар (tqdm) и контрольные метки времени.

import numpy as np
import math
import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List
from tqdm import tqdm
import itertools

# ----------------------------
# Utilities
# ----------------------------
def wilson_hilferty_z(chi2: float, k: int) -> float:
    if k <= 0:
        return float('nan')
    t = (chi2 / k)**(1.0/3.0)
    mu = 1.0 - 2.0/(9.0*k)
    sigma = (2.0/(9.0*k))**0.5
    return abs((t - mu) / sigma)

def chi2_from_residuals(res: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum((res / sigma)**2))

def timestamp() -> str:
    return time.strftime("[%H:%M:%S]")

# ----------------------------
# Experimental inputs
# ----------------------------
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

    tol_rel_model: float = 0.01  # 1% model tolerance

    @property
    def c_nu_exp(self) -> float:
        return self.dm31_NO / self.dm21

    @property
    def sig_c_nu(self) -> float:
        c = self.c_nu_exp
        return c * math.sqrt((self.sig_dm31_NO/self.dm31_NO)**2 + (self.sig_dm21/self.dm21)**2)

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

# ----------------------------
# Matrix builders
# ----------------------------
def build_matrix(size: int, delta: float, r: float,
                 gL: float = 1.0, gR: float = 1.0,
                 h: float = 0.0,
                 h_params: List[float] = None) -> np.ndarray:
    """
    Build symmetric band matrix of size 3, 4, or 6.
    Parameters:
        size : 3, 4, or 6
        delta : central diagonal shift
        r     : nearest-neighbor coupling
        gL,gR : edge couplings
        h,h_params : asymmetries
    """
    if size == 3:
        # classic ladder
        return np.array([[0.0, gL, 0.0],
                         [gL, delta, r],
                         [0.0, r, 0.0]], dtype=float)

    elif size == 4:
        # add one extra state, symmetric
        h1 = h_params[0] if h_params else h
        return np.array([[0.0, gL, 0.0, 0.0],
                         [gL, delta, r, h1],
                         [0.0, r, 0.0, gR],
                         [0.0, h1, gR, 0.0]], dtype=float)

    elif size == 6:
        # block-tridiagonal with asymmetries
        h1, h2, h3 = (h_params if h_params else [h, h, h])
        M = np.zeros((6,6), dtype=float)
        # couplings
        for i in range(5):
            M[i,i+1] = r
            M[i+1,i] = r
        # edges
        M[0,1] = gL; M[1,0] = gL
        M[4,5] = gR; M[5,4] = gR
        # central shift
        np.fill_diagonal(M, 0.0)
        M[2,2] = delta
        M[3,3] = delta
        # asymmetries
        M[1,4] = h1; M[4,1] = h1
        M[2,5] = h2; M[5,2] = h2
        M[0,3] = h3; M[3,0] = h3
        return M

    else:
        raise ValueError("Matrix size must be 3, 4, or 6.")

# ----------------------------
# Ladder ratio computation
# ----------------------------
def ladder_c_from_matrix(M: np.ndarray, target_c: float = None) -> float:
    """
    Compute ladder ratio c = (λ_max-λ_min)/(λ_mid-λ_min)
    For 3×3: trivial.
    For 4×4 or 6×6: choose the best triple of eigenvalues
    that reproduces target_c (if provided), else choose first 3.
    """
    w = np.linalg.eigvalsh(M)
    w.sort()

    if len(w) == 3:
        lam_min, lam_mid, lam_max = w
        eps = lam_mid - lam_min
        if eps <= 0: return np.inf
        return (lam_max - lam_min) / eps

    else:
        best_c = None
        best_err = float('inf')
        # pick all triplets of eigenvalues
        for triple in itertools.combinations(w, 3):
            lam_min, lam_mid, lam_max = sorted(triple)
            eps = lam_mid - lam_min
            if eps <= 0: continue
            c = (lam_max - lam_min) / eps
            if target_c is None:
                return c  # just first triple
            err = abs(c - target_c)
            if err < best_err:
                best_err, best_c = err, c
        return best_c if best_c is not None else np.inf

# ----------------------------
# Search helpers with tqdm
# ----------------------------
def grid_scan_deltas_r(deltas, rs, size, exp_targets,
                       gL=1.0, gR=1.0, h_params=None,
                       mode="independent_all", verbose=False):
    """
    Scan grid of (δ,r) and evaluate ladder ratios.
    exp_targets: dict of target c for sectors {nu, l, u, d}
    Returns: dict with best fit.
    """
    best = {"chi2": float('inf')}
    total_steps = len(deltas) * len(rs)
    start_time = time.time()

    with tqdm(total=total_steps, desc=f"[scan {mode}]", ncols=80) as pbar:
        for i, d in enumerate(deltas):
            for j, r in enumerate(rs):
                chi2_tot = 0.0
                cs = {}
                for sector, target in exp_targets.items():
                    M = build_matrix(size, d, r, gL=gL, gR=gR, h_params=h_params)
                    c_val = ladder_c_from_matrix(M, target_c=target)
                    cs[sector] = c_val
                    if target > 0:
                        chi2_tot += ((c_val - target)/target)**2
                if chi2_tot < best["chi2"]:
                    best = {"delta": d, "r": r, "cs": cs, "chi2": chi2_tot}
                pbar.update(1)
            # контрольное время раз в 10% шагов
            if i % max(1,len(deltas)//10) == 0:
                elapsed = time.time()-start_time
                print(f"[time] δ={d:.3f} done, elapsed {elapsed:.1f} sec")

    best["time"] = time.time()-start_time
    return best


def c_from_B(delta, r, gL=1.0, gR=1.0,
             h1=0.0, h2=0.0, h3=0.0, size=3):
    """
    Построение матрицы B (size x size) и вычисление
    спектрального коэффициента c.
    """
    # создаём базовую матрицу
    B = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                B[i, j] = delta + h1*i + h2*j + h3*(i-j)**2
            else:
                B[i, j] = r * (gL if i < j else gR)

    # собственные значения
    eigvals = np.linalg.eigvalsh(B)
    eigvals = np.sort(eigvals)

    # c = (max - mid) / (mid - min), для size>=3
    if len(eigvals) >= 3:
        lam_min, lam_mid, lam_max = eigvals[0], eigvals[len(eigvals)//2], eigvals[-1]
        c_val = (lam_max - lam_min) / (lam_mid - lam_min)
    else:
        c_val = (eigvals[-1] - eigvals[0]) / (eigvals[1] - eigvals[0])

    return float(c_val)

# ----------------------------
# Modes
# ----------------------------
from tqdm import tqdm
import time

# ----------------------------
# Mode: independent_all
# ----------------------------
def mode_independent_all(exp, args):
    print("\n=== Mode: independent_all ===")
    start_time = time.time()

    results = {}
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        best_local = {"delta": None, "r": None, "c": None, "err": float("inf")}
        deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
        rs = np.geomspace(args.r_min, args.r_max, args.grid_r)

        # прогрессбар по δ
        for d in tqdm(deltas, desc=f"Scanning δ for {sector}", leave=False):
            for r in rs:
                M = build_matrix(
                    args.matrix_size, d, r,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"delta": d, "r": r, "c": c_val, "err": err}

        results[sector] = best_local
        sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
        z = abs(best_local["c"] - target) / sigma
        print(
            f"[{sector}] best (δ={best_local['delta']:.9f}, r={best_local['r']:.9f}) "
            f"→ c={best_local['c']:.9f} (target {target:.9f}), z≈{z:.9f}σ"
        )

    # усреднённый глобальный z
    z_vals = []
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        best = results[sector]
        sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
        z = abs(best["c"] - target) / sigma
        z_vals.append(z)

    z_avg = sum(z_vals) / len(z_vals)
    print(f"Global avg z≈{z_avg:.9f}σ")

    if z_avg < 2.0:
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")

    elapsed = time.time() - start_time
    print(f"[time] independent_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")



from tqdm import tqdm

# ----------------------------
# Mode: shared_r_all
# ----------------------------
def mode_shared_r_all(exp, args):
    print("\n=== Mode: shared_r_all (same r, sector-specific δ; shared g) ===")
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
                c_val = ladder_c_from_matrix(M, target_c=target)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"delta": d, "c": c_val, "err": err}
            results[sector] = best_local
            sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
            chi2_tot += ((best_local["c"] - target) / sigma) ** 2

        if best is None or chi2_tot < best["chi2_tot"]:
            best = {"r": r, "results": results, "chi2_tot": chi2_tot}

    print(f"Best shared r={best['r']:.9f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] δ={res['delta']:.9f} → c≈{res['c']:.9f}, z≈{z:.9f}σ")
    print(f"Global χ²_tot={best['chi2_tot']:.6e}")


# ----------------------------
# Mode: shared_delta_all
# ----------------------------
def mode_shared_delta_all(exp, args):
    print("\n=== Mode: shared_delta_all (same δ, sector-specific r; shared g) ===")
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)

    best = None
    for d in tqdm(deltas, desc="Scanning δ (shared_delta_all)"):
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
                    args.matrix_size, d, r,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"r": r, "c": c_val, "err": err}
            results[sector] = best_local
            sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
            chi2_tot += ((best_local["c"] - target) / sigma) ** 2

        if best is None or chi2_tot < best["chi2_tot"]:
            best = {"delta": d, "results": results, "chi2_tot": chi2_tot}

    print(f"Best shared δ={best['delta']:.9f}")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] r={res['r']:.9f} → c≈{res['c']:.9f}, z≈{z:.9f}σ")
    print(f"Global χ²_tot={best['chi2_tot']:.6e}")


# ----------------------------
# Mode: full_unify_all
# ----------------------------
def mode_full_unify_all(exp, args):
    print("\n=== Mode: full_unify_all (same δ and r for all sectors; shared g) ===")
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)

    best = None
    for d in tqdm(deltas, desc="Scanning δ (full_unify_all)"):
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
                    args.matrix_size, d, r,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target)
                results[sector] = {"c": c_val}
                sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
                chi2_tot += ((c_val - target) / sigma) ** 2

            if best is None or chi2_tot < best["chi2_tot"]:
                best = {"delta": d, "r": r, "results": results, "chi2_tot": chi2_tot}

    print(f"Best (δ={best['delta']:.9f}, r={best['r']:.9f})")
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        res = best["results"][sector]
        sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
        z = abs(res["c"] - target) / sigma
        print(f"  [{sector}] c≈{res['c']:.9f}, z≈{z:.9f}σ")
    print(f"Global χ²_tot={best['chi2_tot']:.6e}")



from tqdm import tqdm
import time

def mode_grand_unify_all(exp: ExpData, args) -> None:
    """
    Grand unified (strict) mode:
      - Same δ, r, gL, gR, h1, h2, h3, size for all sectors (ν, ℓ, u, d).
      - => Один и тот же c для всех четырёх секторов.
      - Это диагностический тест: насколько "одно и то же c" может приблизиться
        к четырём очень разным целям одновременно (обычно — никак).
    """
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs     = np.geomspace(args.r_min, args.r_max, args.grid_r)

    start_time = time.time()
    total = len(deltas) * len(rs)
    pbar  = tqdm(total=total, desc="grand_unify_all scan", mininterval=0.5, leave=False)

    best = None

    for d in deltas:
        for r in rs:
            # один и тот же c для всех секторов
            c_val = c_from_B(
                float(d), float(r),
                gL=args.gL, gR=args.gR,
                h1=args.h1, h2=args.h2, h3=args.h3,
                size=args.matrix_size
            )

            chi2_tot = 0.0
            sector_results = {}

            for sector, target in {
                "ν": exp.c_nu_exp,
                "ℓ": exp.c_lep_exp,
                "u": exp.c_u_exp,
                "d": exp.c_d_exp,
            }.items():
                # σ: 1% модельная для всех, кроме нейтрино (там используем exp.sig_c_nu)
                sigma = abs(target) * 0.01 if sector != "ν" else exp.sig_c_nu
                res   = target - c_val
                z     = abs(res) / sigma
                chi2  = (res / sigma) ** 2
                chi2_tot += chi2
                sector_results[sector] = (c_val, target, z)

            avg_z = float(np.mean([z for (_, _, z) in sector_results.values()]))

            if (best is None) or (chi2_tot < best["chi2_tot"]):
                best = {
                    "delta": float(d), "r": float(r),
                    "gL": args.gL, "gR": args.gR,
                    "h1": args.h1, "h2": args.h2, "h3": args.h3,
                    "chi2_tot": float(chi2_tot),
                    "avg_z": avg_z,
                    "sector_results": sector_results,
                }

            pbar.update(1)

    pbar.close()
    elapsed = time.time() - start_time

    # sanity notice: строгое grand_unify_all даёт один c на всех — это почти всегда провал
    print("\n=== Mode: grand_unify_all (STRICT: one c for all sectors) ===")
    print("Note: this mode enforces the SAME spectrum for ν, ℓ, u, d. "
          "Since their target c differ by orders of magnitude, a perfect fit is mathematically impossible.\n")

    b = best
    print(f"Best (δ={b['delta']:.9f}, r={b['r']:.9f}, "
          f"gL={b['gL']:.9f}, gR={b['gR']:.9f}, "
          f"h1={b['h1']:.9f}, h2={b['h2']:.9f}, h3={b['h3']:.9f})")
    for sector, (c_val, target, z) in b["sector_results"].items():
        print(f"  [{sector}] c≈{c_val:.9f} vs target {target:.9f}, z≈{z:.9f}σ")
    print(f"Global avg z≈{b['avg_z']:.9f}σ")
    print(f"[time] grand_unify_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

    # никакого "УРА" тут не печатаем: строгая унификация — специально жёсткий провальный тест


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["independent_all", "shared_r_all", "shared_delta_all",
                                 "full_unify_all", "grand_unify_all"],
                        help="Calculation mode to run")
    parser.add_argument("--matrix_size", type=int, default=3,
                        help="Matrix size (default: 3)")
    parser.add_argument("--grid_delta", type=int, default=201,
                        help="Grid resolution in δ")
    parser.add_argument("--grid_r", type=int, default=201,
                        help="Grid resolution in r")
    parser.add_argument("--delta_max", type=float, default=200.0,
                        help="Maximum δ to scan")
    parser.add_argument("--r_min", type=float, default=0.001,
                        help="Minimum r to scan")
    parser.add_argument("--r_max", type=float, default=2.0,
                        help="Maximum r to scan")
    parser.add_argument("--gL", type=float, default=1.0,
                        help="Left coupling parameter gL")
    parser.add_argument("--gR", type=float, default=1.0,
                        help="Right coupling parameter gR")
    parser.add_argument("--h1", type=float, default=0.0,
                        help="Deformation parameter h1")
    parser.add_argument("--h2", type=float, default=0.0,
                        help="Deformation parameter h2")
    parser.add_argument("--h3", type=float, default=0.0,
                        help="Deformation parameter h3")
    parser.add_argument("--dense", action="store_true",
                        help="Use dense grid scan")
    parser.add_argument("--parallel", action="store_true",
                        help="Enable parallel execution")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")

    args = parser.parse_args()
    exp = ExpData()

    print(f"=== ZFSC predictor (4 sectors, v6.0) ===")
    print("Units: Δm² in eV^2; lepton/quark masses in MeV; c's are dimensionless ratios.")
    print(f"c_ν(exp) = {exp.c_nu_exp:.9f} ± {exp.sig_c_nu:.9f} (from Δm31²/Δm21²)")
    print(f"c_ℓ(exp) = {exp.c_lep_exp:.9f} (from τ,μ,e)")
    print(f"c_u(exp) = {exp.c_u_exp:.9f} (from t,c,u)")
    print(f"c_d(exp) = {exp.c_d_exp:.9f} (from b,s,d)")
    print(f"Shared structural parameters: gL={args.gL:.9f}, gR={args.gR:.9f}, "
          f"h1={args.h1:.9f}, h2={args.h2:.9f}, h3={args.h3:.9f}")
    print(f"Backend: NumPy/CPU")
    if args.parallel:
        print(f"Parallel: ON (workers={args.workers})")
    else:
        print("Parallel: OFF")

    # --- dispatch by mode ---
    mode_dispatch = {
        "independent_all":  mode_independent_all,
        "shared_r_all":     mode_shared_r_all,
        "shared_delta_all": mode_shared_delta_all,
        "full_unify_all":   mode_full_unify_all,
        "grand_unify_all":  mode_grand_unify_all,
    }

    fn = mode_dispatch.get(args.mode)
    if fn is None:
        raise ValueError(f"Unknown mode {args.mode}")
    fn(exp, args)


if __name__ == "__main__":
    main()
