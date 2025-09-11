#!/usr/bin/env python3
# zfsc_predictor.py v6.2 (hybrid)
# Универсальный спектральный предсказатель (ZFSC):
# Поддержка 3×3, 4×4 и 6×6 матриц.
# Режимы:
#   - independent_all
#   - shared_r_all
#   - shared_delta_all
#   - full_unify_all
#   - grand_unify_all
#   - grand_unify_all_scaled
# Использует математику v6.0 (точные формулы) + CLI из v6.1.

import numpy as np
import math
import argparse
import time
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import itertools


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

    tol_rel_model: float = 0.01  # 1% model tolerance

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


# ==========================
#  Ladder ratio computation
# ==========================
def ladder_c_from_matrix(M: np.ndarray, target_c: float = None) -> float:
    """
    Compute ladder ratio c = (λ_max-λ_min)/(λ_mid-λ_min).
    For 3×3: trivial.
    For 4×4 or 6×6: choose the best triple of eigenvalues
    that reproduces target_c (if provided), else choose first 3.
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
        # перебираем все триплеты собственных значений
        for triple in itertools.combinations(w, 3):
            lam_min, lam_mid, lam_max = sorted(triple)
            eps = lam_mid - lam_min
            if eps <= 0:
                continue
            c = (lam_max - lam_min) / eps
            if target_c is None:
                return c  # берём первый допустимый
            err = abs(c - target_c)
            if err < best_err:
                best_err, best_c = err, c
        return best_c if best_c is not None else np.inf

# ==========================
#  Mode: independent_all
# ==========================
def mode_independent_all(exp: ExpData, args):
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
        print(f"[{sector}] best (δ={best_local['delta']:.9f}, r={best_local['r']:.9f}) "
              f"→ c={best_local['c']:.9f} (target {target:.9f}), z≈{z:.9f}σ")

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


# ==========================
#  Mode: shared_r_all
# ==========================
def mode_shared_r_all(exp: ExpData, args):
    print("\n=== Mode: shared_r_all (same r, sector-specific δ) ===")
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


# ==========================
#  Mode: shared_delta_all
# ==========================
def mode_shared_delta_all(exp: ExpData, args):
    print("\n=== Mode: shared_delta_all (same δ, sector-specific r) ===")
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


# ==========================
#  Mode: full_unify_all
# ==========================
def mode_full_unify_all(exp: ExpData, args):
    print("\n=== Mode: full_unify_all (same δ and r for all sectors) ===")
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

# ==========================
#  Mode: grand_unify_all
# ==========================
def mode_grand_unify_all(exp: ExpData, args):
    """
    Строгая унификация:
      - Одни и те же δ, r, gL, gR, h1, h2, h3 для всех секторов.
      - Один и тот же c для ν, ℓ, u, d.
      - Почти всегда провал, но полезный стресс-тест.
    """
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs     = np.geomspace(args.r_min, args.r_max, args.grid_r)

    best = None
    start_time = time.time()

    for d in tqdm(deltas, desc="Scanning δ (grand_unify_all)"):
        for r in rs:
            M = build_matrix(
                args.matrix_size, d, r,
                gL=args.gL, gR=args.gR,
                h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
            )
            c_val = ladder_c_from_matrix(M)
            chi2_tot = 0.0
            sector_results = {}

            for sector, target in {
                "ν": exp.c_nu_exp,
                "ℓ": exp.c_lep_exp,
                "u": exp.c_u_exp,
                "d": exp.c_d_exp
            }.items():
                sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
                res = target - c_val
                z = abs(res) / sigma
                chi2 = (res / sigma) ** 2
                chi2_tot += chi2
                sector_results[sector] = (c_val, target, z)

            avg_z = float(np.mean([z for (_, _, z) in sector_results.values()]))

            if best is None or chi2_tot < best["chi2_tot"]:
                best = {
                    "delta": d, "r": r,
                    "gL": args.gL, "gR": args.gR,
                    "h1": args.h1, "h2": args.h2, "h3": args.h3,
                    "chi2_tot": chi2_tot,
                    "avg_z": avg_z,
                    "sector_results": sector_results,
                }

    elapsed = time.time() - start_time

    print("\n=== Mode: grand_unify_all (STRICT) ===")
    print("Note: одно и то же c для всех секторов → совпадение маловероятно.\n")

    b = best
    print(f"Best (δ={b['delta']:.9f}, r={b['r']:.9f}, "
          f"gL={b['gL']:.9f}, gR={b['gR']:.9f}, "
          f"h1={b['h1']:.9f}, h2={b['h2']:.9f}, h3={b['h3']:.9f})")
    for sector, (c_val, target, z) in b["sector_results"].items():
        print(f"  [{sector}] c≈{c_val:.9f} vs target {target:.9f}, z≈{z:.9f}σ")
    print(f"Global avg z≈{b['avg_z']:.9f}σ")
    print(f"Global χ²_tot={b['chi2_tot']:.6e}")
    print(f"[time] grand_unify_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")


# ==========================
#  Mode: grand_unify_all_scaled
# ==========================
def mode_grand_unify_all_scaled(exp: ExpData, args):
    """
    Унификация с масштабированием:
      - Общие δ, r, gL, gR, h1, h2, h3.
      - У каждого сектора есть коэффициенты: delta_scales, r_scales, sector_scales.
    """
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs     = np.geomspace(args.r_min, args.r_max, args.grid_r)

    best = None
    start_time = time.time()

    for d in tqdm(deltas, desc="Scanning δ (grand_unify_all_scaled)"):
        for r in rs:
            chi2_tot = 0.0
            results = {}
            for sector, target, dscale, rscale, sscale in zip(
                ["ν", "ℓ", "u", "d"],
                [exp.c_nu_exp, exp.c_lep_exp, exp.c_u_exp, exp.c_d_exp],
                args.delta_scales, args.r_scales, args.sector_scales
            ):
                delta_eff = d * dscale
                r_eff = r * rscale
                M = build_matrix(
                    args.matrix_size, delta_eff, r_eff,
                    gL=args.gL, gR=args.gR,
                    h_params=[args.h1, args.h2, args.h3] if args.matrix_size > 3 else None
                )
                c_val = ladder_c_from_matrix(M, target_c=target)
                c_val *= sscale

                sigma = target * 0.01 if sector != "ν" else exp.sig_c_nu
                z = (c_val - target) / sigma
                chi2_tot += z * z
                results[sector] = (delta_eff, r_eff, c_val, target, z)

            avg_z = np.sqrt(chi2_tot / 4.0)

            if best is None or avg_z < best['avg_z']:
                best = dict(
                    delta=d, r=r,
                    gL=args.gL, gR=args.gR,
                    h1=args.h1, h2=args.h2, h3=args.h3,
                    results=results, avg_z=avg_z, chi2_tot=chi2_tot
                )

    elapsed = time.time() - start_time

    print("\n=== Mode: grand_unify_all_scaled (with sector scales) ===")
    print(f"Best (δ={best['delta']:.9f}, r={best['r']:.9f}, "
          f"gL={best['gL']:.9f}, gR={best['gR']:.9f}, "
          f"h1={best['h1']:.9f}, h2={best['h2']:.9f}, h3={best['h3']:.9f})")

    for sector, (delta_eff, r_eff, c_val, target, z) in best['results'].items():
        print(f"  [{sector}] δ_eff={delta_eff:.9f}, r_eff={r_eff:.9f} "
              f"→ c≈{c_val:.9f} vs target {target:.9f}, z≈{z:.9f}σ")

    print(f"Global avg z≈{best['avg_z']:.9f}σ")
    print(f"Global χ²_tot={best['chi2_tot']:.6e}")
    if best['avg_z'] < 2.0:
        print("УРА!!! Возможно, это прорыв и доказательство теории!!!")
    print(f"[time] grand_unify_all_scaled completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")

# ==========================
#  CLI
# ==========================
def main():
    parser = argparse.ArgumentParser(description="ZFSC predictor v6.2 (hybrid: math v6.0 + CLI v6.1)")

    parser.add_argument("--mode", type=str, required=True,
                        choices=[
                            "independent_all",
                            "shared_r_all",
                            "shared_delta_all",
                            "full_unify_all",
                            "grand_unify_all",
                            "grand_unify_all_scaled",
                        ])
    parser.add_argument("--matrix_size", type=int, default=3)
    parser.add_argument("--grid_delta", type=int, default=201)
    parser.add_argument("--grid_r", type=int, default=201)
    parser.add_argument("--delta_max", type=float, default=200.0)
    parser.add_argument("--r_min", type=float, default=0.001)
    parser.add_argument("--r_max", type=float, default=2.0)
    parser.add_argument("--gL", type=float, default=1.0)
    parser.add_argument("--gR", type=float, default=1.0)
    parser.add_argument("--h1", type=float, default=0.0)
    parser.add_argument("--h2", type=float, default=0.0)
    parser.add_argument("--h3", type=float, default=0.0)

    parser.add_argument("--delta_scales", nargs=4, type=float,
                        default=[1.0, 1.0, 1.0, 1.0],
                        help="Multiplicative delta scale factors for ν, ℓ, u, d")
    parser.add_argument("--r_scales", nargs=4, type=float,
                        default=[1.0, 1.0, 1.0, 1.0],
                        help="Multiplicative r scale factors for ν, ℓ, u, d")
    parser.add_argument("--sector_scales", nargs=4, type=float,
                        default=[1.0, 1.0, 1.0, 1.0],
                        help="Multiplicative output scales for ν, ℓ, u, d")

    parser.add_argument("--dense", action="store_true",
                        help="show tqdm progress bars densely")
    parser.add_argument("--parallel", action="store_true",
                        help="(reserved) enable parallel execution")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of workers if parallel enabled")

    args = parser.parse_args()
    exp = ExpData()

    print(f"=== ZFSC predictor (4 sectors, v6.2) ===")
    print("Units: Δm² in eV^2; lepton/quark masses in MeV; c's are dimensionless ratios.")
    print(f"c_ν(exp) = {exp.c_nu_exp:.9f} ± {exp.sig_c_nu:.9f}")
    print(f"c_ℓ(exp) = {exp.c_lep_exp:.9f}")
    print(f"c_u(exp) = {exp.c_u_exp:.9f}")
    print(f"c_d(exp) = {exp.c_d_exp:.9f}")
    print(f"Shared structural parameters: gL={args.gL:.9f}, gR={args.gR:.9f}, "
          f"h1={args.h1:.9f}, h2={args.h2:.9f}, h3={args.h3:.9f}")
    print("Backend: NumPy/CPU")
    print("Parallel mode: " + ("ON" if args.parallel else "OFF"))

    # --- dispatch by mode ---
    mode_dispatch = {
        "independent_all":        mode_independent_all,
        "shared_r_all":           mode_shared_r_all,
        "shared_delta_all":       mode_shared_delta_all,
        "full_unify_all":         mode_full_unify_all,
        "grand_unify_all":        mode_grand_unify_all,
        "grand_unify_all_scaled": mode_grand_unify_all_scaled,
    }

    fn = mode_dispatch.get(args.mode)
    if fn is None:
        raise ValueError(f"Unknown mode {args.mode}")
    fn(exp, args)


if __name__ == "__main__":
    main()
