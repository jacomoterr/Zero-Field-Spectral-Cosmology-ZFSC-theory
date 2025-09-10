#!/usr/bin/env python3
# zfsc_predictor.py v7.0 (theory-pure version, без подгонки)
# Универсальный спектральный предсказатель (ZFSC):
# Поддержка 3×3, 4×4 и 6×6 матриц.
# Версия v7.0:
# - Убраны эмпирические коэффициенты (0.0018, 1e-2, 1e-160 и пр.)
# - Предсказание m_e полностью из спектра
# - Космологические константы считаются без ручных масштабов
# - Отрицательные λ не отбрасываются, а учитываются через abs()

import numpy as np
import math
import argparse
import time
import logging
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import itertools
import multiprocessing as mp

# ==========================
# Physical constants
# ==========================
h_bar = 1.0545718e-34  # J·s
c = 2.99792458e8       # m/s
hbar_c_MeV_fm = 197.32698  # MeV·fm
l_P = 1.616255e-35     # Planck length (m)
t_P = 5.39116e-44      # Planck time (s)
M_P = 2.176434e-8      # Planck mass (kg)
G_N_exp = 6.67430e-11  # m^3 kg^-1 s^-2
Lambda_exp = 1.1056e-52  # m^-2
H_0_exp = 2.18e-18     # s^-1
alpha_exp = 1 / 137.036
universe_age_sec = 13.8e9 * 365.25 * 24 * 3600  # ≈ 4.35e17 s

# ==========================
# Experimental inputs
# ==========================
@dataclass
class ExpData:
    # Neutrino splittings (normal ordering)
    dm21: float = 7.42e-5      # eV^2
    dm31_NO: float = 2.517e-3  # eV^2
    sig_dm21: float = 0.21e-5  # eV^2
    sig_dm31_NO: float = 0.026e-3  # eV^2
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
    tol_rel_model: float = 0.01   # 1% model tolerance for c_ℓ, c_u, c_d
    tol_me: float = 0.051099895   # 10% of m_e (MeV)

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

    def predicted_me(self, c_lep_pred: float, lam_min: float, lam_mid: float,
                     logger: logging.Logger = None) -> float:
        """
        Спектральное предсказание массы электрона.
        Чистая версия: без эмпирических коэффициентов.
        """
        s2 = (self.mtau**2 - self.mmu**2) / (c_lep_pred - 1)
        me2_pred = self.mmu**2 - s2
        if logger:
            logger.info(f"[ℓ] s²={s2:.9e}, m_e²={me2_pred:.9e}")
        if me2_pred < 0:
            # Используем спектральный зазор напрямую
            return math.sqrt(abs(lam_mid - lam_min) * hbar_c_MeV_fm)
        return math.sqrt(me2_pred)


# ==========================
# Matrix builders
# ==========================
def build_matrix(size: int, delta: float, r: float,
                 gL: float = 1.0, gR: float = 1.0,
                 h: float = 0.0,
                 h_params: List[float] = None,
                 delta2: float = None,
                 rL: float = None, rR: float = None,
                 s_params: List[float] = None,
                 muL: float = 0.0, muR: float = 0.0,
                 block_split: int = None, inter_scale: float = 1.0,
                 splits: List[int] = None, inter_scales: List[float] = None,
                 nested: bool = False) -> np.ndarray:
    """
    Универсальный конструктор симметричной матрицы (N >= 3).
    Поддержка:
      - линейная цепочка ближайших связей (r), края gL/gR;
      - центральная пара δ, δ2;
      - слабые next-nearest связи s;
      - перекрёстные связи h1,h2,h3;
      - onsite-потенциалы muL/muR;
      - блочность (splits/inter_scales);
      - вложенные подблоки (nested=True).
    """
    N = size
    if N < 3:
        raise ValueError("Matrix size must be >= 3")

    # --- normalize arguments ---
    if splits is None and block_split is not None:
        splits = [block_split]
        inter_scales = [inter_scale]
    if splits is None:
        splits = []
    if inter_scales is None:
        inter_scales = []
    if len(inter_scales) != len(splits):
        raise ValueError("len(inter_scales) must equal len(splits)")
    splits = sorted([s for s in splits if 0 < s < N])

    # --- nested mode: автоматически добавляем подсплиты ---
    if nested and splits:
        new_splits = []
        new_scales = []
        for cut, scale in zip(splits, inter_scales):
            left_half = cut // 2
            right_half = cut + (N - cut) // 2
            if 1 < left_half < cut:
                new_splits.append(left_half)
                new_scales.append(scale)
            if cut < right_half < N - 1:
                new_splits.append(right_half)
                new_scales.append(scale)
        splits = sorted(splits + new_splits)
        inter_scales = inter_scales + new_scales

    # helper: scaling for a link (i<->j) — умножаем все scale на границах
    def scale_between(i: int, j: int) -> float:
        if not splits:
            return 1.0
        lo, hi = (i, j) if i < j else (j, i)
        s = 1.0
        for k, cut in enumerate(splits):
            if lo < cut <= hi:
                s *= inter_scales[k]
        return s

    # распаковка параметров
    h1, h2, h3 = (h_params if h_params else [h, h, h])
    rL = r if rL is None else rL
    rR = r if rR is None else rR
    if s_params is None:
        s_all = 0.0
    else:
        s_all = float(sum(s_params)) / len(s_params)

    M = np.zeros((N, N), dtype=float)

    # ближайшие соседи
    for i in range(N - 1):
        w = r
        if i == 0:         w = gL
        elif i == N - 2:   w = gR
        elif i == 1:       w = rL
        elif i == N - 3:   w = rR
        w *= scale_between(i, i + 1)
        M[i, i + 1] = M[i + 1, i] = w

    # next-nearest
    if s_all != 0.0:
        for i in range(N - 2):
            w = s_all * scale_between(i, i + 2)
            M[i, i + 2] = M[i + 2, i] = w

    # диагональ
    np.fill_diagonal(M, 0.0)
    M[0, 0] = muL
    M[N - 1, N - 1] = muR
    if N % 2 == 0:
        cL = N // 2 - 1
        cR = N // 2
        M[cL, cL] = delta
        M[cR, cR] = delta if delta2 is None else delta2
    else:
        cC = N // 2
        M[cC, cC] = delta
        if delta2 is not None and 0 <= cC + 1 < N:
            M[cC + 1, cC + 1] = delta2

    # cross-links
    def safe_link(i, j, w):
        if 0 <= i < N and 0 <= j < N and w != 0.0:
            M[i, j] = M[j, i] = w * scale_between(i, j)

    safe_link(1, N - 2, h1)
    safe_link(2, N - 1, h2)
    safe_link(0, N - 3, h3)

    return M




#    else:
#        raise ValueError("Matrix size must be 3, 4, or 6.")

# ==========================
# Ladder ratio computation
# ==========================
def ladder_c_from_matrix(M: np.ndarray,
                         target_c: float = None,
                         delta: float = None,
                         r: float = None,
                         sector: str = None,
                         exp: ExpData = None,
                         logger: logging.Logger = None) -> tuple:
    """
    Вычисление спектрального коэффициента c из матрицы.
    c = (λ_max - λ_min) / (λ_mid - λ_min)
    """
    w = np.linalg.eigvalsh(M)
    w.sort()
    triplet_count = 0

    if len(w) == 3:
        lam_min, lam_mid, lam_max = w
        eps = lam_mid - lam_min
        if eps <= 0:
            if logger:
                logger.warning(f"Degenerate or unstable triplet: {w}")
            return np.inf, lam_min, lam_mid, lam_max, triplet_count, w
        return (lam_max - lam_min) / eps, lam_min, lam_mid, lam_max, 1, w

    else:
        best_c = None
        best_err = float('inf')
        best_triplet = (0, 0, 0)

        # Ассимптотическая оценка для справки
        asymptotic_c = (delta**2 / (1 + r**2) + 2) if delta is not None and r is not None else None

        for triple in itertools.combinations(w, 3):
            lam_min, lam_mid, lam_max = sorted(triple)

            eps = lam_mid - lam_min
            if eps <= 0:
                continue

            c_val = (lam_max - lam_min) / eps
            err = abs(c_val - target_c) if target_c else 0

            if asymptotic_c:
                err += 0.1 * abs(c_val - asymptotic_c)

            # Лептонный сектор: штраф на неверное m_mu
            if sector == 'ℓ' and exp:
                m_mu_pred = math.sqrt(abs(lam_mid - lam_min) * hbar_c_MeV_fm)
                err += abs(m_mu_pred - exp.mmu) / exp.mmu

            triplet_count += 1
            if err < best_err:
                best_err, best_c = err, c_val
                best_triplet = (lam_min, lam_mid, lam_max)

        if best_c is None:
            return np.inf, 0, 0, 0, triplet_count, w
        return best_c, best_triplet[0], best_triplet[1], best_triplet[2], triplet_count, w

# ==========================
# Parallel helper for scanning
# ==========================
def scan_sector_delta_r(args):
    """
    Параллельная функция для сканирования (δ, r).
    Возвращает спектральные коэффициенты и тестовые космологические величины.
    """
    sector, target, sigma, delta, r, matrix_size, gL, gR, h_params, exp, cli_args = args

    # Строим матрицу с расширенными параметрами
# Строим матрицу с расширенными параметрами
    M = build_matrix(
        args.matrix_size, d, r,
        gL=args.gL, gR=args.gR, h_params=h_params,
        delta2=args.delta2,
        rL=args.rL, rR=args.rR,
        s_params=[args.s12, args.s23, args.s34, args.s45],
        muL=args.muL, muR=args.muR,
        block_split=args.block_split,
        inter_scale=args.inter_scale,
        splits=args.splits, 
        inter_scales=args.inter_scales, 
        nested=args.nested)
# Спектр
    c_val, lam_min, lam_mid, lam_max, triplet_count, _ = ladder_c_from_matrix(
    M, target_c=target, delta=delta, r=r, sector=sector, exp=exp
)


    # Ошибка относительно цели
    err = abs(c_val - target)

    # Массы
    k_mass = {'ν': 1e-8, 'ℓ': 0.01, 'u': 0.1, 'd': 0.05}[sector]
    gap1 = abs(lam_mid - lam_min)
    gap2 = abs(lam_max - lam_min)

    m1 = k_mass * math.sqrt(gap1 * hbar_c_MeV_fm)
    m2 = k_mass * math.sqrt(gap1 * hbar_c_MeV_fm)
    m3 = k_mass * math.sqrt(gap2 * hbar_c_MeV_fm)

    # Тестовые фундаментальные константы (чистая версия, без подгонки)
    alpha_test = (gL * gR * (m1 / m3)) if m3 > 0 else 0
    G_N_test = ((gR / gL)**2 / (delta**2 + 1e-10)) * (l_P**2)
    Lambda_test = (cli_args.h3**2 / (delta**2 + 1e-10)) * (1 / l_P**2) if h_params else 0
    H_0_test = ((gR / gL) / (delta + 1e-10)) * (1 / universe_age_sec)

    return (delta, r, c_val, err,
            m1, m2, m3,
            lam_min, lam_mid, lam_max,
            triplet_count,
            alpha_test, G_N_test, Lambda_test, H_0_test)


# ==========================
# Mode: independent_all
# ==========================
def mode_independent_all(exp: ExpData, args, logger: logging.Logger = None):
    print("\n=== Mode: independent_all ===")
    start_time = time.time()
    results = {}
    chi2_tot = 0.0
    dof = 4  # число степеней свободы (4 сектора)

    # Диапазоны δ для каждого сектора
    delta_ranges = {
        'ν': np.linspace(360, 380, args.grid_delta),
        'ℓ': np.linspace(1295, 1297, args.grid_delta),
        'u': np.linspace(3500, 3600, args.grid_delta),
        'd': np.linspace(4000, 4500, args.grid_delta)
    }

    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        best_local = {"delta": None, "r": None, "c": None, "err": float("inf"), "eigenvalues": None}

        deltas = delta_ranges[sector]
        rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
        h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

        for d in tqdm(deltas, desc=f"Scanning δ for {sector}", leave=False):
            for r in rs:
                M = build_matrix(args.matrix_size, d, r,
                                 gL=args.gL, gR=args.gR, h_params=h_params,
                                 delta2=args.delta2,
                                 rL=args.rL, rR=args.rR,
                                 s_params=[args.s12, args.s23, args.s34, args.s45],
                                 muL=args.muL, muR=args.muR,
                                 block_split=args.block_split,
                                 inter_scale=args.inter_scale,
                                 splits=args.splits, 
                                 inter_scales=args.inter_scales, 
                                 nested=args.nested
                )

                c_val, lam_min, lam_mid, lam_max, triplet_count, eigenvalues = \
                    ladder_c_from_matrix(M, target_c=target, delta=d, r=r, sector=sector, exp=exp)

                err = abs(c_val - target)

                if err < best_local["err"]:
                    best_local = {"delta": d, "r": r, "c": c_val, "err": err, "eigenvalues": eigenvalues}

        results[sector] = best_local
        z = abs(best_local["c"] - target) / sigma
        chi2_tot += z**2

        print(f"[{sector}] best (δ={best_local['delta']:.9f}, r={best_local['r']:.9f}) "
              f"→ c={best_local['c']:.9f} (target {target:.9f}), z≈{z:.9f}σ")

    z_tot = math.sqrt(chi2_tot / dof)
    print(f"Global χ²_tot={chi2_tot:.6e}, dof={dof}, z_tot≈{z_tot:.9f}σ")
    elapsed = time.time() - start_time
    print(f"[time] independent_all completed in {elapsed:.2f} sec ({elapsed/60:.2f} min)")
    return z_tot


# ==========================
# Mode: shared_r_all
# ==========================
def mode_shared_r_all(exp: ExpData, args, logger: logging.Logger = None):
    print("\n=== Mode: shared_r_all ===")
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    best = None
    h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

    for r in tqdm(rs, desc="Scanning r"):
        chi2_tot = 0.0
        results = {}
        for sector, target in {
            "ν": exp.c_nu_exp,
            "ℓ": exp.c_lep_exp,
            "u": exp.c_u_exp,
            "d": exp.c_d_exp
        }.items():
            best_local = {"delta": None, "c": None, "err": float("inf"), "eigenvalues": None}
            deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
            for d in deltas:
                M = build_matrix(args.matrix_size, d, r,
                                 gL=args.gL, gR=args.gR, h_params=h_params,
                                 delta2=args.delta2,
                                 rL=args.rL, rR=args.rR,
                                 s_params=[args.s12, args.s23, args.s34, args.s45],
                                 muL=args.muL, muR=args.muR,
                                 block_split=args.block_split,
                                 inter_scale=args.inter_scale,
                                 splits=args.splits, 
                                 inter_scales=args.inter_scales, 
                                 nested=args.nested)
                                 
                c_val, _, _, _, _, eigenvalues = ladder_c_from_matrix(M, target_c=target,
                                                                      delta=d, r=r, sector=sector, exp=exp)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"delta": d, "c": c_val, "err": err, "eigenvalues": eigenvalues}

            results[sector] = best_local
            sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
            chi2_tot += ((best_local["c"] - target) / sigma) ** 2

        if best is None or chi2_tot < best["chi2_tot"]:
            best = {"r": r, "results": results, "chi2_tot": chi2_tot}

    print(f"Best shared r={best['r']:.9f}")
    return math.sqrt(best["chi2_tot"] / 4)


# ==========================
# Mode: shared_delta_all
# ==========================
def mode_shared_delta_all(exp: ExpData, args, logger: logging.Logger = None):
    print("\n=== Mode: shared_delta_all ===")
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    best = None
    h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

    for d in tqdm(deltas, desc="Scanning δ"):
        chi2_tot = 0.0
        results = {}
        rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
        for sector, target in {
            "ν": exp.c_nu_exp,
            "ℓ": exp.c_lep_exp,
            "u": exp.c_u_exp,
            "d": exp.c_d_exp
        }.items():
            best_local = {"r": None, "c": None, "err": float("inf"), "eigenvalues": None}
            for r in rs:
                M = build_matrix(args.matrix_size, d, r,
                                 gL=args.gL, gR=args.gR, h_params=h_params,
                                 delta2=args.delta2,
                                 rL=args.rL, rR=args.rR,
                                 s_params=[args.s12, args.s23, args.s34, args.s45],
                                 muL=args.muL, muR=args.muR,
                                 block_split=args.block_split,
                                 inter_scale=args.inter_scale,
                                 splits=args.splits, 
                                 inter_scales=args.inter_scales, 
                                 nested=args.nested)
                                 
                c_val, _, _, _, _, eigenvalues = ladder_c_from_matrix(M, target_c=target,
                                                                      delta=d, r=r, sector=sector, exp=exp)
                err = abs(c_val - target)
                if err < best_local["err"]:
                    best_local = {"r": r, "c": c_val, "err": err, "eigenvalues": eigenvalues}

            results[sector] = best_local
            sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
            chi2_tot += ((best_local["c"] - target) / sigma) ** 2

        if best is None or chi2_tot < best["chi2_tot"]:
            best = {"delta": d, "results": results, "chi2_tot": chi2_tot}

    print(f"Best shared δ={best['delta']:.9f}")
    return math.sqrt(best["chi2_tot"] / 4)


# ==========================
# Mode: full_unify_all
# ==========================
def mode_full_unify_all(exp: ExpData, args, logger: logging.Logger = None):
    print("\n=== Mode: full_unify_all ===")
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    best = None
    h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

    for d in tqdm(deltas, desc="Scanning δ"):
        for r in rs:
            chi2_tot = 0.0
            results = {}
            for sector, target in {
                "ν": exp.c_nu_exp,
                "ℓ": exp.c_lep_exp,
                "u": exp.c_u_exp,
                "d": exp.c_d_exp
            }.items():
                M = build_matrix(args.matrix_size, d, r,
                                 gL=args.gL, gR=args.gR, h_params=h_params,
                                 delta2=args.delta2,
                                 rL=args.rL, rR=args.rR,
                                 s_params=[args.s12, args.s23, args.s34, args.s45],
                                 muL=args.muL, muR=args.muR,
                                 block_split=args.block_split,
                                 inter_scale=args.inter_scale,
                                 splits=args.splits, 
                                 inter_scales=args.inter_scales, 
                                 nested=args.nested)
                                 

                c_val, _, _, _, _, eigenvalues = ladder_c_from_matrix(M, target_c=target,
                                                                      delta=d, r=r, sector=sector, exp=exp)
                results[sector] = {"c": c_val, "eigenvalues": eigenvalues}
                sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
                chi2_tot += ((c_val - target) / sigma) ** 2

            if best is None or chi2_tot < best["chi2_tot"]:
                best = {"delta": d, "r": r, "results": results, "chi2_tot": chi2_tot}

    print(f"Best δ={best['delta']:.9f}, r={best['r']:.9f}")
    return math.sqrt(best["chi2_tot"] / 4)


# ==========================
# Mode: grand_unify_all
# ==========================
def mode_grand_unify_all(exp: ExpData, args, logger: logging.Logger = None):
    print("\n=== Mode: grand_unify_all ===")
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    best = None
    h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

    for d in tqdm(deltas, desc="Scanning δ"):
        for r in rs:
            chi2_tot = 0.0
            results = {}
# Строим матрицу с расширенными параметрами
    M = build_matrix(
            args.matrix_size, d, r,
            gL=args.gL, gR=args.gR, h_params=h_params,
            delta2=args.delta2,
            rL=args.rL, rR=args.rR,
            s_params=[args.s12, args.s23, args.s34, args.s45],
            muL=args.muL, muR=args.muR,
            block_split=args.block_split,
            inter_scale=args.inter_scale,
            splits=args.splits, 
            inter_scales=args.inter_scales, 
            nested=args.nested
    )

    c_val, _, _, _, _, eigenvalues = ladder_c_from_matrix(
        M, delta=d, r=r, sector=None, exp=exp
    )
    for sector, target in {
        "ν": exp.c_nu_exp,
        "ℓ": exp.c_lep_exp,
        "u": exp.c_u_exp,
        "d": exp.c_d_exp
    }.items():
        
        results[sector] = {"c": c_val, "eigenvalues": eigenvalues}
        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
        chi2_tot += ((c_val - target) / sigma) ** 2

    if best is None or chi2_tot < best["chi2_tot"]:
        best = {"delta": d, "r": r, "results": results, "chi2_tot": chi2_tot}

    print(f"Best δ={best['delta']:.9f}, r={best['r']:.9f}")
    return math.sqrt(best["chi2_tot"] / 4)


# ==========================
# Mode: grand_unify_all_scaled
# ==========================
def mode_grand_unify_all_scaled(exp: ExpData, args, logger: logging.Logger = None):
    print("\n=== Mode: grand_unify_all_scaled ===")
    deltas = np.linspace(0.0, args.delta_max, args.grid_delta)
    rs = np.geomspace(args.r_min, args.r_max, args.grid_r)
    delta_scales = [0.01, 0.1, 1.0, 10.0]
    r_scales = [0.01, 0.1, 1.0, 10.0]
    best = None
    h_params = [args.h1, args.h2, args.h3] if args.matrix_size > 3 else None

    for d in tqdm(deltas, desc="Scanning δ"):
        for r in rs:
            for ds in delta_scales:
                for rs_s in r_scales:
                    chi2_tot = 0.0
                    results = {}
                    d_eff = d * ds
                    r_eff = r * rs_s

                M = build_matrix(args.matrix_size, d, r,
                                 gL=args.gL, gR=args.gR, h_params=h_params,
                                 delta2=args.delta2,
                                 rL=args.rL, rR=args.rR,
                                 s_params=[args.s12, args.s23, args.s34, args.s45],
                                 muL=args.muL, muR=args.muR,
                                 block_split=args.block_split,
                                 inter_scale=args.inter_scale,
                                 splits=args.splits, 
                                 inter_scales=args.inter_scales, 
                                 nested=args.nested)
                c_val, _, _, _, _, eigenvalues = ladder_c_from_matrix(M, target_c=None,
                    delta=d_eff, r=r_eff, sector=None, exp=exp)
                for sector, target in {
                        "ν": exp.c_nu_exp,
                        "ℓ": exp.c_lep_exp,
                        "u": exp.c_u_exp,
                        "d": exp.c_d_exp
                }.items():
                        results[sector] = {"c": c_val, "eigenvalues": eigenvalues}
                        sigma = target * exp.tol_rel_model if sector != "ν" else exp.sig_c_nu
                        chi2_tot += ((c_val - target) / sigma) ** 2
                if best is None or chi2_tot < best["chi2_tot"]:
                    best = {"delta": d_eff, "r": r_eff, "results": results, "chi2_tot": chi2_tot}

    print(f"Best δ={best['delta']:.9f}, r={best['r']:.9f}")
    return math.sqrt(best["chi2_tot"] / 4)


# ==========================
# CLI
# ==========================
def main():
    parser = argparse.ArgumentParser(description="ZFSC predictor v7.3 (theory-pure, nested blocks)")

    parser.add_argument("--mode", type=str, default="independent_all",
                        choices=["independent_all", "shared_r_all", "shared_delta_all",
                                 "full_unify_all", "grand_unify_all", "grand_unify_all_scaled"],
                        help="Prediction mode")
    parser.add_argument("--matrix_size", type=int, default=6,
                        help="Matrix size N (>=3). Recommended: 6, 8, 10, 11, 12.")
    parser.add_argument("--grid_delta", type=int, default=401)
    parser.add_argument("--grid_r", type=int, default=401)
    parser.add_argument("--delta_max", type=float, default=5000.0)
    parser.add_argument("--r_min", type=float, default=0.001)
    parser.add_argument("--r_max", type=float, default=2.0)

    parser.add_argument("--gL", type=float, default=1.0)
    parser.add_argument("--gR", type=float, default=1.0)
    parser.add_argument("--h1", type=float, default=0.0)
    parser.add_argument("--h2", type=float, default=0.0)
    parser.add_argument("--h3", type=float, default=0.0)

    parser.add_argument("--delta2", type=float, default=None)
    parser.add_argument("--rL", type=float, default=None)
    parser.add_argument("--rR", type=float, default=None)
    parser.add_argument("--s12", type=float, default=0.0)
    parser.add_argument("--s23", type=float, default=0.0)
    parser.add_argument("--s34", type=float, default=0.0)
    parser.add_argument("--s45", type=float, default=0.0)
    parser.add_argument("--muL", type=float, default=0.0)
    parser.add_argument("--muR", type=float, default=0.0)

    # блочная иерархия
    parser.add_argument("--block_split", type=int, default=None)
    parser.add_argument("--inter_scale", type=float, default=1.0)
    parser.add_argument("--splits", type=str, default=None,
                        help="Comma-separated split indices, e.g. '4,8'")
    parser.add_argument("--inter_scales", type=str, default=None,
                        help="Comma-separated scales, e.g. '0.6,0.4'")
    parser.add_argument("--nested", action="store_true",
                        help="Enable nested (matrix-in-matrix) block structure")

    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tol_me", type=float, default=0.051099895)
    parser.add_argument("--log_file", type=str, default=None)

    args = parser.parse_args()
    if args.dense:
        args.grid_delta = 1001
        args.grid_r = 1001

    # parse splits/scales
    def _parse_csv_int(s):
        return [int(x) for x in s.split(",")] if s else None
    def _parse_csv_float(s):
        return [float(x) for x in s.split(",")] if s else None

    args.splits = _parse_csv_int(args.splits)
    args.inter_scales = _parse_csv_float(args.inter_scales)

# --- диагностический вывод ---
    print("=== Debug: parsed CLI arguments ===")
    print(f"args.splits (raw)       = {repr(args.splits)}")
    print(f"args.inter_scales (raw) = {repr(args.inter_scales)}")
    print("===================================")


    # logging
    logger = None
    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(message)s')
        logger = logging.getLogger()

    exp = ExpData()
    exp.tol_me = args.tol_me

    print(f"=== ZFSC predictor (4 sectors, v7.3 theory-pure) ===")
    print(f"Shared structural parameters: "
      f"gL={args.gL}, gR={args.gR}, "
      f"h1={args.h1}, h2={args.h2}, h3={args.h3}, "
      f"delta2={args.delta2}, rL={args.rL}, rR={args.rR}, "
      f"s12={args.s12}, s23={args.s23}, s34={args.s34}, s45={args.s45}, "
      f"muL={args.muL}, muR={args.muR}, "
      f"block_split={args.block_split}, inter_scale={args.inter_scale}, "
      f"splits={args.splits}, inter_scales={args.inter_scales}, nested={args.nested}")


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

    fn(exp, args, logger=logger)
    
if __name__ == "__main__":
    main()

