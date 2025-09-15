import os, json, math, csv, hashlib, datetime as dt
from typing import Dict, Tuple, List
import numpy as np

# === базовый путь — текущая папка проекта ===
BASE = os.path.dirname(os.path.abspath(__file__))

def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

import sys
sys.path.append(BASE)

from meta import VERSION, PROGRAM_HASH
from config_io import load_config
from utils import make_rng
from geometry import build_H_eff, sample_hyperparams
from mixing import apply_mixing
from cycles import apply_cycles
from reservoir import apply_reservoir
from snail import apply_snail
from duality import apply_duality
from plateaus import robust_plateaus
from stability import evaluate_stability
from shells import segment_shells

def diag_with(N: int, params: dict, rng: np.random.Generator):
    H0 = build_H_eff(N, params, rng)
    H1 = apply_mixing(H0, rng, params)
    H2 = apply_cycles(H1, params)
    H3 = apply_reservoir(H2, rng, params)
    H4 = apply_snail(H3, params)
    Hf, doubled = apply_duality(H4, rng, params)
    vals, vecs = np.linalg.eigh(Hf)
    order = np.argsort(vals)
    vals = vals[order]; vecs = vecs[:, order]
    return (vals, vecs), {"doubled": bool(doubled), "N_eff": int(Hf.shape[0]), "H": Hf}

def sector_rng(seed_base: int, sector_tag: str) -> np.random.Generator:
    return make_rng((int(seed_base) ^ (hash(sector_tag) & 0xFFFFFFFF)) & 0xFFFFFFFF)

def sector_params(base_params: dict, sector: str) -> dict:
    p = json.loads(json.dumps(base_params))
    edge = p["umbrella"]["edge"]
    if sector == "u":
        pass
    elif sector == "d":
        p["knot"]["eta"] = -abs(p["knot"]["eta"])
    elif sector == "l":
        p["umbrella"]["edge"] = ("open" if edge == "hard" else "hard")
    elif sector == "nu":
        p["knot"]["g"] = int(max(1, p["knot"]["g"]))
        p["knot"]["g"] = int(min(4, p["knot"]["g"] + 1))
    return p

def plateau_means(eigs: np.ndarray, cfg: dict):
    pts = robust_plateaus(eigs, min_width=int(cfg["filters"]["plateau_min_width"]),
                          tau=float(cfg["filters"]["plateau_tau"]))
    mus = [float(np.mean(p)) for p in pts] if pts else []
    return pts, mus

def align_phases(Ua: np.ndarray, Ub: np.ndarray):
    def normalize(U):
        V = U.copy()
        for j in range(V.shape[1]):
            z = V[0, j]
            if z == 0: continue
            V[:, j] *= np.conj(z) / (abs(z) if abs(z) > 0 else 1.0)
        return V
    Ua_n = normalize(Ua); Ub_n = normalize(Ub)
    D = np.diag(np.diag(Ua_n.conj().T @ Ub_n))
    phases = np.exp(-1j * np.angle(np.diag(D)))
    Ub_n = Ub_n @ np.diag(phases)
    return Ua_n, Ub_n

def ckm_like(Uu: np.ndarray, Ud: np.ndarray, k: int = 3) -> np.ndarray:
    Au, Ad = Uu[:, :k], Ud[:, :k]
    Au, Ad = align_phases(Au, Ad)
    return Au.conj().T @ Ad

def pmns_like(Ul: np.ndarray, Uv: np.ndarray, k: int = 3) -> np.ndarray:
    Al, Av = Ul[:, :k], Uv[:, :k]
    Al, Av = align_phases(Al, Av)
    return Al.conj().T @ Av

def mixing_angles(U: np.ndarray):
    U = U[:3, :3]
    s13 = abs(U[0, 2]); s13 = np.clip(s13, 0.0, 1.0)
    c13 = math.sqrt(max(1e-12, 1 - s13**2))
    s12 = abs(U[0, 1]) / c13 if c13 > 1e-12 else 0.0; s12 = np.clip(s12, 0.0, 1.0)
    s23 = abs(U[1, 2]) / c13 if c13 > 1e-12 else 0.0; s23 = np.clip(s23, 0.0, 1.0)
    th13 = math.asin(s13); th12 = math.asin(s12); th23 = math.asin(s23)
    J = np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0]))
    denom = (math.sin(th12)*math.cos(th12)*math.sin(th23)*math.cos(th23)*
             (math.sin(th13)*(math.cos(th13)**2) + 1e-12))
    delta = 0.0 if denom == 0 else float(math.asin(np.clip(J/denom, -1.0, 1.0)))
    return float(th12), float(th23), float(th13), delta

def write_csv(path: str, rows: List[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close(); return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow(r)

def jump_once(config_path: str) -> str:
    cfg, meta = load_config(config_path)
    runs_root = cfg.get("runs_root", "runs")
    out_root = os.path.join(BASE, runs_root, "Tjump", timestamp()); os.makedirs(out_root, exist_ok=True)

    meta_out = {"VERSION": VERSION, "PROGRAM_HASH": PROGRAM_HASH,
                "CONFIG_VERSION": meta.get("CONFIG_VERSION"), "CONFIG_HASH": meta.get("CONFIG_HASH"),
                "CONFIG_SHA256": sha256_path(config_path), "timestamp": timestamp()}
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    seeds = cfg.get("seeds", [42]); sizes = cfg.get("sizes", [128]); N = int(sizes[0]); seed0 = int(seeds[0])
    rng0 = make_rng(seed0); base_params = sample_hyperparams(rng0, cfg)
    sectors = ["u", "d", "l", "nu"]
    evals = {}; evecs = {}; plateau_mu = {}; robust_rows = []

    for s in sectors:
        rng_s = sector_rng(seed0, s); params_s = sector_params(base_params, s)
        (vals, vecs), info = diag_with(N, params_s, rng_s)
        evals[s], evecs[s] = vals, vecs
        Pset, mus = plateau_means(vals, cfg); plateau_mu[s] = mus

        # правильный вызов устойчивости
        Hf = info["H"]
        stable_plateaus, stab_metrics = evaluate_stability(vals, Pset, Hf, rng_s, cfg)
        clusters, shell_metrics = segment_shells(vals, cfg)

        robust_rows.append({
            "sector": s, "N": info["N_eff"], "num_plateaus": len(Pset),
            "plateau1_mu": (mus[0] if len(mus)>0 else None),
            "plateau2_mu": (mus[1] if len(mus)>1 else None),
            "plateau3_mu": (mus[2] if len(mus)>2 else None),
            "avg_persistence": stab_metrics.get("plateau_persistence"),
            "avg_purity": shell_metrics.get("shell_purity"),
            "shell_count": shell_metrics.get("shell_count")
        })

    # анкер по mZ
    lambda_Z = float(plateau_mu["l"][2]) if len(plateau_mu["l"]) >= 3 else (
               float(plateau_mu["l"][1]) if len(plateau_mu["l"]) >= 2 else
               (float(evals["l"][2]) if evals["l"].size >= 3 else float(evals["l"][-1])))
    mZ = 91.1876; kappa_scale = mZ / lambda_Z if lambda_Z != 0 else float("nan")

    masses_rows = []
    for s in sectors:
        mus = plateau_mu[s]
        while len(mus) < 3 and evals[s].size >= 3:
            mus = list(mus) + [float(evals[s][len(mus)])]
        masses_rows.append({"sector": s,
                            "m1": (kappa_scale * mus[0]) if len(mus)>0 else None,
                            "m2": (kappa_scale * mus[1]) if len(mus)>1 else None,
                            "m3": (kappa_scale * mus[2]) if len(mus)>2 else None})
    write_csv(os.path.join(out_root, "masses.csv"), masses_rows)

    Vckm = ckm_like(evecs["u"], evecs["d"], k=3); Upmns = pmns_like(evecs["l"], evecs["nu"], k=3)
    th12_c, th23_c, th13_c, d_c = mixing_angles(Vckm); th12_p, th23_p, th13_p, d_p = mixing_angles(Upmns)
    mix_rows = []
    def mat_rows(name, M):
        rows = []
        for i in range(3):
            for j in range(3):
                rows.append({"matrix": name, "i": i+1, "j": j+1,
                             "real": float(np.real(M[i,j])), "imag": float(np.imag(M[i,j])),
                             "abs": float(abs(M[i,j]))})
        return rows
    mix_rows += mat_rows("CKM", Vckm)
    mix_rows += [{"matrix": "CKM_angles",
                  "theta12_deg": math.degrees(th12_c), "theta23_deg": math.degrees(th23_c),
                  "theta13_deg": math.degrees(th13_c), "delta_deg": math.degrees(d_c)}]
    mix_rows += mat_rows("PMNS", Upmns)
    mix_rows += [{"matrix": "PMNS_angles",
                  "theta12_deg": math.degrees(th12_p), "theta23_deg": math.degrees(th23_p),
                  "theta13_deg": math.degrees(th13_p), "delta_deg": math.degrees(d_p)}]
    write_csv(os.path.join(out_root, "mixing.csv"), mix_rows)

    const_rows = [{
        "alpha_inv": None, "alpha": None, "g": None, "g_prime": None,
        "g_s": None, "sin2_thetaW": None,
        "note": "Placeholders — φ_P→{α,g,g',g_s} mapping from cycles to be encoded."
    }]
    write_csv(os.path.join(out_root, "constants.csv"), const_rows)
    write_csv(os.path.join(out_root, "robust.csv"), robust_rows)

    out_json = {"kappa_scale_from_mZ": kappa_scale, "lambda_Z_surrogate": lambda_Z,
                "masses": masses_rows, "ckm_abs": np.abs(Vckm).tolist(),
                "pmns_abs": np.abs(Upmns).tolist(),
                "ckm_angles_deg": [math.degrees(th12_c), math.degrees(th23_c), math.degrees(th13_c), math.degrees(d_c)],
                "pmns_angles_deg": [math.degrees(th12_p), math.degrees(th23_p), math.degrees(th13_p), math.degrees(d_p)],
                "VERSION": VERSION, "PROGRAM_HASH": PROGRAM_HASH,
                "CONFIG_VERSION": meta.get("CONFIG_VERSION"), "CONFIG_HASH": meta.get("CONFIG_HASH")}
    with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    return out_root
