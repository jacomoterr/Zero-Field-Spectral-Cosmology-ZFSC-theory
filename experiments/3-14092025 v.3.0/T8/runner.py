# runner_parallel.py
# Zero-Field Spectral Cosmology (ZFSC) v3.1.5
# Основной цикл прогонов с параллельным запуском (8 потоков)
# Сохраняет результаты и визуализацию (surface.png) в папку прогона

import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from utils import VERSION, PROGRAM_HASH, timestamp
from plateaus import robust_plateaus
from geometry import build_H_eff, sample_hyperparams
from config_io import load_config, save_config_copy
from stats import make_summary_stats, write_summary_stats
from outputs import save_meta, save_summary
from mixing import apply_mixing
from cycles import apply_cycles
from reservoir import apply_reservoir
from duality import apply_duality
from snail import apply_snail
from viz_surface import save_surface


def single_run(N, s, i, cfg):
    rng = np.random.default_rng(np.random.SeedSequence([s, i]))
    params = sample_hyperparams(rng, cfg)
    H = build_H_eff(N, params, rng)
    H = apply_mixing(H, rng, params)
    H = apply_cycles(H, params)
    H = apply_reservoir(H, rng, params)
    H = apply_snail(H, params)
    H_eff, doubled = apply_duality(H, rng, params)
    N_eff = H_eff.shape[0]

    try:
        eigvals = np.linalg.eigvalsh(H_eff)
    except np.linalg.LinAlgError:
        return None, params

    eigvals.sort(kind="mergesort")
    plateaus = robust_plateaus(
        eigvals,
        min_width=int(cfg["filters"]["plateau_min_width"]),
        tau=float(cfg["filters"]["plateau_tau"])
    )

    # метрики ширины плато
    plateau_widths = [len(p) for p in plateaus]
    plateau_width = float(np.mean(plateau_widths)) if plateau_widths else None
    plateau_width_max = int(np.max(plateau_widths)) if plateau_widths else None

    # gap_ratio (третья генерация)
    g3 = None
    if len(plateaus) >= 3:
        l2 = float(np.mean(plateaus[1]))
        l3 = float(np.mean(plateaus[2]))
        if len(plateaus) > 3:
            l4 = float(np.mean(plateaus[3]))
            denom = (l3 - l2)
            if denom != 0 and np.isfinite(denom):
                g3 = (l4 - l3) / denom

    row = {
        "N": N,
        "N_eff": N_eff,
        "seed": s,
        "point": i,
        "num_plateaus": len(plateaus),
        "plateau_width": plateau_width,
        "plateau_width_max": plateau_width_max,
        "gap_ratio": (None if g3 is None else float(g3)),
        "edge": params["umbrella"]["edge"],
        "alpha": float(params["umbrella"]["alpha"]),
        "kappa": float(params["umbrella"]["kappa"]),
        "gamma": float(params["lotus"]["gamma"]),
        "m": int(params["umbrella"]["anisotropy"]["m"]),
        "eps": float(params["umbrella"]["anisotropy"]["epsilon"]),
        "eta": float(params["knot"]["eta"]),
        "g_local": int(params["knot"]["g"]),
        "delta": float(params["cycles"]["delta"]),
        "P": int(params["cycles"]["P"]),
        "κ_gr": float(params.get("reservoir", {}).get("kappas", [0,0,0,0])[0]),
        "κ_em": float(params.get("reservoir", {}).get("kappas", [0,0,0,0])[1]),
        "κ_wk": float(params.get("reservoir", {}).get("kappas", [0,0,0,0])[2]),
        "κ_st": float(params.get("reservoir", {}).get("kappas", [0,0,0,0])[3]),
        "dual_enabled": bool(params["dual"]["enabled"]),
        "kappa_lr": float(params["dual"]["kappa_lr"]),
        "epsilon_asym": float(params["dual"]["epsilon_asym"]),
        "phase": float(params["dual"]["phase"]),
        "sn_rho": float(params["snail"]["rho"]),
        "sn_m": int(params["snail"]["m"]),
        "sn_phi": float(params["snail"]["phi"]),
        "sn_p": float(params["snail"]["p"]),
    }
    return row, params


def main():
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config_full.yaml")
    cfg, meta = load_config(CONFIG_PATH)

    run_id = timestamp()
    root = os.path.join(cfg["runs_root"], run_id)
    os.makedirs(root, exist_ok=True)

    save_config_copy(CONFIG_PATH, root)
    meta["time_start"] = run_id
    save_meta(meta, root)

    sizes = list(cfg["sizes"])
    seeds = list(cfg["seeds"])
    points = int(cfg["sampling"]["points"])

    jobs = [(N, s, i) for N in sizes for s in seeds for i in range(points)]

    # Параллельный запуск
    results = Parallel(n_jobs=8)(
        delayed(single_run)(N, s, i, cfg) for (N, s, i) in tqdm(jobs, desc="Universe run", ncols=100)
    )

    # Фильтруем None
    summary_rows = [r for (r, p) in results if r is not None]
    save_summary(summary_rows, root)
    summary_stats = make_summary_stats(summary_rows)
    write_summary_stats(summary_stats, root)

    print(f"\n✅ Прогон {run_id} завершён.")
    print(f"Файлы сохранены в: {root}")
    print(f"  - summary.csv / summary.txt")
    print(f"  - summary_stats.json / summary_stats.csv")

    # --- Визуализация ---
    valid_params = [p for (r, p) in results if r is not None]
    if valid_params:
        params_last = valid_params[-1]
        save_surface(params_last, os.path.join(root, "surface.png"))


if __name__ == "__main__":
    main()
