# core/single_run.py
# одна итерация спектрального прогона

from meta import PROGRAM_HASH   # теперь берём отсюда, без зацикливания
import numpy as np

from geometry import build_H_eff, sample_hyperparams
from mixing import apply_mixing
from cycles import apply_cycles
from reservoir import apply_reservoir
from duality import apply_duality
from snail import apply_snail
from plateaus import robust_plateaus
from stability import evaluate_stability
from shells import segment_shells
from ablation import apply_ablation


def single_run(N, s, i, ablation_id, mode, stab_params, level, cfg):
    rng = np.random.default_rng(
        np.random.SeedSequence([s, i, ablation_id, level, hash(PROGRAM_HASH) & 0xFFFF])
    )
    params = sample_hyperparams(rng, cfg)

    if mode:
        params = apply_ablation(params, mode)

    # --- сборка H_eff ---
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

    # --- поиск плато ---
    plateaus = robust_plateaus(
        eigvals,
        min_width=int(cfg["filters"]["plateau_min_width"]),
        tau=float(cfg["filters"]["plateau_tau"])
    )

    plateau_widths = [len(p) for p in plateaus]
    plateau_width = float(np.mean(plateau_widths)) if plateau_widths else None
    plateau_width_max = int(np.max(plateau_widths)) if plateau_widths else None

    # --- gap_ratio (третья генерация) ---
    g3 = None
    if len(plateaus) >= 3:
        l2 = float(np.mean(plateaus[1]))
        l3 = float(np.mean(plateaus[2]))
        if len(plateaus) > 3:
            l4 = float(np.mean(plateaus[3]))
            denom = (l3 - l2)
            if denom != 0 and np.isfinite(denom):
                g3 = (l4 - l3) / denom

    # --- устойчивость ---
    stable_plateaus, stab_metrics = evaluate_stability(
        eigvals, plateaus, H, rng, {**cfg, "stability": stab_params}
    )

    # --- оболочки ---
    clusters, shell_metrics = segment_shells(eigvals, cfg)

    row = {
        "adaptive_level": level,
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
        # stability params
        "q_min": stab_params["q_min"],
        "n_probe": stab_params["n_probe"],
        "sigma": stab_params["sigma"],
        # ablation
        "ablation_id": ablation_id,
        "abl_mixing": mode.get("mixing", True) if mode else True,
        "abl_cycles": mode.get("cycles", True) if mode else True,
        "abl_snail": mode.get("snail", True) if mode else True,
        "abl_reservoir": mode.get("reservoir", True) if mode else True,
        "abl_dual": mode.get("dual", True) if mode else True,
    }

    row.update(stab_metrics)
    row.update(shell_metrics)
    row["shell_sizes"] = "|".join(str(x) for x in shell_metrics["shell_sizes"])

    return row, params
