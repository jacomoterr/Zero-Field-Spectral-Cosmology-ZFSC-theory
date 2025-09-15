# stress.py
# ZFSC v3.3 — стресс-тест выживших узлов (в том числе manual-run)

import os
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from core.single_run import single_run
from outputs import save_summary
import matplotlib.pyplot as plt


def stress_test(hotspots, cfg, root, level_tag="final"):
    """
    Прогоняет stress-test для hotspot'ов.
    - hotspots: список словарей {N, seed, point, ablation_id}
    - cfg: конфиг
    - root: папка прогона
    - level_tag: имя подпапки
    """
    if not hotspots:
        print("⚠️ Stress-test: нет hotspot'ов для проверки.")
        return []

    stress_cfg = cfg.get("stress", {})
    n_probe_values = stress_cfg.get("n_probe_list", [5, 10, 20])
    sigma_values   = stress_cfg.get("sigma_list", [1e-5, 5e-5, 1e-4])
    N_JOBS = int(cfg.get("parallel", {}).get("N_JOBS", 4))

    level_root = os.path.join(root, f"stress_{level_tag}")
    os.makedirs(level_root, exist_ok=True)

    jobs = []
    for hp in hotspots:
        for n_probe in n_probe_values:
            for sigma in sigma_values:
                stab_params = {
                    "q_min": cfg["stability"].get("q_min", 3.0),
                    "n_probe": n_probe,
                    "sigma": sigma
                }
                jobs.append((
                    hp["N"],
                    hp["seed"],
                    hp["point"],
                    hp["ablation_id"],
                    {},  # ablation_mode не нужен — уже зашит в hp
                    stab_params,
                    f"stress-{level_tag}",
                    cfg
                ))

    print(f"▶ Stress-test jobs = {len(jobs)}")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(single_run)(*job) for job in tqdm(
            jobs,
            desc="Stress compute",
            total=len(jobs),
            unit="task",
            dynamic_ncols=True,
            leave=True
        )
    )

    summary_rows = [r for (r, p) in results if r is not None]
    save_summary(summary_rows, level_root)
    print(f"✅ Stress-test complete, saved to {level_root}")

    # Сохраняем топ-10 узлов по persistence/purity
    build_top10(summary_rows, level_root)

    # Визуализация теплокарт (если включено)
    if stress_cfg.get("viz", False):
        build_persistence_map(summary_rows, n_probe_values, sigma_values, level_root)

    return summary_rows


def build_top10(summary_rows, out_dir):
    """Сохраняет top10.json по persistence/purity"""
    ranked = []
    for r in summary_rows:
        pers = r.get("plateau_persistence")
        pur = r.get("shell_purity")
        if pers is None or pur is None:
            continue
        ranked.append({
            "N": r.get("N"),
            "seed": r.get("seed"),
            "point": r.get("point"),
            "ablation_id": r.get("ablation_id"),
            "persistence": float(pers),
            "purity": float(pur),
            "gap_ratio": float(r.get("gap_ratio") or 0.0)
        })
    ranked.sort(key=lambda x: (x["persistence"], x["purity"]), reverse=True)
    top10 = ranked[:10]

    path = os.path.join(out_dir, "top10.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(top10, f, ensure_ascii=False, indent=2)
    print(f"  Saved top10.json ({len(top10)} узлов)")


def build_persistence_map(summary_rows, n_probe_values, sigma_values, out_dir):
    """Строит теплокарту persistence vs (n_probe, sigma)"""
    data = {}
    for r in summary_rows:
        key = (r.get("N"), r.get("ablation_id"))
        pers = r.get("plateau_persistence")
        n_probe = r.get("n_probe")
        sigma = r.get("sigma")
        if pers is None or n_probe is None or sigma is None:
            continue
        data.setdefault(key, []).append((n_probe, sigma, pers))

    for key, values in data.items():
        N, ablation_id = key
        mat = np.zeros((len(sigma_values), len(n_probe_values)))
        mat[:] = np.nan
        for (n_probe, sigma, pers) in values:
            i = n_probe_values.index(n_probe)
            j = sigma_values.index(sigma)
            mat[j, i] = pers

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(mat, origin="lower", aspect="auto",
                       extent=[min(n_probe_values)-0.5, max(n_probe_values)+0.5,
                               min(sigma_values)-0.5, max(sigma_values)+0.5])
        fig.colorbar(im, ax=ax, label="persistence")
        ax.set_xlabel("n_probe")
        ax.set_ylabel("sigma")
        ax.set_title(f"Persistence map N={N}, abl={ablation_id}")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"persistence_map_N{N}_abl{ablation_id}.png"), dpi=200)
        plt.close(fig)
