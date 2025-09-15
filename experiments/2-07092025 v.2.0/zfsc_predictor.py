import argparse
import numpy as np
import os
import datetime
from geometry import compose_matrix

# -------------------------
# Вспомогательные функции
# -------------------------
def sector_energy(eigs: np.ndarray) -> float:
    return float(np.sum(eigs**2))

def extract_masses(eigs: np.ndarray, n=3):
    eigs = np.sort(np.abs(eigs))
    return eigs[:n] if eigs.size >= n else eigs

def compute_c(masses):
    if len(masses) < 3:
        return None
    m1, m2, m3 = masses[:3]
    denom = m2**2 - m1**2
    if abs(denom) < 1e-12:
        return None
    return (m3**2 - m2**2) / denom

def compute_constants(masses_dict, fracs):
    results = {}
    if "lep" in masses_dict and "u" in masses_dict:
        ml = np.mean(masses_dict["lep"]) if len(masses_dict["lep"])>0 else 1.0
        mu = np.mean(masses_dict["u"]) if len(masses_dict["u"])>0 else 1.0
        if mu > 0:
            results["alpha_model"] = ml/mu
    if "lep" in masses_dict and "u" in masses_dict and "d" in masses_dict:
        ml = np.mean(masses_dict["lep"]) if len(masses_dict["lep"])>0 else 1.0
        mb = (np.mean(masses_dict["u"])+np.mean(masses_dict["d"]))/2.0
        if ml > 0:
            results["baryon_lepton_ratio"] = mb/ml
    if "lep" in fracs and "u" in fracs:
        w = fracs["lep"]
        s = fracs["u"]
        if s > 0:
            results["weak_strong_ratio"] = w/s
    if "alpha_model" in results and "weak_strong_ratio" in results:
        results["unification"] = (results["alpha_model"] + results["weak_strong_ratio"]) / 2
    return results

def mixing_matrix(U_a, U_b, n=3):
    Ua = U_a[:, :n]
    Ub = U_b[:, :n]
    return Ua.conj().T @ Ub

# -------------------------
# Основная программа
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix_size', type=int, default=11)
    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--r', type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join("logs", f"predictor_{stamp}.log")

    with open(logfile, "w", encoding="utf-8") as logf:
        header = f"=== ZFSC predictor run at {stamp} ==="
        print(header); logf.write(header + "\n")

        # параметры запуска
        args_line = "Args: " + ", ".join(f"{k}={v}" for k,v in vars(args).items())
        print(args_line); logf.write(args_line + "\n")

        # матрица и офсеты
        M, offsets = compose_matrix(args.matrix_size, args.delta, args.r)
        eigvals, eigvecs = np.linalg.eigh(M)
        print(f"Matrix size: {M.shape[0]}, dtype={M.dtype}")
        logf.write(f"Matrix size: {M.shape[0]}, dtype={M.dtype}\n")

        masses_dict = {}
        energies = {}
        fracs = {}

        for sec,(i0,i1) in offsets.items():
            vals = eigvals[i0:i1]
            masses = extract_masses(vals, 3)
            masses_dict[sec] = masses
            energies[sec] = sector_energy(vals)
            c_val = compute_c(masses)
            if c_val is not None:
                line = f"[{sec}] masses={masses}, c={c_val:.6g}"
            else:
                line = f"[{sec}] masses={masses}"
            print(line); logf.write(line + "\n")

        # энергодоли
        E_total = sum(energies.values())
        if E_total > 0:
            for sec in energies:
                fracs[sec] = energies[sec]/E_total
            line = "Energy fractions: " + ", ".join(f"{sec}={fracs[sec]:.2%}" for sec in fracs)
            print(line); logf.write(line + "\n")

        # физические константы
        consts = compute_constants(masses_dict, fracs)
        for k,v in consts.items():
            line = f"{k} = {v:.6g}"
            print(line); logf.write(line + "\n")

        # матрицы смешивания
        if "u" in offsets and "d" in offsets:
            i0,i1 = offsets["u"]
            j0,j1 = offsets["d"]
            Uu = eigvecs[i0:i1,:]
            Ud = eigvecs[j0:j1,:]
            CKM = mixing_matrix(Uu, Ud, n=3)
            line = f"CKM matrix:\n{np.round(CKM,3)}"
            print(line); logf.write(line + "\n")

        if "lep" in offsets and "nu" in offsets:
            i0,i1 = offsets["lep"]
            j0,j1 = offsets["nu"]
            Ul = eigvecs[i0:i1,:]
            Un = eigvecs[j0:j1,:]
            PMNS = mixing_matrix(Ul, Un, n=3)
            line = f"PMNS matrix:\n{np.round(PMNS,3)}"
            print(line); logf.write(line + "\n")

        footer = "=== Predictor finished ==="
        print(footer); logf.write(footer + "\n")

    print(f"Лог сохранён в {logfile}")

if __name__ == "__main__":
    main()
