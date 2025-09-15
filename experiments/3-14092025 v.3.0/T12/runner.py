# runner.py
# ZFSC v3.3 — оркестратор запуска (adaptive / stress_only)

import os
import yaml
import datetime
from meta import VERSION, PROGRAM_HASH
from config_io import load_config
from adaptive import adaptive_run, stress_only_run


def main():
    # Конфиг всегда в папке config
    config_file = os.path.join("config", "config_full.yaml")
    cfg, _ = load_config(config_file)   # <<< фикс: распаковка кортежа

    # Логируем версию и хэш
    print("=== Zero-Field Spectral Cosmology (ZFSC) ===")
    print(f" Program version: {VERSION}")
    print(f" Program hash   : {PROGRAM_HASH}")
    print(f" Config version : {cfg.get('CONFIG_VERSION')}")
    print(f" Config hash    : {cfg.get('CONFIG_HASH')}")

    # Создаём папку для текущего прогона
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(cfg.get("runs_root", "runs"), run_id)
    os.makedirs(root, exist_ok=True)
    print(f" Run folder: {root}")

    # Сохраняем полный конфиг копией в run-фолдер
    cfg_path = os.path.join(root, "config_used.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True)

    # --- Запуск выбранного режима ---
    mode = cfg.get("run_mode", "adaptive")
    if mode == "adaptive":
        adaptive_run(cfg, root)
    elif mode == "stress_only":
        stress_only_run(cfg, root)
    else:
        raise ValueError(f"Unknown run_mode: {mode}")

    print("✅ Run complete.")


if __name__ == "__main__":
    main()
