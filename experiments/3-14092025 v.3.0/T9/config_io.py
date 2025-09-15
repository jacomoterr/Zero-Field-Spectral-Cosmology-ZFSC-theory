# config_io.py
# Zero-Field Spectral Cosmology (ZFSC) v3.x
# Работа с конфигами: загрузка/сохранение

import os
import yaml
import shutil
from utils import VERSION, PROGRAM_HASH


def load_config(path):
    """Загружает конфиг YAML и возвращает (cfg, meta)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- раньше тут была жёсткая проверка VERSION/PROGRAM_HASH ---
    # if cfg["VERSION"] != VERSION or cfg["PROGRAM_HASH"] != PROGRAM_HASH:
    #     raise RuntimeError("Config/Program mismatch (VERSION or PROGRAM_HASH)!")

    # Теперь просто добавляем meta-инфо, без ошибок
    meta = {
        "VERSION": VERSION,
        "PROGRAM_HASH": PROGRAM_HASH,
        "CONFIG_VERSION": cfg.get("CONFIG_VERSION", "unknown"),
        "CONFIG_HASH": cfg.get("CONFIG_HASH", "unknown"),
    }
    return cfg, meta


def save_config_copy(config_path, run_root):
    """Сохраняет копию файла конфига в папку прогона."""
    dst = os.path.join(run_root, os.path.basename(config_path))
    shutil.copy2(config_path, dst)
    return dst
