import os, json, shutil, yaml
from utils import sha256_of_file, VERSION, PROGRAM_HASH

def load_config(config_path: str):
    """Загружаем YAML-конфиг, проверяем VERSION/PROGRAM_HASH и возвращаем cfg + мета."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_hash_declared = cfg.get("CONFIG_HASH", "")
    config_version = cfg.get("CONFIG_VERSION", "unknown")

    if cfg.get("VERSION") != VERSION or cfg.get("PROGRAM_HASH") != PROGRAM_HASH:
        raise RuntimeError("Config/Program mismatch (VERSION or PROGRAM_HASH)!")

    config_file_sha256 = sha256_of_file(config_path)
    config_hash_match = (config_file_sha256 == config_hash_declared)

    meta = {
        "VERSION": VERSION,
        "PROGRAM_HASH": PROGRAM_HASH,
        "CONFIG_VERSION": config_version,
        "CONFIG_HASH_declared": config_hash_declared,
        "CONFIG_FILE_sha256": config_file_sha256,
        "config_hash_match": config_hash_match,
        "sizes": cfg["sizes"],
        "seeds": cfg["seeds"],
        "sampling": cfg["sampling"],
    }
    return cfg, meta

def save_config_copy(config_path: str, dest_dir: str):
    """Сохраняем копию конфига внутрь папки запуска."""
    try:
        shutil.copy2(config_path, os.path.join(dest_dir, os.path.basename(config_path)))
    except Exception:
        pass  # не фейлим прогон
