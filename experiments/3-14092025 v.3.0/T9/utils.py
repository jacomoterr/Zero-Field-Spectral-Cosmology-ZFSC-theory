import hashlib, datetime

VERSION = "3.1.4"
PROGRAM_HASH = "T3C9-EXAMPLE-0004"

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

# utils.py — добавить

import numpy as np

def coerce_seed(x) -> int:
    """
    Приводит что угодно к детерминированному 32-bit целому seed.
    - int -> int
    - str/None/другое -> стабильный хеш по строковому представлению.
    """
    try:
        # Частый случай: строка с числом
        return int(x)
    except Exception:
        return int(abs(hash(str(x))) % (2**32))

def make_rng(x) -> np.random.Generator:
    """Генератор Numpy с безопасным приведением seed."""
    return np.random.default_rng(coerce_seed(x))
