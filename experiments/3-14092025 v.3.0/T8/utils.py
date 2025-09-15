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
