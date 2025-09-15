
# jump_run.py — one-click launcher for the "jump of faith"
import os
from jump import jump_once
CONFIG_PATH = os.path.join("config", "config_full.yaml")
if __name__ == "__main__":
    out = jump_once(CONFIG_PATH)
    print("✅ Jump finished:", out)
