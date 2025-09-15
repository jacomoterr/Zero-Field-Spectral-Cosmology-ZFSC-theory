# ablation.py
# управление отключением блоков (mixing, cycles, snail, reservoir, dual)

def apply_ablation(params, mode):
    """Отключает выбранные блоки по словарю mode"""
    if not mode.get("mixing", True):
        params["knot"]["eta"] = 0.0
    if not mode.get("cycles", True):
        params["cycles"]["delta"] = 0.0
    if not mode.get("snail", True):
        params["snail"]["rho"] = 0.0
    if not mode.get("reservoir", True):
        params["reservoir"] = {"kappas": [0.0, 0.0, 0.0, 0.0]}
    if not mode.get("dual", True):
        params["dual"]["enabled"] = False
    return params
