# Safe float conversion with fallback
def to_float(val, default=1.0):
    try:
        return float(val)
    except Exception:
        return float(default)

# Safe int conversion with fallback
def to_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return int(default)

# For dictionaries like command_scales, even if keys come as 0 or '0',
# normalize keys to strings ('0'..'5') and ensure values are floats.
def normalize_numkey_float_values(d):
    if not isinstance(d, dict):
        return {}
    out = {}
    for k, v in d.items():
        out[str(k)] = to_float(v, 1.0)
    return out