from pose_reprojection.poc import identity

METHODS = {
    "identity": identity.apply,
}

def get_method(name):
    if name not in METHODS:
        available = ", ".join(sorted(METHODS.keys()))
        raise KeyError(f"Unknown POC method: {name}. Available methods: {available}")
    return METHODS[name]
