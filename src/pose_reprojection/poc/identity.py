import numpy as np

def apply(data, config=None):
    """Identity POC.

    This intentionally returns the input keypoints unchanged.
    It is used to verify that the POC wrapper/evaluation path does not
    accidentally alter results.
    """
    config = config or {}

    out = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
        else:
            out[key] = value

    out["poc_method"] = np.array("identity")
    out["poc_description"] = np.array("Returns input keypoints unchanged.")

    return out
