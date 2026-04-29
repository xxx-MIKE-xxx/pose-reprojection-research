from pathlib import Path
import numpy as np

REQUIRED_KEYS = ["keypoints", "scores", "frame_indices", "image_size"]

def load_keypoint_npz(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)
    out = {key: data[key] for key in data.files}

    missing = [key for key in REQUIRED_KEYS if key not in out]
    if missing:
        raise KeyError(f"Missing required keys in {path}: {missing}")

    return out

def save_keypoint_npz(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)
    return path
