import numpy as np

def _as_float(x):
    return np.asarray(x, dtype=np.float64)

def root_center(x, root=0):
    x = _as_float(x)
    return x - x[..., root:root + 1, :]

def mpjpe(pred, target):
    pred = _as_float(pred)
    target = _as_float(target)
    return np.mean(np.linalg.norm(pred - target, axis=-1))

def per_joint_error(pred, target):
    pred = _as_float(pred)
    target = _as_float(target)
    return np.linalg.norm(pred - target, axis=-1)

def n_mpjpe(pred, target):
    pred = _as_float(pred)
    target = _as_float(target)

    numerator = np.sum(target * pred, axis=(1, 2), keepdims=True)
    denominator = np.sum(pred ** 2, axis=(1, 2), keepdims=True)
    scale = numerator / np.maximum(denominator, 1e-8)

    return mpjpe(pred * scale, target)

def pck_3d(pred, target, threshold):
    err = per_joint_error(pred, target)
    return float(np.mean(err < threshold))

def auc_3d(pred, target, max_threshold=0.150, num_steps=31):
    thresholds = np.linspace(0.0, max_threshold, num_steps)
    pcks = np.array([pck_3d(pred, target, t) for t in thresholds], dtype=np.float64)
    return float(np.trapz(pcks, thresholds) / max_threshold)

def similarity_transform_single(pred, target):
    """Align pred to target with scale, rotation, and translation.

    pred, target: (J, 3)
    """
    pred = _as_float(pred)
    target = _as_float(target)

    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)

    X = pred - mu_pred
    Y = target - mu_target

    var_X = np.sum(X ** 2)
    if var_X < 1e-12:
        return pred.copy()

    K = X.T @ Y
    U, s, Vt = np.linalg.svd(K)

    Z = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Z[-1, -1] = -1

    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var_X

    aligned = scale * (X @ R.T) + mu_target
    return aligned

def batch_procrustes_align(pred, target):
    pred = _as_float(pred)
    target = _as_float(target)

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    out = np.zeros_like(pred, dtype=np.float64)
    for i in range(pred.shape[0]):
        out[i] = similarity_transform_single(pred[i], target[i])
    return out

def pa_mpjpe(pred, target):
    aligned = batch_procrustes_align(pred, target)
    return mpjpe(aligned, target)

def acceleration(x):
    x = _as_float(x)
    if x.shape[0] < 3:
        return np.zeros((0,) + x.shape[1:], dtype=np.float64)
    return x[2:] - 2 * x[1:-1] + x[:-2]

def acceleration_error(pred, target):
    return mpjpe(acceleration(pred), acceleration(target))
