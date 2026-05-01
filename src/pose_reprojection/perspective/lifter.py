from pathlib import Path
import sys
import numpy as np


def _auto_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_pseudo_lifter(u_norm, z_vec=None, strength=0.25):
    """Deterministic fallback lifter for smoke tests.

    This is not a scientific baseline. It exists only so the pipeline can be
    smoke-tested on machines without VideoPose3D checkpoints.
    """
    u = np.asarray(u_norm, dtype=np.float32)
    x = u[..., 0]
    y = -u[..., 1]
    depth = np.zeros_like(x)

    if z_vec is not None:
        yaw = np.asarray(z_vec[:, 2], dtype=np.float32)[:, None, None] / 55.0
        pitch = np.asarray(z_vec[:, 3], dtype=np.float32)[:, None, None] / 35.0
        fov = (np.asarray(z_vec[:, 5], dtype=np.float32)[:, None, None] - 55.0) / 30.0
        depth = strength * (yaw * x + pitch * y + fov * (x * x + y * y))

    out = np.stack([x, y, depth], axis=-1)
    out = out - out[:, :, 0:1, :]
    return out.astype(np.float32)


def run_videopose3d_lifter(u_norm, config):
    import torch

    vp3d_root = Path(config["lifter"]["third_party_root"]).resolve()
    if not vp3d_root.exists():
        raise FileNotFoundError(vp3d_root)

    sys.path.insert(0, str(vp3d_root))
    from common.model import TemporalModel

    ckpt_path = Path(config["lifter"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = config["lifter"].get("device", "auto")
    if device == "auto":
        device = _auto_device()

    model = TemporalModel(
        num_joints_in=17,
        in_features=2,
        num_joints_out=17,
        filter_widths=[3, 3, 3, 3, 3],
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["model_pos"] if isinstance(ckpt, dict) and "model_pos" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    pad = int(config["lifter"].get("pad", 121))
    preds = []
    batch_size = int(config["lifter"].get("batch_size", 1))

    for start in range(0, len(u_norm), batch_size):
        batch = u_norm[start:start + batch_size].astype(np.float32)
        if pad > 0:
            batch = np.pad(batch, ((0, 0), (pad, pad), (0, 0), (0, 0)), mode="edge")

        inp = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            pred = model(inp).cpu().numpy()

        if pred.ndim != 4 or pred.shape[2:] != (17, 3):
            raise ValueError(f"Unexpected VideoPose3D output shape: {pred.shape}")

        preds.append(pred.astype(np.float32))

    out = np.concatenate(preds, axis=0)

    # With padding, VideoPose3D returns the original unpadded sequence length.
    if out.shape[1] != u_norm.shape[1]:
        min_t = min(out.shape[1], u_norm.shape[1])
        out = out[:, :min_t]
    return out.astype(np.float32)


def fit_global_similarity(pred, target):
    """Fit one similarity transform pred -> target using all train points."""
    X = pred.reshape(-1, 3).astype(np.float64)
    Y = target.reshape(-1, 3).astype(np.float64)

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    X0 = X - mu_x
    Y0 = Y - mu_y

    var_x = np.sum(X0 ** 2)
    if var_x < 1e-12:
        return {"scale": 1.0, "R": np.eye(3), "t": np.zeros(3)}

    K = X0.T @ Y0
    U, _, Vt = np.linalg.svd(K)
    Z = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Z[-1, -1] = -1
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var_x
    t = mu_y - scale * (mu_x @ R.T)

    return {"scale": float(scale), "R": R.astype(np.float32), "t": t.astype(np.float32)}


def apply_global_similarity(pred, transform):
    s = float(transform["scale"])
    R = np.asarray(transform["R"], dtype=np.float32)
    t = np.asarray(transform["t"], dtype=np.float32)
    return (s * (pred @ R.T) + t).astype(np.float32)


def run_frozen_lifter(arrays, config):
    lifter_type = config["lifter"].get("type", "videopose3d")
    if lifter_type == "videopose3d":
        y = run_videopose3d_lifter(arrays["u_norm"], config)
    elif lifter_type == "pseudo":
        y = run_pseudo_lifter(
            arrays["u_norm"],
            z_vec=arrays.get("z"),
            strength=float(config["lifter"].get("pseudo_bias_strength", 0.25)),
        )
    else:
        raise ValueError(f"Unknown lifter type: {lifter_type}")

    # Root-center lifter output before fitting/evaluation.
    y = y - y[:, :, 0:1, :]

    alignment = config["lifter"].get("alignment", "train_global_similarity")
    transform = None
    if alignment == "train_global_similarity":
        train_idx = arrays["train_indices"].astype(np.int64)
        transform = fit_global_similarity(y[train_idx], arrays["x_gt"][train_idx])
        y = apply_global_similarity(y, transform)
    elif alignment in ("none", None):
        pass
    else:
        raise ValueError(f"Unknown lifter alignment: {alignment}")

    y = y - y[:, :, 0:1, :]
    return y.astype(np.float32), transform
