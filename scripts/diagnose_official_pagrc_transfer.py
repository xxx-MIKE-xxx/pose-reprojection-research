from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


OFFICIAL_ROOT = 14  # MPI-INF-3DHP official relevant pelvis/root index
H36M_ROOT = 0       # your training/Pc convention, likely pelvis index 0

OFFICIAL_NAMES = [
    "head_top", "neck",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "pelvis", "spine", "head",
]

H36M_NAMES = [
    "pelvis",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

# Approximate mapping from H36M-style order to official MPI-INF-3DHP relevant order.
# Used only for diagnostics.
H36M_TO_OFFICIAL = np.array([
    14,  # pelvis
    8,   # right_hip
    9,   # right_knee
    10,  # right_ankle
    11,  # left_hip
    12,  # left_knee
    13,  # left_ankle
    15,  # spine
    15,  # thorax approx -> spine
    1,   # neck
    16,  # head
    5,   # left_shoulder
    6,   # left_elbow
    7,   # left_wrist
    2,   # right_shoulder
    3,   # right_elbow
    4,   # right_wrist
], dtype=int)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        print(f"[missing] {path}")
        return {}
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    print(f"\n[npz] {path}")
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:36s} shape={str(v.shape):22s} dtype={v.dtype}")
        else:
            print(f"  {k:36s} type={type(v)}")
    return out


def load_json(path: Path) -> dict:
    if not path.exists():
        print(f"[missing] {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n[json] {path}")
    for k in [
        "frozen_lifter_mpjpe_mm",
        "corrected_mpjpe_mm",
        "xgeo_raw_mpjpe_mm",
        "xgeo_used_mpjpe_mm",
        "mean_gate_y_weight",
        "pc_status",
        "intrinsics_mode",
        "no_oracle_z",
        "z_features_mode",
    ]:
        if k in data:
            print(f"  {k}: {data[k]}")
    return data


def first_existing(d: dict[str, np.ndarray], names: list[str]) -> tuple[str | None, np.ndarray | None]:
    for name in names:
        if name in d:
            return name, d[name]
    # fallback: fuzzy
    lower = {k.lower(): k for k in d.keys()}
    for wanted in names:
        w = wanted.lower()
        for lk, orig in lower.items():
            if w in lk:
                return orig, d[orig]
    return None, None


def ensure_nj3(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim != 3:
        return None
    if x.shape[-1] == 3:
        return x.astype(np.float64)
    if x.shape[1] == 3:
        return np.transpose(x, (0, 2, 1)).astype(np.float64)
    return None


def root_center(x: np.ndarray, root: int) -> np.ndarray:
    return x - x[:, root:root + 1, :]


def per_joint_err(pred: np.ndarray, gt: np.ndarray, root: int = OFFICIAL_ROOT) -> np.ndarray:
    p = root_center(pred, root)
    g = root_center(gt, root)
    return np.linalg.norm(p - g, axis=-1)


def mpjpe(pred: np.ndarray, gt: np.ndarray, root: int = OFFICIAL_ROOT) -> float:
    return float(per_joint_err(pred, gt, root).mean())


def norm_stats(name: str, x: np.ndarray) -> None:
    if x is None:
        return
    flat = x.reshape(-1, 3)
    norms = np.linalg.norm(flat, axis=-1)
    print(
        f"{name:24s} coord_abs_mean={np.mean(np.abs(flat)):9.3f} "
        f"coord_abs_p95={np.percentile(np.abs(flat),95):9.3f} "
        f"norm_mean={norms.mean():9.3f} norm_p95={np.percentile(norms,95):9.3f} "
        f"min={np.nanmin(x):9.3f} max={np.nanmax(x):9.3f}"
    )


def delta_stats(name: str, a: np.ndarray, b: np.ndarray) -> None:
    d = np.linalg.norm(a - b, axis=-1)
    print(
        f"{name:24s} mean={d.mean():9.3f} p50={np.percentile(d,50):9.3f} "
        f"p95={np.percentile(d,95):9.3f} p99={np.percentile(d,99):9.3f} max={d.max():9.3f}"
    )


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, root: int = OFFICIAL_ROOT) -> float:
    pred = root_center(pred, root)
    gt = root_center(gt, root)

    errs = []
    for p, g in zip(pred, gt):
        mu_p = p.mean(axis=0, keepdims=True)
        mu_g = g.mean(axis=0, keepdims=True)
        X = p - mu_p
        Y = g - mu_g

        norm_x = np.linalg.norm(X)
        norm_y = np.linalg.norm(Y)
        if norm_x < 1e-8 or norm_y < 1e-8:
            errs.append(np.linalg.norm(p - g, axis=-1).mean())
            continue

        Xn = X / norm_x
        Yn = Y / norm_y
        H = Xn.T @ Yn
        U, s, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        scale = s.sum() * norm_y / norm_x
        t = mu_g - scale * mu_p @ R
        p_aligned = scale * p @ R + t
        errs.append(np.linalg.norm(p_aligned - g, axis=-1).mean())
    return float(np.mean(errs))


def print_metric_block(name: str, pred: np.ndarray, gt: np.ndarray) -> None:
    print(f"\n[metrics] {name}")
    for root in [OFFICIAL_ROOT, H36M_ROOT]:
        if root < pred.shape[1] and root < gt.shape[1]:
            print(f"  root={root:2d} MPJPE={mpjpe(pred, gt, root):9.3f}  PA-MPJPE={compute_pa_mpjpe(pred, gt, root):9.3f}")

    e = per_joint_err(pred, gt, OFFICIAL_ROOT).mean(axis=0)
    order = np.argsort(-e)
    print("  worst joints official-root:")
    for j in order[:8]:
        nm = OFFICIAL_NAMES[j] if j < len(OFFICIAL_NAMES) else f"joint{j}"
        print(f"    {j:2d} {nm:16s} {e[j]:9.3f} mm")


def maybe_reorder_h36m_pred_to_official(pred: np.ndarray) -> np.ndarray | None:
    if pred is None or pred.shape[1] != 17:
        return None
    out = np.zeros_like(pred)
    counts = np.zeros(17, dtype=np.float64)
    for h_idx, o_idx in enumerate(H36M_TO_OFFICIAL):
        out[:, o_idx, :] += pred[:, h_idx, :]
        counts[o_idx] += 1.0
    ok = counts > 0
    out[:, ok, :] /= counts[ok][None, :, None]
    # For missing official head_top, use head if missing.
    for j in range(17):
        if counts[j] == 0:
            out[:, j, :] = pred[:, 0, :]  # obvious bad fallback, only to keep shape
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("outputs/official_mpi3dhp/official_test_gt2d.npz"))
    ap.add_argument("--pc-dir", type=Path, default=Path("outputs/official_mpi3dhp/eval_gt2d_pagrc_h_noz"))
    ap.add_argument("--baseline-dir", type=Path, default=Path("outputs/official_mpi3dhp/eval_gt2d_frozen_lifter"))
    args = ap.parse_args()

    dataset = load_npz(args.dataset)
    base_pred = load_npz(args.baseline_dir / "predictions.npz")
    pc_pred = load_npz(args.pc_dir / "predictions.npz")

    load_json(args.baseline_dir / "metrics.json")
    load_json(args.pc_dir / "metrics.json")

    gt_name, gt = first_existing(dataset, ["x_gt3d_univ_mm", "x_gt3d_mm", "x_gt"])
    gt = ensure_nj3(gt)
    if gt is None:
        raise RuntimeError("Could not find GT 3D array in official dataset NPZ.")
    print(f"\n[gt] using {gt_name}, shape={gt.shape}")

    candidate_names = {
        "frozen/Y baseline": ["y_official_mm", "y_lifted", "frozen_lifter", "frozen_lifter_pred", "pred_frozen", "y_pred", "prediction"],
        "x_geo_raw": ["x_geo_raw_official_mm", "x_geo_raw", "xgeo_raw", "x_geo"],
        "x_geo_used": ["x_geo_used_official_mm", "x_geo_used", "xgeo_used"],
        "gated_base": ["gated_base_official_mm", "gated_base", "x_base", "base_pose"],
        "corrected/X_hat": ["pc_official_mm", "x_hat", "corrected", "corrected_pred", "prediction_corrected", "x_corrected"],
        "residual/dX": ["dX", "dx", "residual", "predicted_residual"],
        "gate": ["gate_y_weight", "gate", "gates", "mean_gate_y_weight"],
    }

    arrays = {}
    for label, names in candidate_names.items():
        n, arr = first_existing(pc_pred, names)
        if arr is None and label == "frozen/Y baseline":
            n, arr = first_existing(base_pred, names)
        arrays[label] = ensure_nj3(arr) if label != "gate" else arr
        if arr is not None:
            print(f"[found] {label:18s} -> {n}, raw_shape={arr.shape}")

    y = arrays["frozen/Y baseline"]
    xgeo_raw = arrays["x_geo_raw"]
    xgeo_used = arrays["x_geo_used"]
    xbase = arrays["gated_base"]
    xhat = arrays["corrected/X_hat"]
    dx = arrays["residual/dX"]
    gate = arrays["gate"]

    print("\n[scale / units]")
    norm_stats("GT official", gt)
    for label, arr in [
        ("Y frozen", y),
        ("X_geo_raw", xgeo_raw),
        ("X_geo_used", xgeo_used),
        ("X_base/gated", xbase),
        ("X_hat", xhat),
        ("dX", dx),
    ]:
        if arr is not None:
            norm_stats(label, arr)

    print("\n[official metrics by stage]")
    for label, arr in [
        ("Y frozen", y),
        ("X_geo_raw", xgeo_raw),
        ("X_geo_used", xgeo_used),
        ("X_base/gated", xbase),
        ("X_hat", xhat),
    ]:
        if arr is not None and arr.shape[:2] == gt.shape[:2]:
            print_metric_block(label, arr, gt)

    print("\n[deltas]")
    if y is not None and xhat is not None:
        delta_stats("|X_hat - Y|", xhat, y)
    if xbase is not None and xhat is not None:
        delta_stats("|X_hat - X_base| residual effect", xhat, xbase)
    if y is not None and xgeo_used is not None:
        delta_stats("|X_geo_used - Y|", xgeo_used, y)
    if xgeo_raw is not None and xgeo_used is not None:
        delta_stats("|X_geo_raw - X_geo_used|", xgeo_raw, xgeo_used)

    if gate is not None:
        g = np.asarray(gate).astype(np.float64)
        print("\n[gate]")
        print("  shape:", g.shape)
        print(
            f"  mean={np.nanmean(g):.6f} std={np.nanstd(g):.6f} "
            f"min={np.nanmin(g):.6f} p05={np.nanpercentile(g,5):.6f} "
            f"p50={np.nanpercentile(g,50):.6f} p95={np.nanpercentile(g,95):.6f} max={np.nanmax(g):.6f}"
        )

    print("\n[joint-order diagnostic]")
    if y is not None and y.shape[1] == 17:
        print("  If the prediction is actually H36M-order, remapping it to official order gives:")
        y_remap = maybe_reorder_h36m_pred_to_official(y)
        if y_remap is not None:
            print(f"    Y remapped H36M->official MPJPE(root14): {mpjpe(y_remap, gt, OFFICIAL_ROOT):.3f}")
        if xhat is not None:
            xhat_remap = maybe_reorder_h36m_pred_to_official(xhat)
            if xhat_remap is not None:
                print(f"    X_hat remapped H36M->official MPJPE(root14): {mpjpe(xhat_remap, gt, OFFICIAL_ROOT):.3f}")

    print("\n[interpretation hints]")
    print("  - If X_base/gated is already huge, the Y/X_geo coordinate frame or joint order is wrong.")
    print("  - If X_base is reasonable but X_hat is huge, the residual head is out-of-domain or feature order is wrong.")
    print("  - If root=0 metrics are much better than root=14, Pc output is likely in H36M convention.")
    print("  - If H36M->official remap improves a lot, add explicit official<->training joint mapping before Pc.")
    print("  - If |X_hat-X_base| is hundreds of mm, inspect residual normalization/root/scale.")


if __name__ == "__main__":
    main()
