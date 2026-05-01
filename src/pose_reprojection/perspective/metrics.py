import json
from pathlib import Path
import numpy as np

from .camera import project_np, vector_to_camera_params
from .skeleton import H36M17_BONES, H36M17_NAMES, bone_lengths


def root_center(x, root=0):
    return x - x[..., root:root + 1, :]


def mpjpe(pred, target):
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def per_joint_error(pred, target):
    return np.linalg.norm(pred - target, axis=-1)


def pck_3d(pred, target, threshold):
    err = per_joint_error(pred, target)
    return float(np.mean(err < threshold))


def acceleration(x):
    if x.shape[1] < 3:
        return np.zeros((x.shape[0], 0, x.shape[2], x.shape[3]), dtype=x.dtype)
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def acceleration_error(pred, target):
    acc_p = acceleration(pred)
    acc_t = acceleration(target)
    if acc_p.size == 0:
        return 0.0
    return mpjpe(acc_p, acc_t)


def similarity_transform_single(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)
    X = pred - mu_pred
    Y = target - mu_target

    var_X = np.sum(X ** 2)
    if var_X < 1e-12:
        return pred.copy()

    K = X.T @ Y
    U, _, Vt = np.linalg.svd(K)

    Z = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        Z[-1, -1] = -1

    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var_X
    aligned = scale * (X @ R.T) + mu_target
    return aligned


def batch_pa_align(pred, target):
    out = np.zeros_like(pred, dtype=np.float64)
    flat_p = pred.reshape(-1, pred.shape[-2], 3)
    flat_t = target.reshape(-1, target.shape[-2], 3)
    flat_o = out.reshape(-1, pred.shape[-2], 3)
    for i in range(flat_p.shape[0]):
        flat_o[i] = similarity_transform_single(flat_p[i], flat_t[i])
    return out


def bone_length_error(pred, target):
    return float(np.mean(np.abs(bone_lengths(pred) - bone_lengths(target))))


def reprojection_error_px(pred, u_px, z_vec):
    errs = []
    for i in range(pred.shape[0]):
        params = vector_to_camera_params(z_vec[i])
        proj, _ = project_np(pred[i], params)
        errs.append(np.linalg.norm(proj - u_px[i], axis=-1))
    return float(np.mean(np.concatenate([e.reshape(-1) for e in errs], axis=0)))


def summarize_method(pred, gt, u_px, z_vec, pck_threshold=0.150, u_px_clean=None):
    pred_for_reprojection = pred.astype(np.float64)
    pred = root_center(pred_for_reprojection)
    gt = root_center(gt.astype(np.float64))

    pred_pa = batch_pa_align(pred, gt)

    pj = per_joint_error(pred, gt)
    pj_pa = per_joint_error(pred_pa, gt)
    reproj_input = reprojection_error_px(pred_for_reprojection, u_px, z_vec)

    out = {
        "mpjpe_mm": mpjpe(pred, gt) * 1000.0,
        "pa_mpjpe_mm": mpjpe(pred_pa, gt) * 1000.0,
        "p_mpjpe_mm": mpjpe(pred_pa, gt) * 1000.0,
        "pck150": pck_3d(pred, gt, pck_threshold),
        "pck150_pa": pck_3d(pred_pa, gt, pck_threshold),
        "bone_length_error_mm": bone_length_error(pred, gt) * 1000.0,
        "reprojection_error_px": reproj_input,
        "reprojection_error_to_input_px": reproj_input,
        "accel_error_mm_per_frame2": acceleration_error(pred, gt) * 1000.0,
        "per_joint_error_mm": {name: float(pj[..., j].mean() * 1000.0) for j, name in enumerate(H36M17_NAMES)},
        "per_joint_error_pa_mm": {name: float(pj_pa[..., j].mean() * 1000.0) for j, name in enumerate(H36M17_NAMES)},
    }
    if u_px_clean is not None:
        out["reprojection_error_to_clean_px"] = reprojection_error_px(pred_for_reprojection, u_px_clean, z_vec)
    return out


def camera_bucket_masks(z_vec, thresholds):
    distance = z_vec[:, 0]
    height = z_vec[:, 1]
    yaw = np.abs(z_vec[:, 2])
    pitch = np.abs(z_vec[:, 3])
    fov = z_vec[:, 5]

    return {
        "close_camera": distance <= float(thresholds["close_distance_m"]),
        "low_camera": height <= float(thresholds["low_height_m"]),
        "high_camera": height >= float(thresholds["high_height_m"]),
        "high_yaw": yaw >= float(thresholds["high_abs_yaw_deg"]),
        "high_pitch": pitch >= float(thresholds["high_abs_pitch_deg"]),
        "wide_fov": fov >= float(thresholds["wide_fov_deg"]),
    }


def summarize_buckets(pred, gt, z_vec, thresholds):
    masks = camera_bucket_masks(z_vec, thresholds)
    rows = []
    for bucket, mask in masks.items():
        if not np.any(mask):
            continue
        p = root_center(pred[mask].astype(np.float64))
        g = root_center(gt[mask].astype(np.float64))
        rows.append({
            "bucket": bucket,
            "num_sequences": int(mask.sum()),
            "mpjpe_mm": mpjpe(p, g) * 1000.0,
            "pa_mpjpe_mm": mpjpe(batch_pa_align(p, g), g) * 1000.0,
        })
    return rows


def load_external_baselines(paths):
    out = {}
    for p in paths or []:
        path = Path(p)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            out[path.name] = data
        except Exception as exc:
            out[path.name] = {"error": str(exc)}
    return out
