import hashlib
import json

import numpy as np

from .camera import focal_from_fov, vector_to_camera_params


def camera_intrinsics_from_params(params_or_z, image_width=None, image_height=None):
    """Build pinhole intrinsics using the same focal-from-FOV convention as synthetic projection."""
    if isinstance(params_or_z, dict):
        w = float(params_or_z.get("image_width", image_width))
        h = float(params_or_z.get("image_height", image_height))
        f = float(params_or_z.get("focal_px", focal_from_fov(w, params_or_z["fov_deg"])))
        K = np.array([[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float32)
        return K

    z = np.asarray(params_or_z, dtype=np.float32)
    if z.shape[-1] < 9:
        if image_width is None or image_height is None:
            raise ValueError("image_width and image_height are required when z does not include image size/focal")
        f = focal_from_fov(float(image_width), z[..., 5])
        w = np.broadcast_to(float(image_width), z.shape[:-1])
        h = np.broadcast_to(float(image_height), z.shape[:-1])
    else:
        w = z[..., 6]
        h = z[..., 7]
        f = z[..., 8]

    K = np.zeros(z.shape[:-1] + (3, 3), dtype=np.float32)
    K[..., 0, 0] = f
    K[..., 1, 1] = f
    K[..., 0, 2] = 0.5 * w
    K[..., 1, 2] = 0.5 * h
    K[..., 2, 2] = 1.0
    return K


def _broadcast_invK(K, point_leading_ndim):
    invK = np.linalg.inv(np.asarray(K, dtype=np.float32))
    if invK.ndim == 2:
        shape = (1,) * point_leading_ndim + (3, 3)
        return invK.reshape(shape)

    k_leading = invK.shape[:-2]
    if len(k_leading) > point_leading_ndim:
        raise ValueError(f"K leading dims {k_leading} are not compatible with keypoints")
    shape = k_leading + (1,) * (point_leading_ndim - len(k_leading)) + (3, 3)
    return invK.reshape(shape)


def perspective_encoding_from_keypoints(u_px, K):
    """Return z=1 image-plane coordinates inv(K) @ [u, v, 1]."""
    u = np.asarray(u_px, dtype=np.float32)
    if u.shape[-1] == 2:
        ones = np.ones(u.shape[:-1] + (1,), dtype=np.float32)
        p = np.concatenate([u, ones], axis=-1)
    elif u.shape[-1] == 3:
        p = u
    else:
        raise ValueError(f"Expected keypoints with last dim 2 or 3, got {u.shape}")

    invK = _broadcast_invK(K, len(p.shape[:-1]))
    xy1 = np.matmul(invK, p[..., None])[..., 0]
    return xy1.astype(np.float32)


def rays_from_keypoints(u_px, K, normalize=True):
    """Compute per-keypoint camera rays from pixels and intrinsics."""
    rays = perspective_encoding_from_keypoints(u_px, K)
    if normalize:
        norm = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays = rays / np.maximum(norm, 1e-8)
    return rays.astype(np.float32)


def _skew(v):
    z = np.zeros(v.shape[:-1], dtype=np.float32)
    return np.stack(
        [
            np.stack([z, -v[..., 2], v[..., 1]], axis=-1),
            np.stack([v[..., 2], z, -v[..., 0]], axis=-1),
            np.stack([-v[..., 1], v[..., 0], z], axis=-1),
        ],
        axis=-2,
    )


def canonicalize_rays_by_root(rays, root_joint=0):
    """Rotate rays so the root ray aligns with the optical axis [0, 0, 1]."""
    rays = np.asarray(rays, dtype=np.float32)
    root = rays[..., int(root_joint), :]
    root = root / np.maximum(np.linalg.norm(root, axis=-1, keepdims=True), 1e-8)
    target = np.zeros_like(root)
    target[..., 2] = 1.0

    v = np.cross(root, target)
    s = np.linalg.norm(v, axis=-1)
    c = np.sum(root * target, axis=-1)

    # Near 180 degrees, choose a stable axis orthogonal to root.
    anti = (s < 1e-6) & (c < 0.0)
    if np.any(anti):
        fallback = np.zeros_like(v)
        fallback[..., 0] = 1.0
        v = np.where(anti[..., None], fallback, v)
        s = np.where(anti, 1.0, s)
        c = np.where(anti, -1.0, c)

    K = _skew(v)
    eye = np.eye(3, dtype=np.float32).reshape((1,) * len(s.shape) + (3, 3))
    coeff = ((1.0 - c) / np.maximum(s * s, 1e-8))[..., None, None]
    R = eye + K + np.matmul(K, K) * coeff

    same = (s < 1e-6) & (c >= 0.0)
    if np.any(same):
        R = np.where(same[..., None, None], eye, R)

    rays_can = np.einsum("...ij,...kj->...ki", R, rays)
    return rays_can.astype(np.float32), R.astype(np.float32)


def shuffle_features_by_sequence(features, seed):
    features = np.asarray(features, dtype=np.float32)
    n = int(features.shape[0])
    if n <= 1:
        perm = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        order = rng.permutation(n)
        perm = np.empty(n, dtype=np.int64)
        perm[order] = np.roll(order, 1)
    digest = hashlib.sha256(perm.tobytes()).hexdigest()
    return features[perm].astype(np.float32), perm.astype(np.int64), digest


def ensure_geometry_arrays(arrays, config):
    """Attach ray/PE arrays derived from input 2D and true camera intrinsics."""
    z_before = np.asarray(arrays["z"]).copy()
    if "rays_input" not in arrays or "ray_pe_input" not in arrays:
        K = camera_intrinsics_from_params(arrays["z"])
        arrays["rays_input"] = rays_from_keypoints(arrays["u_px"], K).astype(np.float32)
        arrays["ray_pe_input"] = perspective_encoding_from_keypoints(arrays["u_px"], K).astype(np.float32)
        if "u_px_clean" in arrays:
            arrays["rays_clean"] = rays_from_keypoints(arrays["u_px_clean"], K).astype(np.float32)
            arrays["ray_pe_clean"] = perspective_encoding_from_keypoints(arrays["u_px_clean"], K).astype(np.float32)

    norms = np.linalg.norm(arrays["rays_input"], axis=-1)
    checks = {
        "rays_shape": list(arrays["rays_input"].shape),
        "unit_ray_norm_mean": float(norms.mean()),
        "unit_ray_norm_std": float(norms.std()),
        "unit_ray_norm_min": float(norms.min()),
        "unit_ray_norm_max": float(norms.max()),
        "true_z_unchanged": bool(np.array_equal(z_before, arrays["z"])),
    }
    arrays["geometry_checks_json"] = np.array(json.dumps(checks))
    return arrays


def prepare_ray_features(arrays, config):
    inputs = config.get("corrector_inputs", {})
    geom = config.get("geometry_features", {})

    if not (bool(inputs.get("use_ray_features", False)) or bool(inputs.get("use_rays", False))):
        arrays["ray_features"] = np.zeros((arrays["x_gt"].shape[0], arrays["x_gt"].shape[1], 0), dtype=np.float32)
        arrays["ray_feature_info_json"] = np.array(json.dumps({
            "use_ray_features": False,
            "ray_feature_mode": geom.get("ray_feature_mode", "unit_rays"),
            "ray_ablation": inputs.get("ray_ablation", "true"),
            "canonicalize_root_ray": bool(geom.get("canonicalize_root_ray", False)),
            "ray_feature_dim": 0,
        }))
        return arrays

    ensure_geometry_arrays(arrays, config)
    mode = geom.get("ray_feature_mode", "unit_rays")
    root_joint = int(geom.get("root_joint", 0))
    rays = arrays["rays_input"].astype(np.float32)

    if mode == "unit_rays":
        feat = rays
    elif mode == "z1_xy":
        feat = arrays["ray_pe_input"][..., :2]
    elif mode == "root_centered_unit_rays":
        feat = rays - rays[..., root_joint:root_joint + 1, :]
    elif mode == "canonical_unit_rays":
        feat, R = canonicalize_rays_by_root(rays, root_joint=root_joint)
        arrays["ray_canonical_rotation"] = R.astype(np.float32)
    else:
        raise ValueError(f"Unknown geometry_features.ray_feature_mode: {mode}")

    if bool(geom.get("canonicalize_root_ray", False)) and mode != "canonical_unit_rays":
        if feat.shape[-1] != 3:
            raise ValueError("geometry_features.canonicalize_root_ray requires 3D ray features")
        if mode == "root_centered_unit_rays":
            raise ValueError("Use ray_feature_mode='canonical_unit_rays' before root-centering; root-centered rays have no root direction")
        feat, R = canonicalize_rays_by_root(feat, root_joint=root_joint)
        arrays["ray_canonical_rotation"] = R.astype(np.float32)

    n, t = feat.shape[:2]
    flat = feat.reshape(n, t, -1).astype(np.float32)
    ablation = inputs.get("ray_ablation", "true")
    perm = None
    perm_hash = None
    fixed_points = None
    if ablation == "true":
        ray_features = flat
    elif ablation == "zero":
        ray_features = np.zeros_like(flat)
    elif ablation == "shuffle":
        ray_features, perm, perm_hash = shuffle_features_by_sequence(flat, int(config.get("seed", 1234)))
        fixed_points = int(np.sum(perm == np.arange(len(perm))))
        arrays["ray_feature_permutation"] = perm.astype(np.int64)
    else:
        raise ValueError(f"Unknown corrector_inputs.ray_ablation: {ablation}")

    arrays["ray_features"] = ray_features.astype(np.float32)
    info = {
        "use_ray_features": True,
        "ray_feature_mode": mode,
        "ray_ablation": ablation,
        "canonicalize_root_ray": bool(geom.get("canonicalize_root_ray", False) or mode == "canonical_unit_rays"),
        "root_joint": root_joint,
        "ray_feature_dim": int(ray_features.shape[-1]),
        "ray_shuffle_seed": int(config.get("seed", 1234)) if ablation == "shuffle" else None,
        "ray_permutation_hash": perm_hash,
        "ray_shuffle_fixed_point_count": fixed_points,
    }
    arrays["ray_feature_info_json"] = np.array(json.dumps(info))
    return arrays


def camera_params_list_from_z(z):
    z = np.asarray(z, dtype=np.float32)
    return [vector_to_camera_params(row) for row in z.reshape(-1, z.shape[-1])]
