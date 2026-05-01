import json

import numpy as np
import torch


CAMERA_FEATURE_6D = [
    ("distance_m", 0),
    ("height_m", 1),
    ("yaw_deg", 2),
    ("pitch_deg", 3),
    ("roll_deg", 4),
    ("fov_deg", 5),
]

STABLE_SCALE_BONES = [
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (0, 8),
    (11, 12),
    (14, 15),
]


def _camera_feature_info(config, feature_dim, permutation=None):
    inputs = config.get("corrector_inputs", {})
    return {
        "use_camera_parameters": bool(inputs.get("use_camera_parameters", True)),
        "camera_feature_mode": inputs.get("camera_feature_mode", "raw_9d"),
        "z_ablation": inputs.get("z_ablation", "true"),
        "feature_dim": int(feature_dim),
        "shuffle_seed": int(config.get("seed", 1234)),
        "permutation_saved": permutation is not None,
    }


def _normalize_6d(z, ranges):
    out = []
    for name, idx in CAMERA_FEATURE_6D:
        if name not in ranges:
            raise KeyError(f"synthetic.camera_ranges is missing {name}; required for normalized_6d camera features")
        lo, hi = ranges[name]
        center = 0.5 * (float(lo) + float(hi))
        half = max(0.5 * (float(hi) - float(lo)), 1e-6)
        out.append((z[:, idx] - center) / half)
    return np.stack(out, axis=-1).astype(np.float32)


def _shuffle_permutation(n, seed):
    n = int(n)
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    perm = np.empty(n, dtype=np.int64)
    perm[order] = np.roll(order, 1)
    return perm


def prepare_z_features(arrays, config):
    """Prepare Pc-only camera features without mutating true arrays['z']."""
    z = np.asarray(arrays["z"], dtype=np.float32)
    inputs = config.get("corrector_inputs", {})

    if not bool(inputs.get("use_camera_parameters", True)):
        features = np.zeros((z.shape[0], 0), dtype=np.float32)
        arrays["z_features"] = features
        arrays["z_feature_info_json"] = np.array(json.dumps(_camera_feature_info(config, 0)))
        return arrays

    mode = inputs.get("camera_feature_mode", "raw_9d")
    if mode == "raw_9d":
        features = z.copy()
    elif mode == "normalized_6d":
        ranges = config.get("synthetic", {}).get("camera_ranges", {})
        features = _normalize_6d(z, ranges)
    else:
        raise ValueError(f"Unknown corrector_inputs.camera_feature_mode: {mode}")

    ablation = inputs.get("z_ablation", "true")
    permutation = None
    if ablation == "true":
        pass
    elif ablation == "zero":
        features = np.zeros_like(features)
    elif ablation == "shuffle":
        permutation = _shuffle_permutation(z.shape[0], int(config.get("seed", 1234)))
        features = features[permutation]
        arrays["z_feature_permutation"] = permutation.astype(np.int64)
    else:
        raise ValueError(f"Unknown corrector_inputs.z_ablation: {ablation}")

    arrays["z_features"] = features.astype(np.float32)
    arrays["z_feature_info_json"] = np.array(json.dumps(_camera_feature_info(config, features.shape[-1], permutation)))
    return arrays


def stable_body_scale_torch(y_centered, eps=1e-6):
    vals = []
    for a, b in STABLE_SCALE_BONES:
        vals.append(torch.linalg.norm(y_centered[:, :, a] - y_centered[:, :, b], dim=-1))
    scale = torch.stack(vals, dim=-1).mean(dim=-1)
    return scale.clamp_min(float(eps))[:, :, None, None]


def corrector_pose_input(pose, config):
    norm_cfg = config.get("corrector_normalization", {})
    if not bool(norm_cfg.get("enabled", False)):
        return pose

    root_joint = int(norm_cfg.get("root_joint", 0))
    root = pose[:, :, root_joint:root_joint + 1, :]
    centered = pose - root
    scale_mode = norm_cfg.get("scale_mode", "stable_bones")
    if scale_mode != "stable_bones":
        raise ValueError(f"Unknown corrector_normalization.scale_mode: {scale_mode}")
    scale = stable_body_scale_torch(centered, eps=float(norm_cfg.get("eps", 1e-6)))
    return centered / scale


def corrector_y_input(y_lifted, config):
    return corrector_pose_input(y_lifted, config)


def _bone_lengths_np(x):
    vals = []
    for a, b in STABLE_SCALE_BONES:
        vals.append(np.linalg.norm(x[..., a, :] - x[..., b, :], axis=-1))
    return np.stack(vals, axis=-1)


def prepare_reliability_features(arrays, config):
    inputs = config.get("corrector_inputs", {})
    if not bool(inputs.get("use_reliability_features", False)):
        n, t = arrays["y_lifted"].shape[:2]
        arrays["reliability_features"] = np.zeros((n, t, 0), dtype=np.float32)
        arrays["reliability_feature_info_json"] = np.array(json.dumps({
            "use_reliability_features": False,
            "feature_dim": 0,
            "features": [],
        }))
        return arrays

    y = np.asarray(arrays["y_lifted"], dtype=np.float32)
    x_geo = arrays.get("x_geo_used", arrays.get("x_geo"))
    if x_geo is None:
        n, t = y.shape[:2]
        arrays["reliability_features"] = np.zeros((n, t, 0), dtype=np.float32)
        arrays["reliability_feature_info_json"] = np.array(json.dumps({
            "use_reliability_features": True,
            "feature_dim": 0,
            "features": [],
            "missing": "x_geo_used",
        }))
        return arrays

    x_geo = np.asarray(x_geo, dtype=np.float32)
    if x_geo.shape != y.shape:
        raise ValueError(f"Reliability features require x_geo shape {x_geo.shape} to match y_lifted {y.shape}")

    parts = []
    names = []

    joint_delta_m = np.linalg.norm(y - x_geo, axis=-1).astype(np.float32)
    parts.append(joint_delta_m)
    names.extend([f"joint_delta_m_{i}" for i in range(joint_delta_m.shape[-1])])

    raw_meta = arrays.get("raw_2d_metadata")
    if raw_meta is not None:
        raw_meta = np.asarray(raw_meta, dtype=np.float32)
    if raw_meta is not None and raw_meta.ndim == 3 and raw_meta.shape[:2] == y.shape[:2] and raw_meta.shape[-1] >= 5:
        bbox_height = raw_meta[..., 3:4]
        bbox_area = raw_meta[..., 4:5]
        center_dist = np.linalg.norm(raw_meta[..., 0:2] - 0.5, axis=-1, keepdims=True).astype(np.float32)
    else:
        bbox_height = np.zeros(y.shape[:2] + (1,), dtype=np.float32)
        bbox_area = np.zeros_like(bbox_height)
        center_dist = np.zeros_like(bbox_height)
    parts.extend([bbox_height, bbox_area, center_dist])
    names.extend(["bbox_height_norm", "bbox_area_norm", "bbox_center_distance_norm"])

    rmse = arrays.get("xgeo_fit_rmse_mm")
    rmse_m = np.zeros(y.shape[:2] + (1,), dtype=np.float32)
    if rmse is not None:
        rmse_arr = np.asarray(rmse, dtype=np.float32)
        if rmse_arr.shape[:2] == y.shape[:2]:
            rmse_m = rmse_arr[..., None] / 1000.0
    parts.append(rmse_m)
    names.append("xgeo_fit_rmse_m")

    y_bones = _bone_lengths_np(y)
    xg_bones = _bone_lengths_np(x_geo)
    bone_diff = np.abs(y_bones - xg_bones).mean(axis=-1, keepdims=True).astype(np.float32)
    parts.append(bone_diff)
    names.append("stable_bone_mean_abs_diff_m")

    depth_prior = arrays.get("xgeo_depth_prior_mm")
    depth_prior_norm = np.zeros(y.shape[:2] + (1,), dtype=np.float32)
    if depth_prior is not None:
        depth_prior_arr = np.asarray(depth_prior, dtype=np.float32)
        if depth_prior_arr.shape[:2] == y.shape[:2]:
            depth_prior_norm = depth_prior_arr[..., None] / 10000.0
    parts.append(depth_prior_norm)
    names.append("xgeo_depth_prior_10m")

    features = np.concatenate(parts, axis=-1).astype(np.float32) if parts else np.zeros(y.shape[:2] + (0,), dtype=np.float32)
    arrays["reliability_features"] = features
    arrays["reliability_feature_info_json"] = np.array(json.dumps({
        "use_reliability_features": True,
        "feature_dim": int(features.shape[-1]),
        "features": names,
    }))
    return arrays


def apply_xgeo_ablation(arrays, config):
    """Create arrays['x_geo_used'] for Pc inputs and residual bases.

    arrays['x_geo'] remains the raw fitted geometry candidate for diagnostics.
    """
    mode = config.get("geometry_refinement", {}).get("xgeo_ablation", "none")
    if "x_geo" not in arrays:
        if mode == "none":
            return arrays
        raise KeyError("geometry_refinement.xgeo_ablation requires arrays['x_geo']")

    if mode == "none":
        used = arrays["x_geo"]
        applied = False
        source = "true_x_geo"
    elif mode == "y_lifted":
        if "y_lifted" not in arrays:
            raise KeyError("geometry_refinement.xgeo_ablation='y_lifted' requires arrays['y_lifted']")
        if arrays["y_lifted"].shape != arrays["x_geo"].shape:
            raise ValueError(
                f"y_lifted shape {arrays['y_lifted'].shape} does not match x_geo shape {arrays['x_geo'].shape}"
            )
        used = arrays["y_lifted"]
        applied = True
        source = "y_lifted"
    elif mode == "zero":
        used = np.zeros_like(arrays["x_geo"], dtype=np.float32)
        applied = True
        source = "zero"
    else:
        raise ValueError(f"Unknown geometry_refinement.xgeo_ablation: {mode}")

    arrays["x_geo_used"] = np.asarray(used, dtype=np.float32)
    arrays["xgeo_ablation_info_json"] = np.array(json.dumps({
        "xgeo_ablation": mode,
        "xgeo_ablation_applied": bool(applied),
        "xgeo_ablation_source": source,
        "x_geo_shape": list(arrays["x_geo"].shape),
        "x_geo_used_shape": list(arrays["x_geo_used"].shape),
    }))
    return arrays


def _residual_base_pose(y_lifted, x_geo, config):
    output_cfg = config.get("corrector_output", {})
    base = output_cfg.get("base", "y_lifted")
    if base == "y_lifted":
        return y_lifted
    if base == "x_geo":
        if x_geo is None:
            raise ValueError("corrector_output.base='x_geo' requires geometry_refinement.enabled=true and arrays['x_geo']")
        return x_geo
    if base == "gated_y_xgeo":
        raise ValueError("corrector_output.base='gated_y_xgeo' requires a model gate; call _residual_base_pose_with_gate")
    raise ValueError(f"Unknown corrector_output.base: {base}")


def _split_model_output(model_output):
    if isinstance(model_output, dict):
        if "residual" not in model_output:
            raise ValueError("Model output dict is missing 'residual'")
        return model_output["residual"], model_output.get("gate")
    return model_output, None


def _residual_base_pose_with_gate(y_lifted, x_geo, config, gate):
    output_cfg = config.get("corrector_output", {})
    base = output_cfg.get("base", "y_lifted")
    if base != "gated_y_xgeo":
        return _residual_base_pose(y_lifted, x_geo, config)
    if x_geo is None:
        raise ValueError("corrector_output.base='gated_y_xgeo' requires arrays['x_geo']")
    if gate is None:
        raise ValueError("corrector_output.base='gated_y_xgeo' requires a gate from the model")
    if gate.shape[:3] != y_lifted.shape[:3] or gate.shape[-1] != 1:
        raise ValueError(f"Gate shape {tuple(gate.shape)} is incompatible with Y shape {tuple(y_lifted.shape)}")
    return gate * y_lifted + (1.0 - gate) * x_geo


def compose_prediction(y_lifted, model_output, config, x_geo=None):
    output_cfg = config.get("corrector_output", {})
    mode = output_cfg.get("mode", "residual")
    norm_cfg = config.get("corrector_normalization", {})
    normalization_enabled = bool(norm_cfg.get("enabled", False))
    residual, gate = _split_model_output(model_output)

    if mode == "gate_only":
        if output_cfg.get("base", "gated_y_xgeo") != "gated_y_xgeo":
            raise ValueError("corrector_output.mode='gate_only' requires base='gated_y_xgeo'")
        return _residual_base_pose_with_gate(y_lifted, x_geo, config, gate)

    if normalization_enabled:
        if mode != "residual":
            raise ValueError(
                "corrector_output.mode='full_pose' is not supported with "
                "corrector_normalization.enabled=true. Disable normalization for true no-Y full-pose experiments."
            )
        base_pose = _residual_base_pose_with_gate(y_lifted, x_geo, config, gate)
        root_joint = int(norm_cfg.get("root_joint", 0))
        root_base = base_pose[:, :, root_joint:root_joint + 1, :]
        base_centered = base_pose - root_base
        scale_base = stable_body_scale_torch(base_centered, eps=float(norm_cfg.get("eps", 1e-6)))
        base_norm = base_centered / scale_base
        x_hat_norm = base_norm + residual
        return root_base + scale_base * x_hat_norm

    if mode == "residual":
        return _residual_base_pose_with_gate(y_lifted, x_geo, config, gate) + residual
    if mode == "full_pose":
        if isinstance(model_output, dict):
            raise ValueError("corrector_output.mode='full_pose' does not support gated model outputs")
        return residual
    raise ValueError(f"Unknown corrector_output.mode: {mode}")
