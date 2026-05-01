from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np
import torch

from pose_reprojection.perspective.camera import normalize_screen_coordinates_np, raw_2d_metadata
from pose_reprojection.perspective.config import load_config
from pose_reprojection.perspective.features import (
    apply_xgeo_ablation,
    compose_prediction,
    corrector_pose_input,
    corrector_y_input,
    prepare_reliability_features,
)
from pose_reprojection.perspective.geometry import (
    canonicalize_rays_by_root,
    rays_from_keypoints,
)
from pose_reprojection.perspective.lifter import run_pseudo_lifter, run_videopose3d_lifter
from pose_reprojection.perspective.model import ResidualMLP, build_features
from pose_reprojection.perspective.mpi3dhp_official import (
    H36M17_TO_OFFICIAL,
    OFFICIAL_ROOT_INDEX,
    OFFICIAL_THRESHOLDS_MM,
    OFFICIAL_TO_H36M17,
    compute_official_metrics,
    official_intrinsics_for_sequence,
    official_activity_names,
    official_joint_groups,
    sha256_file,
)
from pose_reprojection.perspective.ray_fit import (
    fit_xgeo_closest_to_lifter,
    fit_xgeo_from_rays_and_lifter,
)


def _json_hash(obj):
    text = json.dumps(obj, sort_keys=True, default=str)
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_npz(path):
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def _normalize_sequence(u_px, width, height):
    return normalize_screen_coordinates_np(u_px, int(width), int(height)).astype(np.float32)


def _official_from_h36m_m(x_h36m_m):
    if x_h36m_m is None:
        return np.array([], dtype=np.float32)
    return x_h36m_m[:, H36M17_TO_OFFICIAL].astype(np.float32) * 1000.0


def _alpha_label(alpha):
    text = f"{float(alpha):g}".replace("-", "neg").replace(".", "p")
    return f"residual_alpha_{text}"


def _unique_alphas(values):
    seen = set()
    out = []
    for value in values or []:
        alpha = float(value)
        key = f"{alpha:.12g}"
        if key not in seen:
            out.append(alpha)
            seen.add(key)
    return out


def _model_residual(model_output):
    if isinstance(model_output, dict):
        if "residual" not in model_output:
            raise ValueError("Model output dict is missing 'residual'")
        return model_output["residual"]
    return model_output


def _with_scaled_residual(model_output, alpha):
    residual = _model_residual(model_output)
    scaled = residual * float(alpha)
    if isinstance(model_output, dict):
        out = dict(model_output)
        out["residual"] = scaled
        return out
    return scaled


def _reproject_camera_points(x_cam, K):
    z = np.maximum(x_cam[..., 2], 1e-6)
    u = K[0, 0] * (x_cam[..., 0] / z) + K[0, 2]
    v = -K[1, 1] * (x_cam[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def _reproject_image_ray_points(x_img_ray, K):
    z = np.maximum(x_img_ray[..., 2], 1e-6)
    u = K[0, 0] * (x_img_ray[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (x_img_ray[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1).astype(np.float32)


def _ray_features_from_rays(rays, config):
    inputs = config.get("corrector_inputs", {})
    if not (bool(inputs.get("use_ray_features", False)) or bool(inputs.get("use_rays", False))):
        return np.zeros((rays.shape[0], 0), dtype=np.float32)

    geom = config.get("geometry_features", {})
    mode = geom.get("ray_feature_mode", "unit_rays")
    root_joint = int(geom.get("root_joint", 0))

    if mode == "unit_rays":
        feat = rays
    elif mode == "root_centered_unit_rays":
        feat = rays - rays[:, root_joint:root_joint + 1, :]
    elif mode == "canonical_unit_rays":
        feat, _ = canonicalize_rays_by_root(rays[None], root_joint=root_joint)
        feat = feat[0]
    elif mode == "z1_xy":
        z = np.maximum(rays[..., 2:3], 1e-8)
        feat = rays[..., :2] / z
    else:
        raise ValueError(f"Unsupported official ray_feature_mode: {mode}")

    ablation = inputs.get("ray_ablation", "true")
    flat = feat.reshape(feat.shape[0], -1).astype(np.float32)
    if ablation == "true":
        return flat
    if ablation == "zero":
        return np.zeros_like(flat)
    raise ValueError(f"Unsupported official ray_ablation={ablation}; use true or zero")


def _official_z_feature_dim(config):
    inputs = config.get("corrector_inputs", {})
    if not bool(inputs.get("use_camera_parameters", True)):
        return 0
    mode = inputs.get("camera_feature_mode", "raw_9d")
    if mode == "normalized_6d":
        return 6
    if mode == "raw_9d":
        return 9
    raise ValueError(f"Unsupported official camera_feature_mode={mode}")


def _run_lifter_for_official_gt2d(data, config):
    u_px = data["u_gt2d_px"].astype(np.float32)
    seq_names = data["sequence_names"].astype(str)
    widths = data["image_width"].astype(np.int32)
    heights = data["image_height"].astype(np.int32)

    pred_h36m = np.zeros((u_px.shape[0], 17, 3), dtype=np.float32)
    u_h36m_px = np.zeros((u_px.shape[0], 17, 2), dtype=np.float32)
    u_h36m_norm = np.zeros((u_px.shape[0], 17, 2), dtype=np.float32)

    for seq in sorted(np.unique(seq_names).tolist()):
        idx = np.where(seq_names == seq)[0]
        w = int(widths[idx[0]])
        h = int(heights[idx[0]])
        seq_u_h36m = u_px[idx][:, OFFICIAL_TO_H36M17]
        seq_u_norm = _normalize_sequence(seq_u_h36m, w, h)
        batch = seq_u_norm[None]

        lifter_type = config.get("lifter", {}).get("type", "videopose3d")
        if lifter_type == "videopose3d":
            y = run_videopose3d_lifter(batch, config)[0]
        elif lifter_type == "pseudo":
            y = run_pseudo_lifter(batch, z_vec=None, strength=float(config["lifter"].get("pseudo_bias_strength", 0.25)))[0]
        else:
            raise ValueError(f"Unsupported official eval lifter type: {lifter_type}")

        y = y - y[:, 0:1, :]
        pred_h36m[idx] = y.astype(np.float32)
        u_h36m_px[idx] = seq_u_h36m.astype(np.float32)
        u_h36m_norm[idx] = seq_u_norm.astype(np.float32)
        print(f"[lifter] {seq}: {len(idx)} frames, image={w}x{h}")

    pred_official_mm = pred_h36m[:, H36M17_TO_OFFICIAL] * 1000.0
    return {
        "y_h36m_m": pred_h36m,
        "y_official_mm": pred_official_mm.astype(np.float32),
        "u_h36m_px": u_h36m_px,
        "u_h36m_norm": u_h36m_norm,
    }


def _pc_unsupported_reason(config, disable_z):
    inputs = config.get("corrector_inputs", {})
    geom = config.get("geometry_refinement", {})
    if bool(inputs.get("use_camera_parameters", True)) and not disable_z:
        return "unsupported: official GT2D eval does not provide synthetic oracle z; pass --disable-z to use zero z features"
    needs_xgeo = (
        bool(inputs.get("use_geometry_fit_3d", False))
        or config.get("corrector_output", {}).get("base", "y_lifted") in ("x_geo", "gated_y_xgeo")
    )
    if needs_xgeo and (not bool(geom.get("enabled", False)) or geom.get("mode", "ray_depth_fit") != "ray_depth_fit"):
        return "unsupported: X_geo Pc requires geometry_refinement.enabled=true and mode=ray_depth_fit"
    return None


def _build_official_geometry(data, config, lifter_out, output_dir, test_root, xgeo_fit_options):
    seq_names = data["sequence_names"].astype(str)
    widths = data["image_width"].astype(np.int32)
    heights = data["image_height"].astype(np.int32)

    rays = np.zeros((seq_names.shape[0], 17, 3), dtype=np.float32)
    ray_features = None
    raw_meta_all = np.zeros((seq_names.shape[0], 7), dtype=np.float32)

    x_geo = None
    x_geo_camera_abs = None
    xgeo_depths_mm = None
    xgeo_depth_prior_mm = None
    xgeo_invalid_depth_mask = None
    needs_xgeo = (
        bool(config.get("corrector_inputs", {}).get("use_geometry_fit_3d", False))
        or config.get("corrector_output", {}).get("base", "y_lifted") in ("x_geo", "gated_y_xgeo")
    )
    if needs_xgeo:
        x_geo = np.zeros_like(lifter_out["y_h36m_m"], dtype=np.float32)
        x_geo_camera_abs = np.zeros_like(lifter_out["y_h36m_m"], dtype=np.float32)
        xgeo_depths_mm = np.zeros(lifter_out["y_h36m_m"].shape[:2], dtype=np.float32)
        xgeo_depth_prior_mm = np.zeros(lifter_out["y_h36m_m"].shape[0], dtype=np.float32)
        xgeo_invalid_depth_mask = np.zeros(lifter_out["y_h36m_m"].shape[:2], dtype=bool)

    per_sequence_stats = {}
    intrinsics = {}
    modes = set()
    reproj_vals = []
    xgeo_fit_mode = xgeo_fit_options.get("mode", "closest_y")

    for seq in sorted(np.unique(seq_names).tolist()):
        idx = np.where(seq_names == seq)[0]
        w = int(widths[idx[0]])
        h = int(heights[idx[0]])
        K, info = official_intrinsics_for_sequence(seq, w, h, test_root=test_root)
        intrinsics[seq] = info
        modes.add(info["intrinsics_mode"])
        raw_meta_all[idx] = raw_2d_metadata(lifter_out["u_h36m_px"][idx], w, h)

        seq_rays = rays_from_keypoints(lifter_out["u_h36m_px"][idx], K).astype(np.float32)
        rays[idx] = seq_rays
        seq_ray_features = _ray_features_from_rays(seq_rays, config)
        if ray_features is None and seq_ray_features.shape[-1] > 0:
            ray_features = np.zeros((seq_names.shape[0], seq_ray_features.shape[-1]), dtype=np.float32)
        if ray_features is not None:
            ray_features[idx] = seq_ray_features

        if x_geo is not None and xgeo_fit_mode == "free_depth":
            xg, stats = fit_xgeo_from_rays_and_lifter(
                lifter_out["y_h36m_m"][idx][None],
                seq_rays[None],
                config,
                camera_params=None,
                u_px=None,
            )
            seq_xg = xg[0].astype(np.float32)
            x_geo[idx] = seq_xg
            x_geo_camera_abs[idx] = seq_xg
            reproj = _reproject_camera_points(seq_xg, K)
            reproj_err = float(np.linalg.norm(reproj - lifter_out["u_h36m_px"][idx], axis=-1).mean())
            depth_mm = np.linalg.norm(seq_xg, axis=-1) * 1000.0
            xgeo_depths_mm[idx] = depth_mm.astype(np.float32)
            xgeo_depth_prior_mm[idx] = np.full(len(idx), float(np.mean(depth_mm)), dtype=np.float32)
            xgeo_invalid_depth_mask[idx] = False
            stats.update({
                "xgeo_fit_mode": "free_depth",
                "coordinate_mode": "camera_absolute_free_depth",
                "mean_depth_mm": float(np.mean(depth_mm)),
                "depth_prior_mean_mm": float(np.mean(depth_mm)),
                "invalid_depth_fraction": 0.0,
            })
            stats["mean_reprojection_error_to_input_px"] = reproj_err
            per_sequence_stats[seq] = stats
            reproj_vals.append(reproj_err)
            print(f"[x_geo] {seq}: mode=free_depth, reprojection_to_input_px={reproj_err:.6f}, intrinsics={info['intrinsics_mode']}")
        elif x_geo is not None and xgeo_fit_mode == "closest_y":
            fit = fit_xgeo_closest_to_lifter(
                seq_rays,
                lifter_out["y_h36m_m"][idx].astype(np.float32) * 1000.0,
                root_idx=0,
                u_px=lifter_out["u_h36m_px"][idx],
                intrinsics=K,
                depth_prior_mode=xgeo_fit_options.get("depth_prior_mode", "bbox"),
                root_prior_weight=xgeo_fit_options.get("root_prior_weight", 1.0),
                depth_ridge_weight=xgeo_fit_options.get("depth_ridge_weight", 0.01),
                min_depth_mm=xgeo_fit_options.get("min_depth_mm", 500.0),
                max_depth_mm=xgeo_fit_options.get("max_depth_mm", 10000.0),
            )
            seq_abs_m = fit["x_geo_camera_abs_mm"].astype(np.float32) / 1000.0
            seq_used_m = fit["x_geo_used_mm"].astype(np.float32) / 1000.0
            x_geo_camera_abs[idx] = seq_abs_m
            x_geo[idx] = seq_used_m
            xgeo_depths_mm[idx] = fit["depths_mm"].astype(np.float32)
            xgeo_depth_prior_mm[idx] = fit["depth_prior_mm"].astype(np.float32)
            xgeo_invalid_depth_mask[idx] = fit["invalid_depth_mask"].astype(bool)
            reproj = _reproject_image_ray_points(seq_abs_m, K)
            reproj_err = float(np.linalg.norm(reproj - lifter_out["u_h36m_px"][idx], axis=-1).mean())
            stats = dict(fit["stats"])
            stats["mean_reprojection_error_to_input_px"] = reproj_err
            per_sequence_stats[seq] = stats
            reproj_vals.append(reproj_err)
            print(f"[x_geo] {seq}: mode=closest_y, fit_rmse_mm={stats['mean_fit_rmse_mm']:.3f}, reprojection_to_input_px={reproj_err:.6f}, intrinsics={info['intrinsics_mode']}")
        elif x_geo is not None:
            raise ValueError(f"Unknown official xgeo fit mode: {xgeo_fit_mode}")

    x_geo_used = None
    ablation_info = {
        "xgeo_ablation": config.get("geometry_refinement", {}).get("xgeo_ablation", "none"),
        "xgeo_ablation_applied": False,
        "xgeo_ablation_source": "true_x_geo",
    }
    if x_geo is not None:
        tmp = {
            "x_geo": x_geo[None],
            "y_lifted": lifter_out["y_h36m_m"][None],
        }
        tmp = apply_xgeo_ablation(tmp, config)
        x_geo_used = tmp["x_geo_used"][0].astype(np.float32)
        ablation_info = json.loads(str(tmp["xgeo_ablation_info_json"]))

    reliability_features = None
    reliability_info = {}
    if bool(config.get("corrector_inputs", {}).get("use_reliability_features", False)) and x_geo_used is not None:
        rel_arrays = {
            "y_lifted": lifter_out["y_h36m_m"][None],
            "x_geo_used": x_geo_used[None],
            "raw_2d_metadata": raw_meta_all[None],
        }
        if xgeo_depth_prior_mm is not None:
            rel_arrays["xgeo_depth_prior_mm"] = xgeo_depth_prior_mm[None]
        rel_arrays = prepare_reliability_features(rel_arrays, config)
        reliability_features = rel_arrays["reliability_features"][0].astype(np.float32)
        reliability_info = json.loads(str(rel_arrays["reliability_feature_info_json"]))

    if len(modes) == 1:
        intrinsics_mode = next(iter(modes))
    else:
        intrinsics_mode = "mixed:" + ",".join(sorted(modes))

    geometry = {
        "rays_input": rays,
        "ray_features": ray_features,
        "x_geo": x_geo,
        "x_geo_camera_abs": x_geo_camera_abs,
        "x_geo_used": x_geo_used,
        "xgeo_depths_mm": xgeo_depths_mm,
        "xgeo_depth_prior_mm": xgeo_depth_prior_mm,
        "xgeo_invalid_depth_mask": xgeo_invalid_depth_mask,
        "reliability_features": reliability_features,
        "reliability_feature_info": reliability_info,
        "xgeo_fit_mode": xgeo_fit_mode,
        "coordinate_mode": "root_aligned_to_y" if xgeo_fit_mode == "closest_y" else "camera_absolute_free_depth",
        "intrinsics_mode": intrinsics_mode,
        "intrinsics": intrinsics,
        "x_geo_fit_stats": {
            "used_x_gt": False,
            "used_clean_2d": False,
            "camera_frame_mode": xgeo_fit_mode == "free_depth",
            "xgeo_fit_mode": xgeo_fit_mode,
            "coordinate_mode": "root_aligned_to_y" if xgeo_fit_mode == "closest_y" else "camera_absolute_free_depth",
            "per_sequence": per_sequence_stats,
            "mean_reprojection_error_to_input_px": float(np.mean(reproj_vals)) if reproj_vals else None,
            "mean_depth_mm": float(np.mean(xgeo_depths_mm)) if xgeo_depths_mm is not None else None,
            "depth_prior_mean_mm": float(np.mean(xgeo_depth_prior_mm)) if xgeo_depth_prior_mm is not None else None,
            "invalid_depth_fraction": float(np.mean(xgeo_invalid_depth_mask)) if xgeo_invalid_depth_mask is not None else None,
        },
        "xgeo_ablation_info": ablation_info,
    }
    (output_dir / "official_geometry_manifest.json").write_text(json.dumps({
        "intrinsics_mode": intrinsics_mode,
        "intrinsics": intrinsics,
        "xgeo_fit_mode": xgeo_fit_mode,
        "coordinate_mode": geometry["coordinate_mode"],
        "x_geo_fit_stats": geometry["x_geo_fit_stats"],
        "xgeo_ablation_info": ablation_info,
        "reliability_feature_info": reliability_info,
    }, indent=2), encoding="utf-8")
    return geometry


def _gate_stats(gates):
    if gates is None or len(gates) == 0:
        return {}
    vals = np.concatenate([g.reshape(-1) for g in gates], axis=0).astype(np.float64)
    return {
        "mean_gate_y_weight": float(vals.mean()),
        "std_gate_y_weight": float(vals.std()),
        "min_gate_y_weight": float(vals.min()),
        "max_gate_y_weight": float(vals.max()),
    }


def _run_pc_if_supported(
    data,
    config,
    checkpoint_path,
    lifter_out,
    disable_z,
    output_dir,
    test_root,
    base_only=False,
    residual_alphas=None,
    xgeo_fit_options=None,
):
    reason = _pc_unsupported_reason(config, disable_z)
    if reason is not None:
        raise RuntimeError(reason)
    if config.get("corrector_output", {}).get("mode", "residual") not in ("residual", "gate_only"):
        raise RuntimeError("official Pc diagnostics require corrector_output.mode='residual' or 'gate_only'")
    if checkpoint_path is None:
        raise ValueError("--pc-checkpoint is required when --baseline-only is not set")

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device_name = config.get("training", {}).get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    input_dim = int(ckpt.get("input_dim"))
    model = ResidualMLP(
        input_dim=input_dim,
        num_joints=17,
        hidden_dims=config["model"]["hidden_dims"],
        dropout=float(config["model"].get("dropout", 0.1)),
        zero_init_last=bool(config["model"].get("zero_init_last", True)),
        output_cfg=config.get("corrector_output", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    seq_names = data["sequence_names"].astype(str)
    widths = data["image_width"].astype(np.int32)
    heights = data["image_height"].astype(np.int32)
    x_hat = np.zeros_like(lifter_out["y_h36m_m"], dtype=np.float32)
    x_base = np.zeros_like(lifter_out["y_h36m_m"], dtype=np.float32)
    residual_effect = np.zeros_like(lifter_out["y_h36m_m"], dtype=np.float32)
    gate_y_weight = np.zeros(lifter_out["y_h36m_m"].shape[:2] + (1,), dtype=np.float32)
    alphas = _unique_alphas(residual_alphas)
    alpha_outputs = {
        _alpha_label(alpha): np.zeros_like(lifter_out["y_h36m_m"], dtype=np.float32)
        for alpha in alphas
    }
    geometry = _build_official_geometry(data, config, lifter_out, output_dir, test_root, xgeo_fit_options or {})
    gate_outs = []
    z_feature_dim = _official_z_feature_dim(config)

    with torch.no_grad():
        for seq in sorted(np.unique(seq_names).tolist()):
            idx = np.where(seq_names == seq)[0]
            w = int(widths[idx[0]])
            h = int(heights[idx[0]])
            y = torch.from_numpy(lifter_out["y_h36m_m"][idx][None]).float().to(device)
            u = torch.from_numpy(lifter_out["u_h36m_norm"][idx][None]).float().to(device)
            meta_np = raw_2d_metadata(lifter_out["u_h36m_px"][idx], w, h)[None]
            meta = torch.from_numpy(meta_np).float().to(device)
            z_features = torch.zeros((1, z_feature_dim), dtype=torch.float32, device=device)
            ray_features = None
            if geometry["ray_features"] is not None:
                ray_features = torch.from_numpy(geometry["ray_features"][idx][None]).float().to(device)
            reliability_features = None
            if geometry.get("reliability_features") is not None:
                reliability_features = torch.from_numpy(geometry["reliability_features"][idx][None]).float().to(device)
            x_geo = None
            x_geo_features = None
            if geometry["x_geo_used"] is not None:
                x_geo = torch.from_numpy(geometry["x_geo_used"][idx][None]).float().to(device)
                x_geo_features = corrector_pose_input(x_geo, config)
            feat = build_features(
                corrector_y_input(y, config),
                u,
                meta,
                z_features,
                config["corrector_inputs"],
                ray_features=ray_features,
                x_geo_features=x_geo_features,
                reliability_features=reliability_features,
            )
            if feat.shape[-1] != input_dim:
                raise ValueError(f"Checkpoint input_dim={input_dim}, official features have dim={feat.shape[-1]}")
            model_output = model(feat)
            base_pred = compose_prediction(y, _with_scaled_residual(model_output, 0.0), config, x_geo=x_geo)
            full_pred = compose_prediction(y, model_output, config, x_geo=x_geo)
            pred = base_pred if base_only else full_pred
            x_base[idx] = base_pred.cpu().numpy()[0].astype(np.float32)
            residual_effect[idx] = (full_pred - base_pred).cpu().numpy()[0].astype(np.float32)
            x_hat[idx] = pred.cpu().numpy()[0].astype(np.float32)
            for alpha in alphas:
                label = _alpha_label(alpha)
                alpha_pred = compose_prediction(y, _with_scaled_residual(model_output, alpha), config, x_geo=x_geo)
                alpha_outputs[label][idx] = alpha_pred.cpu().numpy()[0].astype(np.float32)
            if isinstance(model_output, dict) and model_output.get("gate") is not None:
                gate_np = model_output["gate"].cpu().numpy()
                gate_outs.append(gate_np)
                gate_y_weight[idx] = gate_np[0].astype(np.float32)

    info = {
        "pc_status": "ok",
        "base_only": bool(base_only),
        "residual_alpha_values": alphas,
        "z_features_mode": "zero_disabled" if z_feature_dim > 0 else "omitted",
        "z_feature_dim": int(z_feature_dim),
        "geometry": geometry,
        "gate_stats": _gate_stats(gate_outs),
        "x_hat_h36m_m": x_hat,
        "x_base_h36m_m": x_base,
        "residual_h36m_m": residual_effect,
        "gate_y_weight": gate_y_weight,
        "residual_alpha_h36m_m": alpha_outputs,
    }
    return _official_from_h36m_m(x_hat), info


def _write_report(path, metrics, manifest):
    lines = [
        "# Official MPI-INF-3DHP GT-2D Evaluation",
        "",
        f"Dataset SHA256: `{manifest['dataset_hash']}`",
        f"Config SHA256: `{manifest['config_hash']}`",
        f"Official root index: `{manifest['official_root_index']}`",
        f"No oracle z: `{manifest['no_oracle_z']}`",
        f"Intrinsics mode: `{manifest.get('intrinsics_mode', 'not_used')}`",
        f"X_geo fit mode: `{manifest.get('xgeo_fit_mode', 'not_used')}`",
        f"Coordinate mode: `{manifest.get('coordinate_mode', 'not_used')}`",
        f"X_geo ablation: `{manifest.get('xgeo_ablation', 'none')}`",
        f"Base-only: `{manifest.get('base_only', False)}`",
        f"Residual alphas: `{manifest.get('residual_alpha_values', [])}`",
        "",
        "PA-MPJPE is diagnostic Procrustes-aligned MPJPE, not part of the official 3DHP PCK/AUC protocol.",
        "",
        "| Method | MPJPE mm | PA-MPJPE mm | PCK@150 | AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    aliases = {"perspective_corrected", "pagrc"}
    for name, vals in metrics["methods"].items():
        if name in aliases and "x_hat" in metrics["methods"]:
            continue
        overall = vals["overall"]
        lines.append(
            f"| {name} | {overall['mpjpe_mm']:.3f} | {overall['pa_mpjpe_mm']:.3f} | "
            f"{overall['pck150']:.3f} | {overall['auc']:.3f} |"
        )
    if manifest.get("pc_status"):
        lines.extend(["", f"Pc status: `{manifest['pc_status']}`"])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--pc-checkpoint", type=Path, default=None)
    parser.add_argument("--disable-z", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--residual-alpha", type=float, nargs="+", default=[])
    parser.add_argument("--xgeo-ablation", choices=["none", "y_lifted", "zero"], default=None)
    parser.add_argument("--xgeo-fit-mode", choices=["free_depth", "closest_y"], default="closest_y")
    parser.add_argument("--xgeo-depth-prior-mode", choices=["bbox", "constant"], default="bbox")
    parser.add_argument("--xgeo-root-prior-weight", type=float, default=1.0)
    parser.add_argument("--xgeo-depth-ridge-weight", type=float, default=0.01)
    parser.add_argument("--xgeo-min-depth-mm", type=float, default=500.0)
    parser.add_argument("--xgeo-max-depth-mm", type=float, default=10000.0)
    parser.add_argument("--test-root", type=Path, default=Path("data/raw/mpi_inf_3dhp/mpi_inf_3dhp_test_set"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config_overrides = {}
    if args.xgeo_ablation is not None:
        config.setdefault("geometry_refinement", {})
        config["geometry_refinement"]["xgeo_ablation"] = args.xgeo_ablation
        config_overrides["geometry_refinement.xgeo_ablation"] = args.xgeo_ablation
    xgeo_fit_options = {
        "mode": args.xgeo_fit_mode,
        "depth_prior_mode": args.xgeo_depth_prior_mode,
        "root_prior_weight": float(args.xgeo_root_prior_weight),
        "depth_ridge_weight": float(args.xgeo_depth_ridge_weight),
        "min_depth_mm": float(args.xgeo_min_depth_mm),
        "max_depth_mm": float(args.xgeo_max_depth_mm),
    }
    config.setdefault("lifter", {})
    config["lifter"]["alignment"] = "none"

    data = _load_npz(args.dataset)
    args.output.mkdir(parents=True, exist_ok=True)

    lifter_out = _run_lifter_for_official_gt2d(data, config)
    gt_mm = data["x_gt3d_univ_mm"].astype(np.float32)

    metrics = {
        "methods": {
            "frozen_lifter": compute_official_metrics(
                lifter_out["y_official_mm"],
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
        }
    }

    pc_pred = None
    pc_status = "baseline_only"
    pc_info = {}
    if not args.baseline_only:
        pc_pred, pc_info = _run_pc_if_supported(
            data,
            config,
            args.pc_checkpoint,
            lifter_out,
            args.disable_z,
            args.output,
            args.test_root,
            base_only=args.base_only,
            residual_alphas=args.residual_alpha,
            xgeo_fit_options=xgeo_fit_options,
        )
        pc_status = pc_info["pc_status"]

        geom = pc_info.get("geometry", {})
        if geom.get("x_geo_camera_abs") is not None:
            xgeo_camera_abs_official = _official_from_h36m_m(geom["x_geo_camera_abs"])
            metrics["methods"]["x_geo_camera_abs"] = compute_official_metrics(
                xgeo_camera_abs_official,
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
        if geom.get("x_geo") is not None:
            xgeo_raw_official = _official_from_h36m_m(geom["x_geo"])
            metrics["methods"]["x_geo_raw"] = compute_official_metrics(
                xgeo_raw_official,
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
        if geom.get("x_geo_used") is not None:
            xgeo_used_official = _official_from_h36m_m(geom["x_geo_used"])
            metrics["methods"]["x_geo_used"] = compute_official_metrics(
                xgeo_used_official,
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
        if pc_info.get("x_base_h36m_m") is not None:
            metrics["methods"]["x_base"] = compute_official_metrics(
                _official_from_h36m_m(pc_info["x_base_h36m_m"]),
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
        for label, alpha_pred_h36m in pc_info.get("residual_alpha_h36m_m", {}).items():
            metrics["methods"][label] = compute_official_metrics(
                _official_from_h36m_m(alpha_pred_h36m),
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
        if pc_pred is not None:
            corr_metrics = compute_official_metrics(
                pc_pred,
                gt_mm,
                data["sequence_names"],
                data["activity_labels"],
            )
            metrics["methods"]["x_hat"] = corr_metrics
            metrics["methods"]["perspective_corrected"] = corr_metrics
            metrics["methods"]["pagrc"] = corr_metrics

    source_config_hash = sha256_file(args.config) if args.config.exists() else None
    effective_config_hash = _json_hash(config)
    config_hash = effective_config_hash if config_overrides else (source_config_hash or effective_config_hash)

    manifest = {
        "dataset": str(args.dataset),
        "dataset_hash": sha256_file(args.dataset),
        "config": str(args.config),
        "config_hash": config_hash,
        "source_config_hash": source_config_hash,
        "effective_config_hash": effective_config_hash,
        "config_overrides": config_overrides,
        "output_dir": str(args.output),
        "official_root_index": OFFICIAL_ROOT_INDEX,
        "official_root_index_matlab": 15,
        "pck_thresholds_mm": OFFICIAL_THRESHOLDS_MM.astype(int).tolist(),
        "joint_groups": official_joint_groups(),
        "activity_names": official_activity_names(),
        "no_oracle_z": True,
        "input_2d_source": "gt2d_official",
        "pc_checkpoint": str(args.pc_checkpoint) if args.pc_checkpoint else None,
        "pc_status": pc_status,
        "base_only": bool(args.base_only),
        "residual_alpha_values": [float(x) for x in args.residual_alpha],
        "xgeo_ablation": config.get("geometry_refinement", {}).get("xgeo_ablation", "none"),
        "xgeo_fit_mode": pc_info.get("geometry", {}).get("xgeo_fit_mode", "not_used"),
        "xgeo_fit_options": xgeo_fit_options,
        "coordinate_mode": pc_info.get("geometry", {}).get("coordinate_mode", "not_used"),
        "z_features_mode": pc_info.get("z_features_mode", "not_used"),
        "z_feature_dim": pc_info.get("z_feature_dim", 0),
        "intrinsics_mode": pc_info.get("geometry", {}).get("intrinsics_mode", "not_used"),
        "x_geo_fit_stats": pc_info.get("geometry", {}).get("x_geo_fit_stats"),
        "xgeo_ablation_info": pc_info.get("geometry", {}).get("xgeo_ablation_info"),
    }

    metrics.update({
        "dataset_hash": manifest["dataset_hash"],
        "config_hash": manifest["config_hash"],
        "official_root_index": OFFICIAL_ROOT_INDEX,
        "pck_thresholds_mm": manifest["pck_thresholds_mm"],
        "no_oracle_z": True,
        "input_2d_source": "gt2d_official",
        "pc_status": pc_status,
        "base_only": bool(args.base_only),
        "residual_alpha_values": [float(x) for x in args.residual_alpha],
        "xgeo_ablation": manifest["xgeo_ablation"],
        "xgeo_fit_mode": manifest["xgeo_fit_mode"],
        "coordinate_mode": manifest["coordinate_mode"],
        "z_features_mode": manifest["z_features_mode"],
        "z_feature_dim": manifest["z_feature_dim"],
        "intrinsics_mode": manifest["intrinsics_mode"],
    })
    if pc_info.get("gate_stats"):
        metrics.update(pc_info["gate_stats"])
    if pc_info.get("gate_y_weight") is not None and np.asarray(pc_info.get("gate_y_weight")).size:
        gate_arr = np.asarray(pc_info["gate_y_weight"], dtype=np.float64)
        if gate_arr.ndim == 4:
            gate_by_joint = gate_arr.mean(axis=(0, 1, 3))
        elif gate_arr.ndim == 3:
            gate_by_joint = gate_arr.mean(axis=(0, 2))
        else:
            gate_by_joint = gate_arr.reshape(-1)
        metrics["gate_y_weight_by_h36m_joint"] = {
            str(i): float(v) for i, v in enumerate(gate_by_joint)
        }
    geom = pc_info.get("geometry", {})
    if geom:
        metrics["official_intrinsics"] = geom.get("intrinsics", {})
        metrics["x_geo_fit_stats"] = geom.get("x_geo_fit_stats", {})
        metrics["xgeo_ablation_info"] = geom.get("xgeo_ablation_info", {})
        fit_stats = geom.get("x_geo_fit_stats", {})
        metrics.update({
            "xgeo_mean_depth_mm": fit_stats.get("mean_depth_mm"),
            "xgeo_depth_prior_mean_mm": fit_stats.get("depth_prior_mean_mm"),
            "xgeo_invalid_depth_fraction": fit_stats.get("invalid_depth_fraction"),
        })
    frozen = metrics["methods"]["frozen_lifter"]["overall"]
    metrics.update({
        "y_mpjpe_mm": frozen["mpjpe_mm"],
        "y_pa_mpjpe_mm": frozen["pa_mpjpe_mm"],
        "y_pck150": frozen["pck150"],
        "y_auc": frozen["auc"],
    })
    if "x_hat" in metrics["methods"]:
        corr = metrics["methods"]["x_hat"]["overall"]
        metrics.update({
            "corrected_mpjpe_mm": corr["mpjpe_mm"],
            "corrected_pa_mpjpe_mm": corr["pa_mpjpe_mm"],
            "corrected_pck150": corr["pck150"],
            "corrected_auc": corr["auc"],
        })
    if "x_base" in metrics["methods"]:
        base = metrics["methods"]["x_base"]["overall"]
        metrics.update({
            "xbase_mpjpe_mm": base["mpjpe_mm"],
            "xbase_pa_mpjpe_mm": base["pa_mpjpe_mm"],
            "xbase_pck150": base["pck150"],
            "xbase_auc": base["auc"],
        })
    if "x_geo_camera_abs" in metrics["methods"]:
        cam_abs = metrics["methods"]["x_geo_camera_abs"]["overall"]
        metrics.update({
            "xgeo_camera_abs_mpjpe_mm": cam_abs["mpjpe_mm"],
            "xgeo_camera_abs_pa_mpjpe_mm": cam_abs["pa_mpjpe_mm"],
            "xgeo_camera_abs_pck150": cam_abs["pck150"],
            "xgeo_camera_abs_auc": cam_abs["auc"],
        })
    alpha_metrics = {}
    for label in pc_info.get("residual_alpha_h36m_m", {}):
        overall = metrics["methods"][label]["overall"]
        alpha_metrics[label] = {
            "mpjpe_mm": overall["mpjpe_mm"],
            "pa_mpjpe_mm": overall["pa_mpjpe_mm"],
            "pck150": overall["pck150"],
            "auc": overall["auc"],
        }
    if alpha_metrics:
        metrics["residual_alpha_metrics"] = alpha_metrics
    if "x_geo_raw" in metrics["methods"]:
        raw = metrics["methods"]["x_geo_raw"]["overall"]
        metrics.update({
            "xgeo_raw_mpjpe_mm": raw["mpjpe_mm"],
            "xgeo_raw_pa_mpjpe_mm": raw["pa_mpjpe_mm"],
            "xgeo_raw_pck150": raw["pck150"],
            "xgeo_raw_auc": raw["auc"],
        })
    if "x_geo_used" in metrics["methods"]:
        used = metrics["methods"]["x_geo_used"]["overall"]
        metrics.update({
            "xgeo_used_mpjpe_mm": used["mpjpe_mm"],
            "xgeo_used_pa_mpjpe_mm": used["pa_mpjpe_mm"],
            "xgeo_used_pck150": used["pck150"],
            "xgeo_used_auc": used["auc"],
        })

    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "official_eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(args.output / "report.md", metrics, manifest)

    x_geo_raw_official_mm = (
        _official_from_h36m_m(pc_info.get("geometry", {}).get("x_geo"))
        if pc_info.get("geometry", {}).get("x_geo") is not None else np.array([], dtype=np.float32)
    )
    x_geo_camera_abs_official_mm = (
        _official_from_h36m_m(pc_info.get("geometry", {}).get("x_geo_camera_abs"))
        if pc_info.get("geometry", {}).get("x_geo_camera_abs") is not None else np.array([], dtype=np.float32)
    )
    x_geo_used_official_mm = (
        _official_from_h36m_m(pc_info.get("geometry", {}).get("x_geo_used"))
        if pc_info.get("geometry", {}).get("x_geo_used") is not None else np.array([], dtype=np.float32)
    )
    gated_base_official_mm = (
        _official_from_h36m_m(pc_info.get("x_base_h36m_m"))
        if pc_info.get("x_base_h36m_m") is not None else np.array([], dtype=np.float32)
    )
    residual_official_mm = (
        _official_from_h36m_m(pc_info.get("residual_h36m_m"))
        if pc_info.get("residual_h36m_m") is not None else np.array([], dtype=np.float32)
    )
    alpha_labels = list(pc_info.get("residual_alpha_h36m_m", {}).keys())
    if alpha_labels:
        residual_alpha_official_mm = np.stack(
            [_official_from_h36m_m(pc_info["residual_alpha_h36m_m"][label]) for label in alpha_labels],
            axis=0,
        ).astype(np.float32)
    else:
        residual_alpha_official_mm = np.zeros((0,) + gt_mm.shape, dtype=np.float32)

    np.savez_compressed(
        args.output / "predictions.npz",
        y_official_mm=lifter_out["y_official_mm"].astype(np.float32),
        y_h36m_m=lifter_out["y_h36m_m"].astype(np.float32),
        pc_official_mm=pc_pred.astype(np.float32) if pc_pred is not None else np.array([], dtype=np.float32),
        x_geo_camera_abs_official_mm=x_geo_camera_abs_official_mm,
        x_geo_raw_official_mm=x_geo_raw_official_mm,
        x_geo_used_official_mm=x_geo_used_official_mm,
        x_base_official_mm=gated_base_official_mm,
        gated_base_official_mm=gated_base_official_mm,
        residual_official_mm=residual_official_mm,
        gate_y_weight=pc_info.get("gate_y_weight", np.array([], dtype=np.float32)),
        xgeo_depths_mm=(
            pc_info.get("geometry", {}).get("xgeo_depths_mm")
            if pc_info.get("geometry", {}).get("xgeo_depths_mm") is not None else np.array([], dtype=np.float32)
        ),
        xgeo_depth_prior_mm=(
            pc_info.get("geometry", {}).get("xgeo_depth_prior_mm")
            if pc_info.get("geometry", {}).get("xgeo_depth_prior_mm") is not None else np.array([], dtype=np.float32)
        ),
        xgeo_fit_mode=np.asarray(metrics.get("xgeo_fit_mode", "not_used")),
        residual_alpha_values=np.asarray(pc_info.get("residual_alpha_values", []), dtype=np.float32),
        residual_alpha_labels=np.asarray(alpha_labels, dtype="<U64"),
        residual_alpha_official_mm=residual_alpha_official_mm,
        x_gt3d_univ_mm=gt_mm.astype(np.float32),
        sequence_names=data["sequence_names"],
        frame_indices=data["frame_indices"],
        activity_labels=data["activity_labels"],
        u_h36m_norm=lifter_out["u_h36m_norm"].astype(np.float32),
    )

    print(json.dumps({
        "frozen_lifter_mpjpe_mm": metrics["methods"]["frozen_lifter"]["overall"]["mpjpe_mm"],
        "frozen_lifter_pa_mpjpe_mm": metrics["methods"]["frozen_lifter"]["overall"]["pa_mpjpe_mm"],
        "frozen_lifter_pck150": metrics["methods"]["frozen_lifter"]["overall"]["pck150"],
        "frozen_lifter_auc": metrics["methods"]["frozen_lifter"]["overall"]["auc"],
        "pc_status": pc_status,
        "corrected_mpjpe_mm": metrics.get("corrected_mpjpe_mm"),
        "xbase_mpjpe_mm": metrics.get("xbase_mpjpe_mm"),
        "intrinsics_mode": metrics.get("intrinsics_mode"),
        "metrics_path": str(args.output / "metrics.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
