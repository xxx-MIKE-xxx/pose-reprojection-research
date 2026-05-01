import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from .camera import _torch_rotation, project_torch
from .features import stable_body_scale_torch
from .skeleton import H36M17_BONES


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _camera_rays_from_image_rays(rays):
    rays_cam = rays.clone()
    rays_cam[..., 1] = -rays_cam[..., 1]
    return rays_cam


def _world_to_camera(x_world, z_vec):
    distance = z_vec[:, 0]
    height = z_vec[:, 1]
    yaw = z_vec[:, 2] * math.pi / 180.0
    pitch = z_vec[:, 3] * math.pi / 180.0
    roll = z_vec[:, 4] * math.pi / 180.0
    R = _torch_rotation(yaw, pitch, roll, x_world.device, x_world.dtype)
    cam = torch.einsum("btjc,bkc->btjk", x_world, R)
    cam = cam.clone()
    cam[..., 1] = cam[..., 1] - height[:, None, None]
    cam[..., 2] = cam[..., 2] + distance[:, None, None]
    return cam


def _camera_to_world(x_cam, z_vec):
    distance = z_vec[:, 0]
    height = z_vec[:, 1]
    yaw = z_vec[:, 2] * math.pi / 180.0
    pitch = z_vec[:, 3] * math.pi / 180.0
    roll = z_vec[:, 4] * math.pi / 180.0
    R = _torch_rotation(yaw, pitch, roll, x_cam.device, x_cam.dtype)
    centered = x_cam.clone()
    centered[..., 1] = centered[..., 1] + height[:, None, None]
    centered[..., 2] = centered[..., 2] - distance[:, None, None]
    return torch.einsum("btjc,bcd->btjd", centered, R)


def _inverse_softplus(x):
    x = torch.clamp(x, min=1e-6)
    return x + torch.log(-torch.expm1(-x))


def _normalize_pose(x, root_joint=0, eps=1e-6):
    centered = x - x[:, :, root_joint:root_joint + 1, :]
    scale = stable_body_scale_torch(centered, eps=eps)
    return centered / scale


def _bone_lengths_torch(x):
    vals = []
    for a, b in H36M17_BONES:
        vals.append(torch.linalg.norm(x[:, :, a] - x[:, :, b], dim=-1))
    return torch.stack(vals, dim=-1)


def _acceleration(x):
    if x.shape[1] < 3:
        return x.new_zeros((x.shape[0], 0, x.shape[2], x.shape[3]))
    return x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]


def _config_hashable(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def _image_rays_to_camera_np(rays):
    rays_cam = np.asarray(rays, dtype=np.float64).copy()
    rays_cam[..., 1] = -rays_cam[..., 1]
    norm = np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    return (rays_cam / np.maximum(norm, 1e-8)).astype(np.float64)


def image_rays_to_camera_rays(rays):
    """Convert K^-1[u,v,1] image rays to the y-up synthetic camera frame."""
    return _image_rays_to_camera_np(rays).astype(np.float32)


def _broadcast_intrinsics(intrinsics, n):
    if intrinsics is None:
        return None
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape == (3, 3):
        return np.broadcast_to(K[None], (int(n), 3, 3)).copy()
    if K.shape == (int(n), 3, 3):
        return K
    raise ValueError(f"intrinsics must have shape (3,3) or ({n},3,3), got {K.shape}")


def _depth_prior_from_bbox(y_rel, u_px, intrinsics, min_depth_mm, max_depth_mm):
    n = y_rel.shape[0]
    if u_px is None or intrinsics is None:
        return np.full(n, 4000.0, dtype=np.float64)

    u = np.asarray(u_px, dtype=np.float64)
    K = _broadcast_intrinsics(intrinsics, n)
    if u.shape[:2] != y_rel.shape[:2]:
        raise ValueError(f"u_px shape {u.shape} is incompatible with y_pose shape {y_rel.shape}")

    f = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
    bbox_height_px = np.nanmax(u[..., 1], axis=1) - np.nanmin(u[..., 1], axis=1)
    bbox_height_px = np.maximum(bbox_height_px, 50.0)

    y_extent = np.nanpercentile(y_rel[..., 1], 95, axis=1) - np.nanpercentile(y_rel[..., 1], 5, axis=1)
    xyz_extent = np.nanpercentile(np.linalg.norm(y_rel, axis=-1), 95, axis=1)
    pose_height_mm = np.maximum(np.abs(y_extent), xyz_extent)
    pose_height_mm = np.clip(pose_height_mm, 700.0, 2200.0)

    prior = f * pose_height_mm / bbox_height_px
    return np.clip(prior, float(min_depth_mm), float(max_depth_mm)).astype(np.float64)


def fit_xgeo_closest_to_lifter(
    rays,
    y_pose_mm,
    root_idx=14,
    u_px=None,
    intrinsics=None,
    depth_prior_mode="bbox",
    root_prior_weight=1.0,
    depth_ridge_weight=0.01,
    min_depth_mm=500.0,
    max_depth_mm=10000.0,
):
    """Fit ray depths whose root-relative 3D shape is closest to the lifter pose.

    This is intended for official GT-2D evaluation where no synthetic/world camera
    z is available. It uses only input rays, input 2D/intrinsics for a depth prior,
    and the frozen lifter output. No GT 3D, GT scale, or test bone lengths are used.
    """
    rays_np = np.asarray(rays, dtype=np.float64)
    y_np = np.asarray(y_pose_mm, dtype=np.float64)
    if rays_np.ndim != 3 or rays_np.shape[-1] != 3:
        raise ValueError(f"rays must have shape (N,J,3), got {rays_np.shape}")
    if y_np.shape != rays_np.shape:
        raise ValueError(f"y_pose_mm shape {y_np.shape} must match rays shape {rays_np.shape}")

    n, j, _ = rays_np.shape
    root = int(root_idx)
    if root < 0 or root >= j:
        raise ValueError(f"root_idx={root_idx} is out of range for {j} joints")

    min_depth = float(min_depth_mm)
    max_depth = float(max_depth_mm)
    if not (0.0 < min_depth < max_depth):
        raise ValueError("Expected 0 < min_depth_mm < max_depth_mm")

    # Use the literal K^-1 [u, v, 1] ray convention here. This y-down image-ray
    # frame matches the official GT-2D lifter convention better than the y-up
    # synthetic camera frame used by project_np/project_torch.
    rays_cam = rays_np / np.maximum(np.linalg.norm(rays_np, axis=-1, keepdims=True), 1e-8)
    y_rel = y_np - y_np[:, root:root + 1, :]

    if depth_prior_mode == "bbox":
        depth_prior = _depth_prior_from_bbox(y_rel, u_px, intrinsics, min_depth, max_depth)
    elif depth_prior_mode == "constant":
        depth_prior = np.full(n, 4000.0, dtype=np.float64)
    else:
        raise ValueError(f"Unknown xgeo depth prior mode: {depth_prior_mode}")

    sqrt_root = math.sqrt(max(float(root_prior_weight), 0.0))
    sqrt_ridge = math.sqrt(max(float(depth_ridge_weight), 0.0))
    depths = np.zeros((n, j), dtype=np.float64)
    invalid_before_fallback = np.zeros((n, j), dtype=bool)
    residual_norm = np.zeros(n, dtype=np.float64)

    non_root = [idx for idx in range(j) if idx != root]
    for frame in range(n):
        rows = []
        targets = []
        r_root = rays_cam[frame, root]
        for joint in non_root:
            r_joint = rays_cam[frame, joint]
            for axis in range(3):
                row = np.zeros(j, dtype=np.float64)
                row[joint] = r_joint[axis]
                row[root] = -r_root[axis]
                rows.append(row)
                targets.append(y_rel[frame, joint, axis])

        prior = float(depth_prior[frame])
        if sqrt_root > 0.0:
            row = np.zeros(j, dtype=np.float64)
            row[root] = sqrt_root
            rows.append(row)
            targets.append(sqrt_root * prior)
        if sqrt_ridge > 0.0:
            for joint in range(j):
                row = np.zeros(j, dtype=np.float64)
                row[joint] = sqrt_ridge
                rows.append(row)
                targets.append(sqrt_ridge * prior)

        A = np.stack(rows, axis=0)
        b = np.asarray(targets, dtype=np.float64)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            sol = np.full(j, prior, dtype=np.float64)

        invalid = (~np.isfinite(sol)) | (sol < min_depth) | (sol > max_depth)
        invalid_before_fallback[frame] = invalid
        if np.any(invalid):
            sol = sol.copy()
            sol[invalid] = prior
        sol = np.clip(sol, min_depth, max_depth)
        depths[frame] = sol
        pred_rel = sol[:, None] * rays_cam[frame] - sol[root] * r_root[None, :]
        residual_norm[frame] = float(np.sqrt(np.mean((pred_rel - y_rel[frame]) ** 2)))

    x_abs = depths[..., None] * rays_cam
    x_rootrel = x_abs - x_abs[:, root:root + 1, :]
    x_used = x_rootrel + y_np[:, root:root + 1, :]
    stats = {
        "enabled": True,
        "mode": "closest_y",
        "used_x_gt": False,
        "used_clean_2d": False,
        "coordinate_mode": "root_aligned_to_y",
        "root_idx": int(root),
        "depth_prior_mode": depth_prior_mode,
        "root_prior_weight": float(root_prior_weight),
        "depth_ridge_weight": float(depth_ridge_weight),
        "min_depth_mm": float(min_depth),
        "max_depth_mm": float(max_depth),
        "mean_depth_mm": float(np.mean(depths)),
        "min_depth_observed_mm": float(np.min(depths)),
        "max_depth_observed_mm": float(np.max(depths)),
        "depth_prior_mean_mm": float(np.mean(depth_prior)),
        "depth_prior_min_mm": float(np.min(depth_prior)),
        "depth_prior_max_mm": float(np.max(depth_prior)),
        "invalid_depth_fraction": float(np.mean(invalid_before_fallback)),
        "mean_fit_rmse_mm": float(np.mean(residual_norm)),
        "max_fit_rmse_mm": float(np.max(residual_norm)),
    }
    return {
        "x_geo_camera_abs_mm": x_abs.astype(np.float32),
        "x_geo_rootrel_mm": x_rootrel.astype(np.float32),
        "x_geo_used_mm": x_used.astype(np.float32),
        "depths_mm": depths.astype(np.float32),
        "depth_prior_mm": depth_prior.astype(np.float32),
        "fit_rmse_mm": residual_norm.astype(np.float32),
        "invalid_depth_mask": invalid_before_fallback,
        "stats": stats,
    }


def fit_xgeo_closest_to_lifter_frame_aware(
    rays_camera,
    y_pose_world_mm,
    root_idx=0,
    camera_R_world_to_camera=None,
    **kwargs,
):
    """Fit closest-Y ray depths in camera frame, then return to Y/world frame.

    `project_np` uses row-vector math equivalent to:
      x_camera = x_world @ R_world_to_camera.T + t.
    The wrapper therefore rotates root-relative Y into camera coordinates with
    `@ R.T` and rotates fitted root-relative X_geo back with `@ R`.
    """
    rays_np = np.asarray(rays_camera, dtype=np.float64)
    y_world = np.asarray(y_pose_world_mm, dtype=np.float64)
    if rays_np.ndim != 3 or rays_np.shape[-1] != 3:
        raise ValueError(f"rays_camera must have shape (N,J,3), got {rays_np.shape}")
    if y_world.shape != rays_np.shape:
        raise ValueError(f"y_pose_world_mm shape {y_world.shape} must match rays_camera {rays_np.shape}")

    n, j, _ = rays_np.shape
    root = int(root_idx)
    if root < 0 or root >= j:
        raise ValueError(f"root_idx={root_idx} is out of range for {j} joints")

    if camera_R_world_to_camera is None:
        R = np.broadcast_to(np.eye(3, dtype=np.float64)[None], (n, 3, 3)).copy()
        frame_mode = "identity_unframed"
    else:
        R = np.asarray(camera_R_world_to_camera, dtype=np.float64)
        if R.shape == (3, 3):
            R = np.broadcast_to(R[None], (n, 3, 3)).copy()
        elif R.shape != (n, 3, 3):
            raise ValueError(f"camera_R_world_to_camera must have shape (3,3) or ({n},3,3), got {R.shape}")
        frame_mode = "world_to_camera_to_world"

    y_root_world = y_world[:, root:root + 1, :]
    y_rel_world = y_world - y_root_world
    y_rel_camera = np.einsum("njc,nkc->njk", y_rel_world, R)

    fit = fit_xgeo_closest_to_lifter(
        rays_np,
        y_rel_camera.astype(np.float32),
        root_idx=root,
        **kwargs,
    )

    x_geo_camera_rel = np.asarray(fit["x_geo_rootrel_mm"], dtype=np.float64)
    x_geo_rel_world = np.einsum("njc,nck->njk", x_geo_camera_rel, R)
    x_geo_used_world = x_geo_rel_world + y_root_world

    stats = dict(fit["stats"])
    stats.update({
        "xgeo_fit_mode": "closest_y",
        "xgeo_frame_mode": frame_mode,
        "coordinate_mode": "root_aligned_to_y",
        "frame_aware": camera_R_world_to_camera is not None,
    })
    return {
        "x_geo_camera_abs_mm": fit["x_geo_camera_abs_mm"],
        "x_geo_camera_rel_mm": x_geo_camera_rel.astype(np.float32),
        "x_geo_used_mm": x_geo_used_world.astype(np.float32),
        "depths_mm": fit["depths_mm"],
        "depth_prior_mm": fit["depth_prior_mm"],
        "fit_rmse_mm": fit["fit_rmse_mm"],
        "invalid_depth_mask": fit["invalid_depth_mask"],
        "stats": stats,
    }


def fit_xgeo_from_rays_and_lifter(Y, rays, config, camera_params=None, u_px=None, confidence=None):
    """Fit a non-leaky 3D candidate constrained to input rays.

    The optimizer only sees the frozen lifter output Y, input-derived rays, optional
    confidence weights, and true virtual camera parameters needed to place rays in
    the repo's world coordinate frame. It does not use X_gt or clean 2D.
    """
    refine_cfg = config.get("geometry_refinement", {})
    if refine_cfg.get("mode", "ray_depth_fit") != "ray_depth_fit":
        raise ValueError(f"Unknown geometry_refinement.mode: {refine_cfg.get('mode')}")

    dev = _device(refine_cfg.get("device", config.get("training", {}).get("device", "auto")))
    min_depth = float(refine_cfg.get("min_depth_m", 0.5))
    steps = int(refine_cfg.get("steps", 60))
    lr = float(refine_cfg.get("lr", 0.03))
    batch_size = int(refine_cfg.get("batch_size", 16))
    root_joint = int(refine_cfg.get("root_joint", 0))
    eps = float(refine_cfg.get("eps", 1e-6))
    weights = refine_cfg.get("losses", {})

    Y_np = np.asarray(Y, dtype=np.float32)
    rays_np = np.asarray(rays, dtype=np.float32)
    if Y_np.shape[:3] != rays_np.shape[:3]:
        raise ValueError(f"Y shape {Y_np.shape} and rays shape {rays_np.shape} are incompatible")
    camera_frame_mode = camera_params is None
    z_np = None if camera_frame_mode else np.asarray(camera_params, dtype=np.float32)
    u_np = None if u_px is None else np.asarray(u_px, dtype=np.float32)
    conf_np = None if confidence is None else np.asarray(confidence, dtype=np.float32)

    n = Y_np.shape[0]
    out_chunks = []
    stat_rows = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y = torch.from_numpy(Y_np[start:end]).to(dev)
        r_img = torch.from_numpy(rays_np[start:end]).to(dev)
        z = None if camera_frame_mode else torch.from_numpy(z_np[start:end]).to(dev)
        r_cam = _camera_rays_from_image_rays(r_img)
        r_cam = r_cam / torch.linalg.norm(r_cam, dim=-1, keepdim=True).clamp_min(1e-8)

        with torch.no_grad():
            if camera_frame_mode:
                init_depth = float(refine_cfg.get("init_depth_m", 4.0))
                depth0 = torch.full_like(r_cam[..., 0], init_depth)
            else:
                y_cam = _world_to_camera(y, z)
                depth0 = torch.sum(y_cam * r_cam, dim=-1).clamp_min(min_depth + 0.25)
            raw_depth0 = _inverse_softplus(depth0 - min_depth)

        raw_depth = torch.nn.Parameter(raw_depth0)
        opt = torch.optim.Adam([raw_depth], lr=lr)
        y_norm = _normalize_pose(y, root_joint=root_joint, eps=eps)
        y_bones = _bone_lengths_torch(y_norm).detach()
        y_acc = _acceleration(y_norm).detach()
        conf = None if conf_np is None else torch.from_numpy(conf_np[start:end]).to(dev)

        last = {}
        for _ in range(steps):
            depth = F.softplus(raw_depth) + min_depth
            cam = depth[..., None] * r_cam
            x_geo = cam if camera_frame_mode else _camera_to_world(cam, z)
            x_norm = _normalize_pose(x_geo, root_joint=root_joint, eps=eps)

            pose_err = torch.linalg.norm(x_norm - y_norm, dim=-1)
            if conf is not None:
                pose_prior = (pose_err * conf).sum() / conf.sum().clamp_min(1.0)
            else:
                pose_prior = pose_err.mean()
            bone_y = torch.abs(_bone_lengths_torch(x_norm) - y_bones).mean()

            if x_norm.shape[1] >= 3:
                temporal = torch.linalg.norm(_acceleration(x_norm) - y_acc, dim=-1).mean()
                depth_smooth = torch.abs(depth[:, 1:] - depth[:, :-1]).mean()
            else:
                temporal = x_norm.new_tensor(0.0)
                depth_smooth = x_norm.new_tensor(0.0)
            depth_smooth = depth_smooth + 0.1 * depth.std(unbiased=False)

            loss = (
                float(weights.get("pose_prior", 1.0)) * pose_prior
                + float(weights.get("bone_y", 0.25)) * bone_y
                + float(weights.get("temporal", 0.01)) * temporal
                + float(weights.get("depth_smooth", 0.001)) * depth_smooth
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = {
                "total": float(loss.detach().cpu()),
                "pose_prior": float(pose_prior.detach().cpu()),
                "bone_y": float(bone_y.detach().cpu()),
                "temporal": float(temporal.detach().cpu()),
                "depth_smooth": float(depth_smooth.detach().cpu()),
            }

        with torch.no_grad():
            depth = F.softplus(raw_depth) + min_depth
            cam = depth[..., None] * r_cam
            x_geo = cam if camera_frame_mode else _camera_to_world(cam, z)
            if u_np is not None and not camera_frame_mode:
                u = torch.from_numpy(u_np[start:end]).to(dev)
                reproj = torch.linalg.norm(project_torch(x_geo, z) - u, dim=-1).mean()
                last["reprojection_error_to_input_px"] = float(reproj.cpu())
            last["depth_mean_m"] = float(depth.mean().cpu())
            last["depth_min_m"] = float(depth.min().cpu())
            last["depth_max_m"] = float(depth.max().cpu())
            out_chunks.append(x_geo.cpu().numpy().astype(np.float32))
            stat_rows.append(last)

    x_geo_np = np.concatenate(out_chunks, axis=0).astype(np.float32)
    keys = sorted({k for row in stat_rows for k in row.keys()})
    stats = {
        "enabled": True,
        "mode": "ray_depth_fit",
        "used_x_gt": False,
        "used_clean_2d": False,
        "camera_frame_mode": bool(camera_frame_mode),
        "num_sequences": int(n),
        "config_hash_material": _config_hashable(refine_cfg),
    }
    for key in keys:
        vals = [row[key] for row in stat_rows if key in row]
        stats[f"mean_{key}"] = float(np.mean(vals)) if vals else None

    return x_geo_np, stats
