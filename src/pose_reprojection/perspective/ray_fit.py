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
    if camera_params is None:
        raise ValueError("camera_params/z are required to convert ray-depth camera points back to world coordinates")

    z_np = np.asarray(camera_params, dtype=np.float32)
    u_np = None if u_px is None else np.asarray(u_px, dtype=np.float32)
    conf_np = None if confidence is None else np.asarray(confidence, dtype=np.float32)

    n = Y_np.shape[0]
    out_chunks = []
    stat_rows = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        y = torch.from_numpy(Y_np[start:end]).to(dev)
        r_img = torch.from_numpy(rays_np[start:end]).to(dev)
        z = torch.from_numpy(z_np[start:end]).to(dev)
        r_cam = _camera_rays_from_image_rays(r_img)
        r_cam = r_cam / torch.linalg.norm(r_cam, dim=-1, keepdim=True).clamp_min(1e-8)

        with torch.no_grad():
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
            x_geo = _camera_to_world(cam, z)
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
            x_geo = _camera_to_world(cam, z)
            if u_np is not None:
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
        "num_sequences": int(n),
        "config_hash_material": _config_hashable(refine_cfg),
    }
    for key in keys:
        vals = [row[key] for row in stat_rows if key in row]
        stats[f"mean_{key}"] = float(np.mean(vals)) if vals else None

    return x_geo_np, stats
