import math
import numpy as np
import torch


CAMERA_PARAM_NAMES = [
    "distance_m", "height_m", "yaw_deg", "pitch_deg", "roll_deg", "fov_deg",
    "image_width", "image_height", "focal_px",
]


def focal_from_fov(width, fov_deg):
    fov_rad = np.deg2rad(fov_deg)
    return 0.5 * float(width) / np.tan(0.5 * fov_rad)


def _rot_x(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def rotation_from_camera_params(params):
    return _rot_z(params["roll_deg"]) @ _rot_x(params["pitch_deg"]) @ _rot_y(params["yaw_deg"])


def sample_camera_params(rng, ranges, image_width, image_height):
    p = {}
    for key, bounds in ranges.items():
        lo, hi = bounds
        p[key] = float(rng.uniform(float(lo), float(hi)))
    p["image_width"] = int(image_width)
    p["image_height"] = int(image_height)
    p["focal_px"] = float(focal_from_fov(image_width, p["fov_deg"]))
    return p


def canonical_camera_params(canonical, image_width, image_height):
    p = dict(canonical)
    p["image_width"] = int(image_width)
    p["image_height"] = int(image_height)
    p["focal_px"] = float(focal_from_fov(image_width, p["fov_deg"]))
    return p


def camera_params_to_vector(params):
    return np.array([params[name] for name in CAMERA_PARAM_NAMES], dtype=np.float32)


def vector_to_camera_params(vec):
    return {name: float(vec[i]) for i, name in enumerate(CAMERA_PARAM_NAMES)}


def _intrinsics_from_params(params):
    w = float(params["image_width"])
    h = float(params["image_height"])
    f = float(params["focal_px"])
    return np.array(
        [[f, 0.0, 0.5 * w], [0.0, f, 0.5 * h], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project_np(x_world, params, return_camera=False):
    """Project root-centered 3D skeletons to pixels.

    x_world: (..., 17, 3), meters.
    camera looks along +Z. Image v uses upward-positive body y, so v = cy - f*y/z.
    """
    x = np.asarray(x_world, dtype=np.float32)
    R = rotation_from_camera_params(params)
    flat = x.reshape(-1, 3)
    cam = flat @ R.T
    cam = cam.reshape(x.shape)

    t = np.array([0.0, -float(params["height_m"]), float(params["distance_m"])], dtype=np.float32)
    cam = cam.copy()
    cam += t

    z = np.maximum(cam[..., 2], 1e-3)
    w = float(params["image_width"])
    h = float(params["image_height"])
    f = float(params["focal_px"])
    u = f * (cam[..., 0] / z) + 0.5 * w
    v = -f * (cam[..., 1] / z) + 0.5 * h
    pixels = np.stack([u, v], axis=-1).astype(np.float32)
    cam = cam.astype(np.float32)
    if not return_camera:
        return pixels, cam
    return pixels, cam, {
        "R_world_to_camera": R.astype(np.float32),
        "K": _intrinsics_from_params(params),
        "t_world_to_camera": t.astype(np.float32),
    }


def normalize_screen_coordinates_np(u, w, h):
    u = np.asarray(u, dtype=np.float32)
    return u / float(w) * 2.0 - np.array([1.0, float(h) / float(w)], dtype=np.float32)


def unnormalize_screen_coordinates_np(u_norm, w, h):
    u_norm = np.asarray(u_norm, dtype=np.float32)
    return (u_norm + np.array([1.0, float(h) / float(w)], dtype=np.float32)) * float(w) / 2.0


def raw_2d_metadata(u_px, image_width, image_height):
    u = np.asarray(u_px, dtype=np.float32)
    mn = np.nanmin(u, axis=-2)
    mx = np.nanmax(u, axis=-2)
    center = (mn + mx) * 0.5
    size = np.maximum(mx - mn, 1e-6)

    w = float(image_width)
    h = float(image_height)
    meta = np.stack([
        center[..., 0] / w,
        center[..., 1] / h,
        size[..., 0] / w,
        size[..., 1] / h,
        (size[..., 0] * size[..., 1]) / (w * h),
        np.full(center.shape[:-1], w / max(h, 1.0), dtype=np.float32),
        np.maximum(size[..., 0] / w, size[..., 1] / h),
    ], axis=-1)
    return meta.astype(np.float32)


def _torch_rotation(yaw, pitch, roll, device, dtype):
    # yaw/pitch/roll shape: (B,)
    def mat_y(a):
        c, s = torch.cos(a), torch.sin(a)
        z = torch.zeros_like(a)
        o = torch.ones_like(a)
        return torch.stack([
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ], dim=-2)

    def mat_x(a):
        c, s = torch.cos(a), torch.sin(a)
        z = torch.zeros_like(a)
        o = torch.ones_like(a)
        return torch.stack([
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ], dim=-2)

    def mat_z(a):
        c, s = torch.cos(a), torch.sin(a)
        z = torch.zeros_like(a)
        o = torch.ones_like(a)
        return torch.stack([
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ], dim=-2)

    return mat_z(roll) @ mat_x(pitch) @ mat_y(yaw)


def project_torch(x_world, z_vec):
    """Differentiable projection.

    x_world: (B,T,J,3)
    z_vec: (B,D), D follows CAMERA_PARAM_NAMES.
    returns pixels: (B,T,J,2)
    """
    distance = z_vec[:, 0]
    height = z_vec[:, 1]
    yaw = z_vec[:, 2] * math.pi / 180.0
    pitch = z_vec[:, 3] * math.pi / 180.0
    roll = z_vec[:, 4] * math.pi / 180.0
    width = z_vec[:, 6]
    height_img = z_vec[:, 7]
    focal = z_vec[:, 8]

    R = _torch_rotation(yaw, pitch, roll, x_world.device, x_world.dtype)
    cam = torch.einsum("btjc,bkc->btjk", x_world, R)

    cam_y = cam[..., 1] - height[:, None, None]
    cam_z = torch.clamp(cam[..., 2] + distance[:, None, None], min=1e-3)
    cam_x = cam[..., 0]

    u = focal[:, None, None] * (cam_x / cam_z) + 0.5 * width[:, None, None]
    v = -focal[:, None, None] * (cam_y / cam_z) + 0.5 * height_img[:, None, None]
    return torch.stack([u, v], dim=-1)
