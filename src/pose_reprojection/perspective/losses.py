import torch
import torch.nn.functional as F

from .camera import project_torch
from .skeleton import H36M17_BONES


def mpjpe_loss(x_hat, x_gt):
    return torch.linalg.norm(x_hat - x_gt, dim=-1).mean()


def reprojection_loss(x_hat, u_px, z_vec, unit="normalized_image"):
    proj = project_torch(x_hat, z_vec)
    diff = proj - u_px

    if unit == "normalized_image":
        # Normalize by image diagonal per sequence.
        w = z_vec[:, 6]
        h = z_vec[:, 7]
        diag = torch.sqrt(w ** 2 + h ** 2).clamp_min(1.0)
        diff = diff / diag[:, None, None, None]
    elif unit == "pixels":
        pass
    else:
        raise ValueError(f"Unknown reprojection loss unit: {unit}")

    return torch.linalg.norm(diff, dim=-1).mean()


def bone_length_loss(x_hat, x_gt, bones=H36M17_BONES):
    vals = []
    for a, b in bones:
        len_hat = torch.linalg.norm(x_hat[:, :, a] - x_hat[:, :, b], dim=-1)
        len_gt = torch.linalg.norm(x_gt[:, :, a] - x_gt[:, :, b], dim=-1)
        vals.append(torch.abs(len_hat - len_gt).mean())
    return torch.stack(vals).mean()


def temporal_loss(x_hat, x_gt=None, mode="match_gt_accel"):
    if x_hat.shape[1] < 3:
        return x_hat.new_tensor(0.0)

    acc_hat = x_hat[:, 2:] - 2.0 * x_hat[:, 1:-1] + x_hat[:, :-2]

    if mode == "smoothness":
        return torch.linalg.norm(acc_hat, dim=-1).mean()

    if mode == "match_gt_accel":
        if x_gt is None:
            raise ValueError("x_gt required for match_gt_accel temporal loss")
        acc_gt = x_gt[:, 2:] - 2.0 * x_gt[:, 1:-1] + x_gt[:, :-2]
        return torch.linalg.norm(acc_hat - acc_gt, dim=-1).mean()

    raise ValueError(f"Unknown temporal loss mode: {mode}")


def compute_losses(x_hat, x_gt, u_px, z_vec, loss_cfg):
    out = {}
    total = x_hat.new_tensor(0.0)

    w = float(loss_cfg.get("mpjpe", {}).get("weight", 0.0))
    if w:
        val = mpjpe_loss(x_hat, x_gt)
        out["mpjpe"] = val
        total = total + w * val

    r_cfg = loss_cfg.get("reprojection", {})
    w = float(r_cfg.get("weight", 0.0))
    if w:
        val = reprojection_loss(x_hat, u_px, z_vec, unit=r_cfg.get("unit", "normalized_image"))
        out["reprojection"] = val
        total = total + w * val

    b_cfg = loss_cfg.get("bone", {})
    w = float(b_cfg.get("weight", 0.0))
    if w:
        val = bone_length_loss(x_hat, x_gt)
        out["bone"] = val
        total = total + w * val

    t_cfg = loss_cfg.get("temporal", {})
    w = float(t_cfg.get("weight", 0.0))
    if w:
        val = temporal_loss(x_hat, x_gt=x_gt, mode=t_cfg.get("mode", "match_gt_accel"))
        out["temporal"] = val
        total = total + w * val

    out["total"] = total
    return out
