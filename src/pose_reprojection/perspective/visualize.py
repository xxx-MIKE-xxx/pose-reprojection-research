from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .skeleton import H36M17_BONES, H36M17_NAMES
from .metrics import root_center, per_joint_error


def _draw_2d(ax, pts, title):
    ax.set_title(title)
    ax.scatter(pts[:, 0], pts[:, 1], s=16)
    for a, b in H36M17_BONES:
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], linewidth=2)
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _draw_3d(ax, pts, title):
    ax.set_title(title)
    pts = pts.copy()
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], s=16)
    for a, b in H36M17_BONES:
        ax.plot(
            [pts[a, 0], pts[b, 0]],
            [pts[a, 2], pts[b, 2]],
            [-pts[a, 1], -pts[b, 1]],
            linewidth=2,
        )
    radius = max(float(np.max(np.abs(pts))), 0.5)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def visualize_outputs(config, out_dir):
    out_dir = Path(out_dir)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "test_predictions.npz"
    if not pred_path.exists():
        print(f"[visualize] missing {pred_path}; skipping")
        return

    data = np.load(pred_path)
    frame = int(config["visualization"].get("frame_index", 0))
    frame = min(frame, data["x_gt"].shape[1] - 1)
    seq = 0

    gt = root_center(data["x_gt"])[seq, frame]
    y = root_center(data["y_lifted"])[seq, frame]
    xhat = root_center(data["x_hat"])[seq, frame]
    u = data["u_px"][seq, frame]
    ucanon = data["canonical_2d_px"][seq, frame]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    _draw_2d(axes[0], ucanon, "Synthetic canonical 2D projection")
    _draw_2d(axes[1], u, "Synthetic virtual-camera 2D projection")
    fig.tight_layout()
    fig.savefig(vis_dir / "synthetic_2d_canonical_vs_virtual.png")
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5), dpi=140)
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")
    _draw_3d(ax1, gt, "GT 3D")
    _draw_3d(ax2, y, "Frozen lifter Y")
    _draw_3d(ax3, xhat, "Pc corrected X_hat")
    fig.tight_layout()
    fig.savefig(vis_dir / "3d_gt_vs_lifter_vs_corrected.png")
    plt.close(fig)

    base_err = per_joint_error(root_center(data["y_lifted"]), root_center(data["x_gt"])).mean(axis=(0, 1)) * 1000.0
    corr_err = per_joint_error(root_center(data["x_hat"]), root_center(data["x_gt"])).mean(axis=(0, 1)) * 1000.0

    x = np.arange(len(H36M17_NAMES))
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    ax.plot(x, base_err, marker="o", label="Frozen lifter")
    ax.plot(x, corr_err, marker="o", label="Pc corrected")
    ax.set_xticks(x)
    ax.set_xticklabels(H36M17_NAMES, rotation=60, ha="right")
    ax.set_ylabel("Mean per-joint error, mm")
    ax.set_title("Per-joint error")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(vis_dir / "per_joint_error.png")
    plt.close(fig)

    _plot_error_by_camera_variable(data, vis_dir)


def _bin_plot(values, base_err, corr_err, xlabel, out_path, bins=8):
    values = np.asarray(values)
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(values, qs))
    if len(edges) < 3:
        return

    centers = []
    base = []
    corr = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (values >= lo) & (values <= hi)
        if not np.any(mask):
            continue
        centers.append((lo + hi) / 2.0)
        base.append(float(base_err[mask].mean()))
        corr.append(float(corr_err[mask].mean()))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.plot(centers, base, marker="o", label="Frozen lifter")
    ax.plot(centers, corr, marker="o", label="Pc corrected")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MPJPE, mm")
    ax.set_title(f"MPJPE({xlabel})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_error_by_camera_variable(data, vis_dir):
    gt = root_center(data["x_gt"])
    y = root_center(data["y_lifted"])
    xhat = root_center(data["x_hat"])

    base_seq = np.linalg.norm(y - gt, axis=-1).mean(axis=(1, 2)) * 1000.0
    corr_seq = np.linalg.norm(xhat - gt, axis=-1).mean(axis=(1, 2)) * 1000.0
    z = data["z"]

    specs = [
        (z[:, 0], "camera_distance_m", "mpjpe_by_camera_distance.png"),
        (z[:, 1], "camera_height_m", "mpjpe_by_camera_height.png"),
        (np.abs(z[:, 2]), "abs_yaw_deg", "mpjpe_by_abs_yaw.png"),
        (np.abs(z[:, 3]), "abs_pitch_deg", "mpjpe_by_abs_pitch.png"),
        (z[:, 5], "fov_deg", "mpjpe_by_fov.png"),
    ]
    for values, label, name in specs:
        _bin_plot(values, base_seq, corr_seq, label, vis_dir / name)
