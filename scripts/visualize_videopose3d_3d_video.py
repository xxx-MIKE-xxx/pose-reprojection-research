from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRED_PATH = Path("outputs/videopose3d_mpi_clip_fullseq/video0_first243_videopose3d_fullseq.npz")
OUT_DIR = Path("outputs/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "video0_first243_3d_videopose3d.mp4"

SKELETON = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

data = np.load(PRED_PATH)
pred = data["pred_3d"]

if pred.ndim == 4:
    pred = pred[0]

if pred.ndim != 3 or pred.shape[1:] != (17, 3):
    raise ValueError(f"Expected pred shape (T, 17, 3), got {pred.shape}")

# Root-center for clearer visualization.
pred = pred.copy()
pred = pred - pred[:, :1, :]

# Convert to plotting coordinates.
# VideoPose3D coords can look different from image coords; this is for visual sanity.
plot_pred = np.zeros_like(pred)
plot_pred[:, :, 0] = pred[:, :, 0]
plot_pred[:, :, 1] = pred[:, :, 2]
plot_pred[:, :, 2] = -pred[:, :, 1]

radius = float(np.max(np.abs(plot_pred)))
radius = max(radius, 0.5)

fps = 25.0
width = 900
height = 900

writer = cv2.VideoWriter(
    str(OUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    raise RuntimeError(f"Could not open writer for {OUT_PATH}")

fig = plt.figure(figsize=(6, 6), dpi=150)
ax = fig.add_subplot(111, projection="3d")

for t in range(plot_pred.shape[0]):
    pts = plot_pred[t]

    ax.clear()
    ax.set_title(f"VideoPose3D 3D skeleton | frame {t}")
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_zlim(-radius, radius)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=15, azim=70)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    for a, b in SKELETON:
        xs = [pts[a, 0], pts[b, 0]]
        ys = [pts[a, 1], pts[b, 1]]
        zs = [pts[a, 2], pts[b, 2]]
        ax.plot(xs, ys, zs, linewidth=3)

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=20)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    frame = cv2.resize(frame, (width, height))
    writer.write(frame)

    if t % 25 == 0:
        print(f"wrote 3D frame {t}/{plot_pred.shape[0]}")

writer.release()
plt.close(fig)

print("saved:", OUT_PATH)
