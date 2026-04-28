from pathlib import Path
import cv2
import numpy as np

VIDEO_PATH = Path("data/raw/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi")
KPTS_PATH = Path("outputs/rtmlib_mpi_clip/video0_first243_rtmpose.npz")
OUT_DIR = Path("outputs/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "video0_first243_2d_rtmpose.mp4"

SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 5), (0, 6),
]

KPT_THR = 0.25
OUT_SIZE = 1024

data = np.load(KPTS_PATH)
keypoints = data["keypoints"]
scores = data["scores"]
frame_indices = data["frame_indices"]

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

writer = cv2.VideoWriter(
    str(OUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (OUT_SIZE, OUT_SIZE),
)

if not writer.isOpened():
    raise RuntimeError(f"Could not open video writer for {OUT_PATH}")

for i, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        print(f"Could not read frame {frame_idx}")
        continue

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (OUT_SIZE, OUT_SIZE))
    sx = OUT_SIZE / w
    sy = OUT_SIZE / h

    kpts = keypoints[i].copy()
    kpts[:, 0] *= sx
    kpts[:, 1] *= sy
    sc = scores[i]

    for a, b in SKELETON:
        if sc[a] >= KPT_THR and sc[b] >= KPT_THR:
            pa = tuple(kpts[a].astype(int))
            pb = tuple(kpts[b].astype(int))
            cv2.line(frame, pa, pb, (0, 255, 0), 3)

    for j, (x, y) in enumerate(kpts):
        if sc[j] >= KPT_THR:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    cv2.putText(
        frame,
        f"RTMPose 2D | frame {int(frame_idx)}",
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)

    if i % 25 == 0:
        print(f"wrote frame {i}/{len(frame_indices)}")

cap.release()
writer.release()

print("saved:", OUT_PATH)
