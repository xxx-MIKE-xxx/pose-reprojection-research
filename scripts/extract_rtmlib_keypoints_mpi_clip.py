from pathlib import Path
import cv2
import numpy as np
import torch
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

from rtmlib import Body

VIDEO_PATH = Path("data/raw/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi")
OUT_DIR = Path("outputs/rtmlib_mpi_clip")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 243
OUT_PATH = OUT_DIR / "video0_first243_rtmpose.npz"

print("torch cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("providers:", ort.get_available_providers())

pose = Body(
    mode="balanced",
    backend="onnxruntime",
    device="cuda",
    to_openpose=False,
)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

all_keypoints = []
all_scores = []
frame_indices = []
image_size = None
last_kpts = None
last_scores = None

for frame_idx in range(NUM_FRAMES):
    ok, frame = cap.read()
    if not ok:
        print(f"Stopped early at frame {frame_idx}")
        break

    h, w = frame.shape[:2]
    image_size = (w, h)

    keypoints, scores = pose(frame)

    if keypoints.shape[0] > 0:
        # Use first detected person.
        kpts = keypoints[0].astype(np.float32)
        sc = scores[0].astype(np.float32)
        last_kpts = kpts
        last_scores = sc
    else:
        # If detector misses, reuse previous frame if available.
        if last_kpts is None:
            kpts = np.zeros((17, 2), dtype=np.float32)
            sc = np.zeros((17,), dtype=np.float32)
        else:
            kpts = last_kpts.copy()
            sc = last_scores.copy()

    all_keypoints.append(kpts)
    all_scores.append(sc)
    frame_indices.append(frame_idx)

    if frame_idx % 25 == 0:
        print(f"processed frame {frame_idx}/{NUM_FRAMES}")

cap.release()

all_keypoints = np.stack(all_keypoints, axis=0)
all_scores = np.stack(all_scores, axis=0)
frame_indices = np.array(frame_indices, dtype=np.int32)

np.savez(
    OUT_PATH,
    keypoints=all_keypoints,
    scores=all_scores,
    frame_indices=frame_indices,
    image_size=np.array(image_size, dtype=np.int32),
)

print("saved:", OUT_PATH)
print("keypoints:", all_keypoints.shape)
print("scores:", all_scores.shape)
print("image_size:", image_size)
