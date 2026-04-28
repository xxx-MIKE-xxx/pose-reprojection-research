from pathlib import Path
import cv2
import numpy as np
import torch
import onnxruntime as ort

# Import torch first so CUDA/cuDNN DLLs are available to ONNX Runtime.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

from rtmlib import Body, draw_skeleton

img_path = Path("data/processed/mpi_frame0.jpg")
video_path = Path("data/raw/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi")
out_dir = Path("outputs/rtmlib_rtmpose_gpu")
out_dir.mkdir(parents=True, exist_ok=True)

# Recreate the test frame if it was removed.
if not img_path.exists():
    img_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    cv2.imwrite(str(img_path), frame)

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(img_path)

print("torch:", torch.__version__)
print("torch cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("onnxruntime:", ort.__version__)
print("providers:", ort.get_available_providers())

if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise RuntimeError("ONNX Runtime CUDAExecutionProvider is not available.")

pose = Body(
    mode="balanced",
    backend="onnxruntime",
    device="cuda",
    to_openpose=False,
)

keypoints, scores = pose(img)

vis = draw_skeleton(
    img.copy(),
    keypoints,
    scores,
    kpt_thr=0.3,
    openpose_skeleton=False,
)

cv2.imwrite(str(out_dir / "mpi_frame0_rtmpose.jpg"), vis)
np.savez(
    out_dir / "mpi_frame0_rtmpose_keypoints.npz",
    keypoints=keypoints,
    scores=scores,
)

print("keypoints shape:", keypoints.shape)
print("scores shape:", scores.shape)
print("saved image:", out_dir / "mpi_frame0_rtmpose.jpg")
print("saved keypoints:", out_dir / "mpi_frame0_rtmpose_keypoints.npz")
