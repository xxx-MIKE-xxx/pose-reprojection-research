from pathlib import Path
import cv2

video = Path("data/raw/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi")

if not video.exists():
    raise FileNotFoundError(video)

cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"Could not open {video}")

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ok, frame = cap.read()
cap.release()

print("video:", video)
print("frames:", frames)
print("fps:", fps)
print("size:", w, "x", h)
print("first_frame_loaded:", ok)
