from pathlib import Path
import cv2

video = Path("data/raw/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi")
out = Path("data/processed/mpi_frame0.jpg")
out.parent.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read frame")

cv2.imwrite(str(out), frame)
print("saved", out)
