from pathlib import Path
import sys
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VP3D_ROOT = ROOT / "third_party" / "VideoPose3D"
sys.path.insert(0, str(VP3D_ROOT))

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

KPTS_PATH = ROOT / "outputs" / "rtmlib_rtmpose_gpu" / "mpi_frame0_rtmpose_keypoints.npz"
IMG_PATH = ROOT / "data" / "processed" / "mpi_frame0.jpg"
CKPT_PATH = ROOT / "checkpoints" / "videopose3d" / "pretrained_h36m_cpn.bin"
OUT_DIR = ROOT / "outputs" / "videopose3d_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COCO = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

def avg(a, b):
    return (a + b) / 2.0

def coco17_to_h36m17(coco):
    """Approximate COCO-17 to Human3.6M-17 mapping.

    H36M order used by VideoPose3D:
    0 pelvis, 1 r_hip, 2 r_knee, 3 r_foot,
    4 l_hip, 5 l_knee, 6 l_foot,
    7 spine, 8 thorax, 9 neck, 10 head,
    11 l_shoulder, 12 l_elbow, 13 l_wrist,
    14 r_shoulder, 15 r_elbow, 16 r_wrist
    """
    out = np.zeros((17, 2), dtype=np.float32)

    l_hip = coco[COCO["l_hip"]]
    r_hip = coco[COCO["r_hip"]]
    l_sh = coco[COCO["l_shoulder"]]
    r_sh = coco[COCO["r_shoulder"]]
    pelvis = avg(l_hip, r_hip)
    thorax = avg(l_sh, r_sh)
    spine = avg(pelvis, thorax)
    neck = thorax
    head = coco[COCO["nose"]]

    out[0] = pelvis
    out[1] = r_hip
    out[2] = coco[COCO["r_knee"]]
    out[3] = coco[COCO["r_ankle"]]
    out[4] = l_hip
    out[5] = coco[COCO["l_knee"]]
    out[6] = coco[COCO["l_ankle"]]
    out[7] = spine
    out[8] = thorax
    out[9] = neck
    out[10] = head
    out[11] = l_sh
    out[12] = coco[COCO["l_elbow"]]
    out[13] = coco[COCO["l_wrist"]]
    out[14] = r_sh
    out[15] = coco[COCO["r_elbow"]]
    out[16] = coco[COCO["r_wrist"]]

    return out

data = np.load(KPTS_PATH)
keypoints = data["keypoints"]
scores = data["scores"]

if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 2):
    raise ValueError(f"Expected keypoints shape (N, 17, 2), got {keypoints.shape}")

# Use first detected person.
coco_2d = keypoints[0].astype(np.float32)
h36m_2d = coco17_to_h36m17(coco_2d)

img = cv2.imread(str(IMG_PATH))
if img is None:
    raise FileNotFoundError(IMG_PATH)

h, w = img.shape[:2]

# VideoPose3D pretrained model expects normalized 2D coordinates.
h36m_2d_norm = normalize_screen_coordinates(h36m_2d.copy(), w=w, h=h)

# The pretrained temporal model has receptive field 243 for architecture 3,3,3,3,3.
# For this smoke test, repeat one frame 243 times. This is not a real evaluation.
seq_len = 243
input_2d = np.repeat(h36m_2d_norm[None, :, :], seq_len, axis=0)
input_2d = torch.from_numpy(input_2d.astype(np.float32)).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TemporalModel(
    num_joints_in=17,
    in_features=2,
    num_joints_out=17,
    filter_widths=[3, 3, 3, 3, 3],
    causal=False,
    dropout=0.25,
    channels=1024,
    dense=False,
).to(device)

ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
state = ckpt["model_pos"] if isinstance(ckpt, dict) and "model_pos" in ckpt else ckpt
model.load_state_dict(state, strict=True)

model.eval()
with torch.no_grad():
    pred_3d = model(input_2d.to(device)).cpu().numpy()

np.savez(
    OUT_DIR / "mpi_frame0_videopose3d_smoke.npz",
    pred_3d=pred_3d,
    input_2d_h36m_pixels=h36m_2d,
    input_2d_h36m_normalized=h36m_2d_norm,
    rtmpose_scores=scores[0],
)

print("device:", device)
print("input_2d:", tuple(input_2d.shape))
print("pred_3d:", pred_3d.shape)
print("saved:", OUT_DIR / "mpi_frame0_videopose3d_smoke.npz")
print("NOTE: This is a plumbing smoke test, not a valid metric/evaluation yet.")
