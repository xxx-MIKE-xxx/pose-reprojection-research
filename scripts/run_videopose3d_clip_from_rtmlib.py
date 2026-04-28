from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VP3D_ROOT = ROOT / "third_party" / "VideoPose3D"
sys.path.insert(0, str(VP3D_ROOT))

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

KPTS_PATH = ROOT / "outputs" / "rtmlib_mpi_clip" / "video0_first243_rtmpose.npz"
CKPT_PATH = ROOT / "checkpoints" / "videopose3d" / "pretrained_h36m_cpn.bin"
OUT_DIR = ROOT / "outputs" / "videopose3d_mpi_clip"
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

def coco17_to_h36m17_seq(coco_seq):
    out = np.zeros((coco_seq.shape[0], 17, 2), dtype=np.float32)

    l_hip = coco_seq[:, COCO["l_hip"]]
    r_hip = coco_seq[:, COCO["r_hip"]]
    l_sh = coco_seq[:, COCO["l_shoulder"]]
    r_sh = coco_seq[:, COCO["r_shoulder"]]

    pelvis = avg(l_hip, r_hip)
    thorax = avg(l_sh, r_sh)
    spine = avg(pelvis, thorax)
    neck = thorax
    head = coco_seq[:, COCO["nose"]]

    out[:, 0] = pelvis
    out[:, 1] = r_hip
    out[:, 2] = coco_seq[:, COCO["r_knee"]]
    out[:, 3] = coco_seq[:, COCO["r_ankle"]]
    out[:, 4] = l_hip
    out[:, 5] = coco_seq[:, COCO["l_knee"]]
    out[:, 6] = coco_seq[:, COCO["l_ankle"]]
    out[:, 7] = spine
    out[:, 8] = thorax
    out[:, 9] = neck
    out[:, 10] = head
    out[:, 11] = l_sh
    out[:, 12] = coco_seq[:, COCO["l_elbow"]]
    out[:, 13] = coco_seq[:, COCO["l_wrist"]]
    out[:, 14] = r_sh
    out[:, 15] = coco_seq[:, COCO["r_elbow"]]
    out[:, 16] = coco_seq[:, COCO["r_wrist"]]

    return out

data = np.load(KPTS_PATH)
coco_2d = data["keypoints"].astype(np.float32)
scores = data["scores"].astype(np.float32)
frame_indices = data["frame_indices"]
w, h = data["image_size"]

if coco_2d.shape != (243, 17, 2):
    raise ValueError(f"Expected (243, 17, 2), got {coco_2d.shape}")

h36m_2d = coco17_to_h36m17_seq(coco_2d)
h36m_2d_norm = normalize_screen_coordinates(h36m_2d.copy(), w=int(w), h=int(h))

input_2d = torch.from_numpy(h36m_2d_norm.astype(np.float32)).unsqueeze(0)

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
model.load_state_dict(ckpt["model_pos"], strict=True)

model.eval()
with torch.no_grad():
    pred_3d = model(input_2d.to(device)).cpu().numpy()

out_path = OUT_DIR / "video0_first243_videopose3d.npz"
np.savez(
    out_path,
    pred_3d=pred_3d,
    input_2d_h36m_pixels=h36m_2d,
    input_2d_h36m_normalized=h36m_2d_norm,
    rtmpose_scores=scores,
    frame_indices=frame_indices,
    image_size=np.array([w, h], dtype=np.int32),
)

print("device:", device)
print("input_2d:", tuple(input_2d.shape))
print("pred_3d:", pred_3d.shape)
print("saved:", out_path)
print("center frame index:", int(frame_indices[len(frame_indices)//2]))
