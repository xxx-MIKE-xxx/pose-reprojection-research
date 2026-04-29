from pathlib import Path
import argparse
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VP3D_ROOT = ROOT / "third_party" / "VideoPose3D"
sys.path.insert(0, str(VP3D_ROOT))

from common.model import TemporalModel
from common.camera import normalize_screen_coordinates

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

    out[:, 0] = pelvis
    out[:, 1] = r_hip
    out[:, 2] = coco_seq[:, COCO["r_knee"]]
    out[:, 3] = coco_seq[:, COCO["r_ankle"]]
    out[:, 4] = l_hip
    out[:, 5] = coco_seq[:, COCO["l_knee"]]
    out[:, 6] = coco_seq[:, COCO["l_ankle"]]
    out[:, 7] = spine
    out[:, 8] = thorax
    out[:, 9] = thorax
    out[:, 10] = coco_seq[:, COCO["nose"]]
    out[:, 11] = l_sh
    out[:, 12] = coco_seq[:, COCO["l_elbow"]]
    out[:, 13] = coco_seq[:, COCO["l_wrist"]]
    out[:, 14] = r_sh
    out[:, 15] = coco_seq[:, COCO["r_elbow"]]
    out[:, 16] = coco_seq[:, COCO["r_wrist"]]

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-keypoints", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/videopose3d/pretrained_h36m_cpn.bin"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pad", type=int, default=121)
    args = parser.parse_args()

    data = np.load(args.input_keypoints)
    coco_2d = data["keypoints"].astype(np.float32)
    scores = data["scores"].astype(np.float32)
    frame_indices = data["frame_indices"]
    w, h = data["image_size"]

    h36m_2d = coco17_to_h36m17_seq(coco_2d)
    h36m_2d_norm = normalize_screen_coordinates(h36m_2d.copy(), w=int(w), h=int(h))

    padded = np.pad(h36m_2d_norm, ((args.pad, args.pad), (0, 0), (0, 0)), mode="edge")
    input_2d = torch.from_numpy(padded.astype(np.float32)).unsqueeze(0)

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

    ckpt = torch.load(str(args.checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_pos"], strict=True)

    model.eval()
    with torch.no_grad():
        pred_3d = model(input_2d.to(device)).cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        pred_3d=pred_3d,
        input_2d_h36m_pixels=h36m_2d,
        input_2d_h36m_normalized=h36m_2d_norm,
        rtmpose_scores=scores,
        frame_indices=frame_indices,
        image_size=np.array([w, h], dtype=np.int32),
        source_keypoints=str(args.input_keypoints),
    )

    print("device:", device)
    print("input keypoints:", args.input_keypoints)
    print("input_2d padded:", tuple(input_2d.shape))
    print("pred_3d:", pred_3d.shape)
    print("saved:", args.output)

if __name__ == "__main__":
    main()
