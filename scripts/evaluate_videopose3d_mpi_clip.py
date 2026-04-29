from pathlib import Path
import argparse
import json
import numpy as np

from pose3d_metrics import (
    root_center,
    mpjpe,
    n_mpjpe,
    pa_mpjpe,
    pck_3d,
    auc_3d,
    acceleration_error,
    batch_procrustes_align,
)

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

BODY14 = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16]

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

def velocity_px(kpts):
    if kpts.shape[0] < 2:
        return np.zeros((0, kpts.shape[1]), dtype=np.float32)
    return np.linalg.norm(np.diff(kpts, axis=0), axis=-1)

def load_pred3d(path):
    data = np.load(path)
    pred = data["pred_3d"]
    if pred.ndim == 4:
        pred = pred[0]
    if pred.ndim != 3 or pred.shape[1:] != (17, 3):
        raise ValueError(f"Expected pred_3d as (T, 17, 3), got {pred.shape}")
    return pred.astype(np.float64)

def summarize_3d(pred, gt):
    pred_root = root_center(pred, root=0)
    gt_root = root_center(gt, root=0)

    pred14 = pred_root[:, BODY14]
    gt14 = gt_root[:, BODY14]

    pred_pa = batch_procrustes_align(pred_root, gt_root)
    pred14_pa = pred_pa[:, BODY14]

    return {
        "root_mpjpe_all17_mm": float(mpjpe(pred_root, gt_root) * 1000.0),
        "root_mpjpe_body14_mm": float(mpjpe(pred14, gt14) * 1000.0),
        "n_mpjpe_all17_mm": float(n_mpjpe(pred_root, gt_root) * 1000.0),
        "n_mpjpe_body14_mm": float(n_mpjpe(pred14, gt14) * 1000.0),
        "pa_mpjpe_all17_mm": float(pa_mpjpe(pred_root, gt_root) * 1000.0),
        "pa_mpjpe_body14_mm": float(pa_mpjpe(pred14, gt14) * 1000.0),
        "pck150_root_all17": float(pck_3d(pred_root, gt_root, threshold=0.150)),
        "pck150_pa_all17": float(pck_3d(pred_pa, gt_root, threshold=0.150)),
        "pck150_pa_body14": float(pck_3d(pred14_pa, gt14, threshold=0.150)),
        "auc150_pa_all17": float(auc_3d(pred_pa, gt_root, max_threshold=0.150)),
        "accel_error_root_all17_mm_per_frame2": float(acceleration_error(pred_root, gt_root) * 1000.0),
    }

def summarize_2d(pred_2d_h36m, gt_2d_h36m):
    err = np.linalg.norm(pred_2d_h36m - gt_2d_h36m, axis=-1)
    vel = velocity_px(pred_2d_h36m)

    return {
        "mean_2d_error_all17_px": float(np.mean(err)),
        "median_2d_error_all17_px": float(np.median(err)),
        "mean_2d_error_body14_px": float(np.mean(err[:, BODY14])),
        "mean_2d_velocity_all17_px_per_frame": float(np.mean(vel)),
        "max_2d_velocity_all17_px_per_frame": float(np.max(vel)) if vel.size else 0.0,
        "mean_2d_velocity_wrists_px_per_frame": float(np.mean(vel[:, [13, 16]])) if vel.size else 0.0,
        "max_2d_velocity_wrists_px_per_frame": float(np.max(vel[:, [13, 16]])) if vel.size else 0.0,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, default=Path("outputs/eval/mpi_s1_seq1_cam0_frames0_242_gt.npz"))
    parser.add_argument("--pred3d", type=Path, default=Path("outputs/videopose3d_mpi_clip_fullseq/video0_first243_videopose3d_fullseq.npz"))
    parser.add_argument("--pred2d", type=Path, default=Path("outputs/rtmlib_mpi_clip/video0_first243_rtmpose.npz"))
    parser.add_argument("--name", type=str, default="rtmpose_raw_videopose3d")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/baseline_mpi_s1_seq1_cam0_243.json"))
    args = parser.parse_args()

    gt_data = np.load(args.gt)
    gt3d = gt_data["h36m17_3d_m"].astype(np.float64)
    gt2d = gt_data["h36m17_2d_pixels"].astype(np.float64)
    gt_frames = gt_data["frame_indices"]

    pred3d = load_pred3d(args.pred3d)

    pred2d_data = np.load(args.pred2d)
    pred2d_coco = pred2d_data["keypoints"].astype(np.float32)
    pred2d_h36m = coco17_to_h36m17_seq(pred2d_coco).astype(np.float64)
    pred_frames = pred2d_data["frame_indices"]

    T = min(len(gt3d), len(pred3d), len(pred2d_h36m))
    gt3d = gt3d[:T]
    gt2d = gt2d[:T]
    pred3d = pred3d[:T]
    pred2d_h36m = pred2d_h36m[:T]

    result = {
        "clip": {
            "dataset": "MPI-INF-3DHP",
            "subject": "S1",
            "sequence": "Seq1",
            "camera": int(gt_data["cam_idx"]),
            "num_frames": int(T),
            "gt_frame_start": int(gt_frames[0]),
            "gt_frame_end": int(gt_frames[T - 1]),
            "pred_frame_start": int(pred_frames[0]),
            "pred_frame_end": int(pred_frames[T - 1]),
            "image_size": gt_data["image_size"].astype(int).tolist(),
        },
        "methods": {
            args.name: {
                **summarize_3d(pred3d, gt3d),
                **summarize_2d(pred2d_h36m, gt2d),
            }
        },
        "notes": [
            "MPJPE numbers are only meaningful after joint order, scale, and coordinate-frame checks.",
            "PA-MPJPE/P-MPJPE is the safest first comparison because it allows similarity alignment.",
            "Current VideoPose3D checkpoint was trained for Human3.6M-style 2D detections, so domain mismatch is expected.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("saved:", args.output)

if __name__ == "__main__":
    main()
