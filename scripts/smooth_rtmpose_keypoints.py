from pathlib import Path
import argparse
import cv2
import numpy as np

DEFAULT_IN = Path("outputs/rtmlib_mpi_clip/video0_first243_rtmpose.npz")
DEFAULT_OUT = Path("outputs/rtmlib_mpi_clip/video0_first243_rtmpose_smoothed.npz")
DEFAULT_VIDEO = Path("data/raw/mpi_inf_3dhp/S1/Seq1/imageSequence/video_0.avi")
DEFAULT_VIS = Path("outputs/visualizations/video0_first243_2d_raw_vs_smoothed.mp4")

COCO_SKELETON = [
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 5), (0, 6),
]

JOINT_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip",
    "l_knee", "r_knee", "l_ankle", "r_ankle",
]

def smooth_1d(x, window):
    if window <= 1:
        return x.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2
    ramp = np.arange(1, pad + 2, dtype=np.float32)
    kernel = np.concatenate([ramp, ramp[-2::-1]])
    kernel = kernel / kernel.sum()

    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

def interpolate_joint(points, valid):
    t = np.arange(points.shape[0])

    if valid.sum() == 0:
        return points.copy()

    if valid.sum() == 1:
        only = points[valid][0]
        return np.repeat(only[None, :], points.shape[0], axis=0)

    out = points.copy()
    valid_t = t[valid]

    for d in range(2):
        out[:, d] = np.interp(t, valid_t, points[valid, d])

    return out

def smooth_keypoints(keypoints, scores, conf_thr=0.30, jump_thr=120.0, window=7):
    keypoints = keypoints.astype(np.float32)
    scores = scores.astype(np.float32)

    smoothed = np.zeros_like(keypoints)
    valid_masks = np.zeros(scores.shape, dtype=bool)

    for j in range(keypoints.shape[1]):
        raw_pts = keypoints[:, j, :]
        valid = scores[:, j] >= conf_thr

        # First fill using confidence only.
        filled_once = interpolate_joint(raw_pts, valid)

        # Mark impossible one-frame jumps as invalid.
        velocity = np.linalg.norm(np.diff(filled_once, axis=0), axis=1)
        jump_bad = np.concatenate([[False], velocity > jump_thr])
        valid = valid & (~jump_bad)
        valid_masks[:, j] = valid

        # Fill invalid frames, then smooth.
        filled = interpolate_joint(raw_pts, valid)

        for d in range(2):
            smoothed[:, j, d] = smooth_1d(filled[:, d], window)

    return smoothed, valid_masks

def joint_velocity(kpts):
    return np.linalg.norm(np.diff(kpts, axis=0), axis=2)

def draw_pose(frame, kpts, scores, scale_x, scale_y, label, line_color, point_color, kpt_thr=0.25):
    k = kpts.copy()
    k[:, 0] *= scale_x
    k[:, 1] *= scale_y

    for a, b in COCO_SKELETON:
        if scores[a] >= kpt_thr and scores[b] >= kpt_thr:
            pa = tuple(k[a].astype(int))
            pb = tuple(k[b].astype(int))
            cv2.line(frame, pa, pb, line_color, 3)

    for j, (x, y) in enumerate(k):
        if scores[j] >= kpt_thr:
            cv2.circle(frame, (int(x), int(y)), 5, point_color, -1)

    cv2.putText(
        frame,
        label,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame

def make_comparison_video(video_path, out_path, raw_kpts, smooth_kpts, scores, frame_indices):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    side = 768
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (side * 2, side),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not write {out_path}")

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            print(f"[WARN] Could not read frame {frame_idx}")
            continue

        h, w = frame.shape[:2]
        sx = side / w
        sy = side / h

        left = cv2.resize(frame.copy(), (side, side))
        right = cv2.resize(frame.copy(), (side, side))

        left = draw_pose(
            left,
            raw_kpts[i],
            scores[i],
            sx,
            sy,
            "RAW RTMPose",
            (0, 255, 0),
            (0, 0, 255),
        )
        right = draw_pose(
            right,
            smooth_kpts[i],
            scores[i],
            sx,
            sy,
            "SMOOTHED RTMPose",
            (255, 180, 0),
            (0, 255, 255),
        )

        combined = np.concatenate([left, right], axis=1)
        cv2.putText(
            combined,
            f"frame {int(frame_idx)}",
            (side - 90, side - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(combined)

        if i % 25 == 0:
            print(f"[VIS] wrote {i}/{len(frame_indices)}")

    cap.release()
    writer.release()
    print("saved comparison video:", out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--comparison-video", type=Path, default=DEFAULT_VIS)
    parser.add_argument("--conf-thr", type=float, default=0.30)
    parser.add_argument("--jump-thr", type=float, default=120.0)
    parser.add_argument("--window", type=int, default=7)
    parser.add_argument("--make-video", action="store_true")
    args = parser.parse_args()

    data = np.load(args.input)
    keypoints = data["keypoints"].astype(np.float32)
    scores = data["scores"].astype(np.float32)
    frame_indices = data["frame_indices"]
    image_size = data["image_size"]

    smoothed, valid_masks = smooth_keypoints(
        keypoints,
        scores,
        conf_thr=args.conf_thr,
        jump_thr=args.jump_thr,
        window=args.window,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keypoints=smoothed,
        scores=scores,
        frame_indices=frame_indices,
        image_size=image_size,
        raw_keypoints=keypoints,
        valid_masks=valid_masks,
        smoothing_conf_thr=np.array(args.conf_thr, dtype=np.float32),
        smoothing_jump_thr=np.array(args.jump_thr, dtype=np.float32),
        smoothing_window=np.array(args.window, dtype=np.int32),
    )

    raw_vel = joint_velocity(keypoints)
    smooth_vel = joint_velocity(smoothed)

    print("input:", args.input)
    print("output:", args.output)
    print("keypoints:", keypoints.shape)
    print("smoothed:", smoothed.shape)
    print("valid ratio:", float(valid_masks.mean()))
    print("")
    print("Velocity diagnostics, px/frame:")
    for name in ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist"]:
        j = JOINT_NAMES.index(name)
        print(
            f"{name:12s} "
            f"raw_mean={raw_vel[:, j].mean():7.2f} "
            f"smooth_mean={smooth_vel[:, j].mean():7.2f} "
            f"raw_max={raw_vel[:, j].max():7.2f} "
            f"smooth_max={smooth_vel[:, j].max():7.2f}"
        )

    if args.make_video:
        make_comparison_video(
            args.video,
            args.comparison_video,
            keypoints,
            smoothed,
            scores,
            frame_indices,
        )

if __name__ == "__main__":
    main()
