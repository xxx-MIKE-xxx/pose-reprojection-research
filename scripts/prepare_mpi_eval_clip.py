from pathlib import Path
import argparse
import re
import numpy as np
from scipy.io import loadmat

RAW28_TO_MPI17 = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

MPI17_NAMES = [
    "head_top", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "root", "spine", "head",
]

H36M17_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

# Convert MPI17 order to the H36M-like order used by your VideoPose3D scripts.
MPI17_TO_H36M17 = [
    14,  # pelvis/root
    8, 9, 10,  # right leg
    11, 12, 13,  # left leg
    15,  # spine
    1,   # thorax approximated as neck
    1,   # neck
    16,  # head
    5, 6, 7,  # left arm
    2, 3, 4,  # right arm
]

def parse_floats(line):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]

def parse_camera_calibration(path, cam_idx):
    lines = path.read_text(errors="ignore").splitlines()
    start = cam_idx * 7

    if start + 6 >= len(lines):
        raise ValueError(f"Camera {cam_idx} block not found in {path}")

    size_nums = parse_floats(lines[start + 3])
    k_nums = parse_floats(lines[start + 5])
    rt_nums = parse_floats(lines[start + 6])

    if len(size_nums) < 2:
        raise ValueError(f"Could not parse image size for camera {cam_idx}")
    if len(k_nums) < 9:
        raise ValueError(f"Could not parse intrinsic K for camera {cam_idx}")
    if len(rt_nums) < 16:
        raise ValueError(f"Could not parse extrinsic RT for camera {cam_idx}")

    w, h = int(size_nums[-2]), int(size_nums[-1])
    K = np.array(k_nums[-9:], dtype=np.float32).reshape(3, 3)
    RT = np.array(rt_nums[-16:], dtype=np.float32).reshape(4, 4)

    return {
        "w": w,
        "h": h,
        "K": K,
        "RT": RT,
        "R": RT[:3, :3],
        "T": RT[:3, 3],
    }

def extract_camera_array(mat_obj, cam_idx):
    arr = np.asarray(mat_obj)

    if arr.dtype == object:
        squeezed = arr.squeeze()
        if squeezed.ndim == 1:
            return np.asarray(squeezed[cam_idx]).squeeze()
        if squeezed.ndim == 2:
            return np.asarray(squeezed[cam_idx, 0]).squeeze()
        flat = squeezed.flatten()
        return np.asarray(flat[cam_idx]).squeeze()

    squeezed = arr.squeeze()

    if squeezed.ndim >= 3 and squeezed.shape[0] > cam_idx:
        return squeezed[cam_idx]

    return squeezed

def reshape_joints(arr, dims):
    arr = np.asarray(arr, dtype=np.float32).squeeze()

    if arr.ndim == 2:
        if arr.shape[1] % dims == 0:
            frames = arr.shape[0]
            joints = arr.shape[1] // dims
            return arr.reshape(frames, joints, dims)
        if arr.shape[0] % dims == 0:
            frames = arr.shape[1]
            joints = arr.shape[0] // dims
            return arr.T.reshape(frames, joints, dims)

    if arr.ndim == 3:
        if arr.shape[-1] == dims:
            return arr
        if arr.shape[1] == dims:
            return np.transpose(arr, (0, 2, 1))
        if arr.shape[0] == dims:
            return np.transpose(arr, (2, 1, 0))

    raise ValueError(f"Could not reshape annotation array with shape {arr.shape} into joints with dims={dims}")

def reduce_to_mpi17(joints):
    if joints.shape[1] == 17:
        return joints

    if joints.shape[1] >= 28:
        return joints[:, RAW28_TO_MPI17]

    raise ValueError(f"Expected 17 or >=28 joints, got {joints.shape[1]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-root", type=Path, default=Path("data/raw/mpi_inf_3dhp/S1/Seq1"))
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=243)
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/mpi_s1_seq1_cam0_frames0_242_gt.npz"))
    args = parser.parse_args()

    annot_path = args.seq_root / "annot.mat"
    calib_path = args.seq_root / "camera.calibration"

    if not annot_path.exists():
        raise FileNotFoundError(annot_path)
    if not calib_path.exists():
        raise FileNotFoundError(calib_path)

    mat = loadmat(str(annot_path))
    if "annot2" not in mat or "annot3" not in mat:
        raise KeyError(f"Expected annot2 and annot3 in {annot_path}; keys={list(mat.keys())}")

    cam_annot2 = extract_camera_array(mat["annot2"], args.cam)
    cam_annot3 = extract_camera_array(mat["annot3"], args.cam)

    joints2d_raw = reshape_joints(cam_annot2, dims=2)
    joints3d_raw = reshape_joints(cam_annot3, dims=3)

    end = args.start + args.num_frames
    joints2d_raw = joints2d_raw[args.start:end]
    joints3d_raw = joints3d_raw[args.start:end]

    joints2d_mpi17 = reduce_to_mpi17(joints2d_raw)
    joints3d_mpi17 = reduce_to_mpi17(joints3d_raw)

    # MPI train annot3 is in millimeters in the common converters.
    joints3d_mpi17_m = joints3d_mpi17 * 0.001

    joints2d_h36m17 = joints2d_mpi17[:, MPI17_TO_H36M17]
    joints3d_h36m17_m = joints3d_mpi17_m[:, MPI17_TO_H36M17]

    calib = parse_camera_calibration(calib_path, args.cam)

    frame_indices = np.arange(args.start, args.start + len(joints2d_h36m17), dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        mpi17_2d_pixels=joints2d_mpi17.astype(np.float32),
        mpi17_3d_m=joints3d_mpi17_m.astype(np.float32),
        h36m17_2d_pixels=joints2d_h36m17.astype(np.float32),
        h36m17_3d_m=joints3d_h36m17_m.astype(np.float32),
        frame_indices=frame_indices,
        image_size=np.array([calib["w"], calib["h"]], dtype=np.int32),
        K=calib["K"],
        RT=calib["RT"],
        R=calib["R"],
        T=calib["T"],
        cam_idx=np.array(args.cam, dtype=np.int32),
        mpi17_names=np.array(MPI17_NAMES),
        h36m17_names=np.array(H36M17_NAMES),
        mpi17_to_h36m17=np.array(MPI17_TO_H36M17, dtype=np.int32),
    )

    print("saved:", args.output)
    print("raw 2d:", joints2d_raw.shape)
    print("raw 3d:", joints3d_raw.shape)
    print("mpi17 2d:", joints2d_mpi17.shape)
    print("mpi17 3d meters:", joints3d_mpi17_m.shape)
    print("h36m17 2d:", joints2d_h36m17.shape)
    print("h36m17 3d meters:", joints3d_h36m17_m.shape)
    print("image_size:", [calib["w"], calib["h"]])
    print("frame range:", int(frame_indices[0]), "to", int(frame_indices[-1]))

if __name__ == "__main__":
    main()
