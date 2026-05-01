from pathlib import Path
import hashlib
import json
import re

import h5py
import numpy as np


OFFICIAL_ROOT_INDEX = 14
OFFICIAL_THRESHOLDS_MM = np.arange(0, 151, 5, dtype=np.float64)

OFFICIAL_JOINT_NAMES = [
    "head_top",
    "neck",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "pelvis",
    "spine",
    "head",
]

# Official MPI-INF-3DHP relevant-joint order -> the H36M17 order used by VideoPose3D here.
OFFICIAL_TO_H36M17 = [14, 8, 9, 10, 11, 12, 13, 15, 1, 1, 16, 5, 6, 7, 2, 3, 4]

# Approximate inverse for evaluating H36M17 predictions in the official relevant-joint order.
# H36M17 has no separate official head_top joint, so head_top and head share the H36M head.
H36M17_TO_OFFICIAL = [10, 9, 14, 15, 16, 11, 12, 13, 1, 2, 3, 4, 5, 6, 0, 7, 10]

FALLBACK_FOCAL_MM = {
    "TS1": 7.320339203,
    "TS2": 7.320339203,
    "TS3": 7.320339203,
    "TS4": 7.320339203,
    "TS5": 8.770747185,
    "TS6": 8.770747185,
}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_mpi3dhp_test_set(root):
    root = Path(root)
    if (root / "mpi_inf_3dhp_test_set").is_dir():
        root = root / "mpi_inf_3dhp_test_set"
    ts_dirs = [root / f"TS{i}" for i in range(1, 7)]
    missing = [str(p) for p in ts_dirs if not (p / "annot_data.mat").exists()]
    if missing:
        raise FileNotFoundError(f"Missing MPI-INF-3DHP test annotations: {missing}")
    return ts_dirs


def load_h5_mat(path):
    return h5py.File(Path(path), "r")


def _decode_h5_chars(arr):
    arr = np.asarray(arr)
    if arr.dtype.kind in ("u", "i") and arr.ndim >= 1:
        flat = arr.reshape(-1)
        try:
            return "".join(chr(int(x)) for x in flat if int(x) != 0)
        except Exception:
            return arr
    return arr


def extract_dataset_or_refs(f, key):
    obj = f[key] if isinstance(key, str) else key
    if isinstance(obj, h5py.Group):
        return {k: extract_dataset_or_refs(f, obj[k]) for k in obj.keys()}
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"Unsupported HDF5 object for {key}: {type(obj)}")

    data = obj[()]
    if h5py.check_dtype(ref=obj.dtype) is None:
        return data

    out = np.empty(data.shape, dtype=object)
    for idx, ref in np.ndenumerate(data):
        if not ref:
            out[idx] = None
            continue
        deref = f[ref][()]
        out[idx] = _decode_h5_chars(deref)
    return out


def _dataset_manifest(f):
    manifest = {}
    for key in f.keys():
        if key == "#refs#":
            continue
        obj = f[key]
        if isinstance(obj, h5py.Dataset):
            manifest[key] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "is_reference": h5py.check_dtype(ref=obj.dtype) is not None,
            }
        else:
            manifest[key] = {"type": "group", "keys": list(obj.keys())}
    return manifest


def _squeeze_single_subject(arr, expected_last_dims):
    arr = np.asarray(arr)
    if arr.ndim == len(expected_last_dims) + 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.shape[-len(expected_last_dims):] != tuple(expected_last_dims):
        raise ValueError(f"Expected trailing shape {expected_last_dims}, got {arr.shape}")
    return arr


def load_ts_annot(ts_dir):
    ts_dir = Path(ts_dir)
    annot_path = ts_dir / "annot_data.mat"
    with load_h5_mat(annot_path) as f:
        manifest = _dataset_manifest(f)
        annot2 = _squeeze_single_subject(extract_dataset_or_refs(f, "annot2"), (17, 2)).astype(np.float32)
        annot3 = _squeeze_single_subject(extract_dataset_or_refs(f, "annot3"), (17, 3)).astype(np.float32)
        univ_annot3 = _squeeze_single_subject(extract_dataset_or_refs(f, "univ_annot3"), (17, 3)).astype(np.float32)
        valid_frame = np.asarray(extract_dataset_or_refs(f, "valid_frame")).reshape(-1).astype(bool)
        activity = np.asarray(extract_dataset_or_refs(f, "activity_annotation")).reshape(-1).astype(np.int32)

    if len(valid_frame) != annot2.shape[0]:
        raise ValueError(f"{annot_path}: valid_frame length {len(valid_frame)} != annot2 frames {annot2.shape[0]}")

    return {
        "sequence_name": ts_dir.name,
        "annot_path": str(annot_path),
        "annot_hash_sha256": sha256_file(annot_path),
        "annot2": annot2,
        "annot3": annot3,
        "univ_annot3": univ_annot3,
        "valid_frame": valid_frame,
        "activity_annotation": activity,
        "manifest": manifest,
    }


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return 2048, 2048


def prepare_official_gt2d_gt3d(root):
    ts_dirs = discover_mpi3dhp_test_set(root)
    test_root = ts_dirs[0].parent

    u_list = []
    x_list = []
    seq_names = []
    frame_indices = []
    activities = []
    image_paths = []
    image_widths = []
    image_heights = []
    valid_masks = []
    sequence_start_indices = []
    sequence_frame_counts = []
    sequence_valid_counts = []
    source_hashes = {}
    ts_manifests = {}

    valid_offset = 0
    for ts_dir in ts_dirs:
        rec = load_ts_annot(ts_dir)
        seq = rec["sequence_name"]
        valid = rec["valid_frame"]
        n_frames = int(len(valid))
        frame_all = np.arange(1, n_frames + 1, dtype=np.int32)
        rel_paths = np.array(
            [f"{seq}/imageSequence/img_{i:06d}.jpg" for i in frame_all],
            dtype="<U128",
        )

        first_img = test_root / rel_paths[0]
        width, height = _image_size(first_img)

        idx = np.where(valid)[0]
        u_list.append(rec["annot2"][idx].astype(np.float32))
        x_list.append(rec["univ_annot3"][idx].astype(np.float32))
        seq_names.extend([seq] * len(idx))
        frame_indices.extend(frame_all[idx].tolist())
        activities.extend(rec["activity_annotation"][idx].astype(np.int32).tolist())
        image_paths.extend(rel_paths[idx].tolist())
        image_widths.extend([width] * len(idx))
        image_heights.extend([height] * len(idx))
        valid_masks.append(valid.astype(bool))
        sequence_start_indices.append(valid_offset)
        sequence_frame_counts.append(n_frames)
        sequence_valid_counts.append(int(len(idx)))
        valid_offset += int(len(idx))
        source_hashes[seq] = rec["annot_hash_sha256"]
        ts_manifests[seq] = rec["manifest"]

    arrays = {
        "u_gt2d_px": np.concatenate(u_list, axis=0).astype(np.float32),
        "x_gt3d_univ_mm": np.concatenate(x_list, axis=0).astype(np.float32),
        "valid_mask": np.concatenate(valid_masks, axis=0).astype(bool),
        "sequence_names": np.asarray(seq_names, dtype="<U8"),
        "frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "activity_labels": np.asarray(activities, dtype=np.int32),
        "image_paths": np.asarray(image_paths, dtype="<U128"),
        "image_width": np.asarray(image_widths, dtype=np.int32),
        "image_height": np.asarray(image_heights, dtype=np.int32),
        "official_joint_names": np.asarray(OFFICIAL_JOINT_NAMES, dtype="<U32"),
        "root_index": np.asarray(OFFICIAL_ROOT_INDEX, dtype=np.int32),
        "pck_thresholds_mm": OFFICIAL_THRESHOLDS_MM.astype(np.float32),
        "sequence_start_indices": np.asarray(sequence_start_indices, dtype=np.int32),
        "sequence_frame_counts": np.asarray(sequence_frame_counts, dtype=np.int32),
        "sequence_valid_counts": np.asarray(sequence_valid_counts, dtype=np.int32),
        "source_annotation_hashes_json": np.asarray(json.dumps(source_hashes, sort_keys=True)),
        "loader_manifest_json": np.asarray(json.dumps(ts_manifests, sort_keys=True)),
    }
    manifest = build_official_eval_manifest(root, arrays, source_hashes, ts_manifests)
    return arrays, manifest


def official_root_center_17(x, root_idx=OFFICIAL_ROOT_INDEX):
    x = np.asarray(x)
    return x - x[..., int(root_idx):int(root_idx) + 1, :]


def official_pck_auc(errors_mm):
    errors = np.asarray(errors_mm, dtype=np.float64)
    valid = np.isfinite(errors)
    if not np.any(valid):
        curve = np.zeros_like(OFFICIAL_THRESHOLDS_MM, dtype=np.float64)
    else:
        curve = np.asarray([np.mean(errors[valid] < t) for t in OFFICIAL_THRESHOLDS_MM], dtype=np.float64)
    return {
        "thresholds_mm": OFFICIAL_THRESHOLDS_MM.astype(int).tolist(),
        "pck_curve": curve.tolist(),
        "pck_curve_percent": (100.0 * curve).tolist(),
        "pck150": float(100.0 * curve[-1]),
        "pck150_fraction": float(curve[-1]),
        "auc": float(100.0 * curve.mean()),
    }


def official_joint_groups():
    return {
        "Head": [0],
        "Neck": [1],
        "Shou": [2, 5],
        "Elbow": [3, 6],
        "Wrist": [4, 7],
        "Hip": [8, 11],
        "Knee": [9, 12],
        "Ankle": [10, 13],
    }


def official_activity_names():
    return {
        1: "Standing/Walking",
        2: "Exercising",
        3: "Sitting",
        4: "Reaching/Crouching",
        5: "On The Floor",
        6: "Sports",
        7: "Miscellaneous",
    }


def _strip_comment(line):
    return line.split("#", 1)[0].strip()


def _numbers_from_line(line):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", _strip_comment(line))]


def _parse_calibration_file(path):
    vals = {}
    path = Path(path)
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("sensorSize"):
            nums = _numbers_from_line(stripped)
            if len(nums) >= 2:
                vals["sensor_width_mm"] = nums[0]
                vals["sensor_height_mm"] = nums[1]
        elif stripped.startswith("focalLength"):
            nums = _numbers_from_line(stripped)
            if nums:
                vals["focal_length_mm"] = nums[0]
        elif stripped.startswith("centerOffset"):
            nums = _numbers_from_line(stripped)
            if len(nums) >= 2:
                vals["center_offset_mm"] = [nums[0], nums[1]]
        elif stripped.startswith("pixelAspect"):
            nums = _numbers_from_line(stripped)
            if nums:
                vals["pixel_aspect"] = nums[0]
    if "sensor_width_mm" in vals and "focal_length_mm" in vals:
        return vals
    return None


def official_intrinsics_for_sequence(sequence_name, image_width, image_height, test_root=None):
    """Return K and metadata for a TS sequence.

    Calibration files are used when available. If parsing fails, this falls back to
    the focal constants used by mpii_perspective_correction_code.m and image-center
    principal point.
    """
    seq = str(sequence_name)
    w = float(image_width)
    h = float(image_height)
    if test_root is None:
        test_root = Path("data/raw/mpi_inf_3dhp/mpi_inf_3dhp_test_set")
    test_root = Path(test_root)
    calib = test_root / "test_util" / "camera_calibration" / (
        "ts1-4cameras.calib" if seq in ("TS1", "TS2", "TS3", "TS4") else "ts5-6cameras.calib"
    )
    parsed = _parse_calibration_file(calib)

    if parsed is not None:
        sx = float(parsed["sensor_width_mm"])
        sy = float(parsed.get("sensor_height_mm", sx))
        focal_mm = float(parsed["focal_length_mm"])
        fx = w / sx * focal_mm
        fy = h / sy * focal_mm
        off = parsed.get("center_offset_mm", [0.0, 0.0])
        cx = 0.5 * w + float(off[0]) * (w / sx)
        cy = 0.5 * h + float(off[1]) * (h / sy)
        mode = "official_calibration"
        source = str(calib)
    else:
        focal_mm = FALLBACK_FOCAL_MM[seq]
        fx = w / 10.0 * focal_mm
        fy = fx
        cx = 0.5 * w
        cy = 0.5 * h
        mode = "official_test_util_fallback"
        source = "mpii_perspective_correction_code.m"

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, {
        "sequence": seq,
        "intrinsics_mode": mode,
        "source": source,
        "image_width": int(w),
        "image_height": int(h),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "distortion_ignored": True,
    }


def build_official_eval_manifest(root, arrays, source_hashes=None, loader_manifest=None):
    sequence_names = [str(x) for x in np.unique(arrays["sequence_names"]).tolist()]
    valid_counts = {
        seq: int(np.sum(arrays["sequence_names"].astype(str) == seq))
        for seq in sequence_names
    }
    return {
        "dataset_root": str(Path(root)),
        "protocol": "MPI-INF-3DHP official test-set protocol",
        "annotation_format": "MATLAB v7.3 HDF5 via h5py",
        "root_index": OFFICIAL_ROOT_INDEX,
        "root_index_matlab": 15,
        "pck_thresholds_mm": OFFICIAL_THRESHOLDS_MM.astype(int).tolist(),
        "pck150_threshold_mm": 150,
        "auc_definition": "100 * mean(PCK curve over thresholds 0:5:150)",
        "activity_names": official_activity_names(),
        "joint_groups": official_joint_groups(),
        "official_joint_names": OFFICIAL_JOINT_NAMES,
        "official_to_h36m17": OFFICIAL_TO_H36M17,
        "h36m17_to_official": H36M17_TO_OFFICIAL,
        "num_valid_frames": int(arrays["u_gt2d_px"].shape[0]),
        "sequence_valid_counts": valid_counts,
        "source_annotation_hashes": source_hashes or {},
        "h5_key_manifest": loader_manifest or {},
    }


def similarity_transform_single(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mu_pred = pred.mean(axis=0)
    mu_target = target.mean(axis=0)
    X = pred - mu_pred
    Y = target - mu_target
    var_x = np.sum(X ** 2)
    if var_x < 1e-12:
        return pred.copy()
    H = X.T @ Y
    U, s, Vt = np.linalg.svd(H)
    Z = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Z[-1, -1] = -1
    R = U @ Z @ Vt
    scale = np.sum(s * np.diag(Z)) / var_x
    return scale * (X @ R) + mu_target


def pa_mpjpe_mm(pred_rc, gt_rc):
    aligned = np.zeros_like(pred_rc, dtype=np.float64)
    for i in range(pred_rc.shape[0]):
        aligned[i] = similarity_transform_single(pred_rc[i], gt_rc[i])
    return float(np.mean(np.linalg.norm(aligned - gt_rc, axis=-1)))


def summarize_official_errors(errors_mm):
    errors = np.asarray(errors_mm, dtype=np.float64)
    pck = official_pck_auc(errors)
    return {
        "num_frames": int(errors.shape[0]),
        "mpjpe_mm": float(np.mean(errors)),
        "pck150": pck["pck150"],
        "pck150_fraction": pck["pck150_fraction"],
        "auc": pck["auc"],
        "pck_curve_percent": pck["pck_curve_percent"],
    }


def compute_official_metrics(pred_mm, gt_mm, sequence_names, activity_labels):
    pred_rc = official_root_center_17(pred_mm)
    gt_rc = official_root_center_17(gt_mm)
    errors = np.linalg.norm(pred_rc - gt_rc, axis=-1)
    overall = summarize_official_errors(errors)
    overall["pa_mpjpe_mm"] = pa_mpjpe_mm(pred_rc, gt_rc)
    overall["per_joint_mpjpe_mm"] = {
        name: float(errors[:, j].mean()) for j, name in enumerate(OFFICIAL_JOINT_NAMES)
    }

    sequence_names = np.asarray(sequence_names).astype(str)
    activity_labels = np.asarray(activity_labels).astype(np.int32)
    per_sequence = {}
    for seq in sorted(np.unique(sequence_names).tolist()):
        mask = sequence_names == seq
        vals = summarize_official_errors(errors[mask])
        vals["pa_mpjpe_mm"] = pa_mpjpe_mm(pred_rc[mask], gt_rc[mask])
        per_sequence[seq] = vals

    per_activity = {}
    for aid, name in official_activity_names().items():
        mask = activity_labels == int(aid)
        if not np.any(mask):
            continue
        vals = summarize_official_errors(errors[mask])
        vals["pa_mpjpe_mm"] = pa_mpjpe_mm(pred_rc[mask], gt_rc[mask])
        per_activity[name] = vals

    joint_groups = {}
    for group, idx in official_joint_groups().items():
        joint_groups[group] = official_pck_auc(errors[:, idx])

    overall["joint_groups"] = joint_groups
    return {
        "overall": overall,
        "per_sequence": per_sequence,
        "per_activity": per_activity,
        "per_joint_mpjpe_mm": overall["per_joint_mpjpe_mm"],
    }
