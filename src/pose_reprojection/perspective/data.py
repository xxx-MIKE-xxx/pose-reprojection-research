from pathlib import Path
import numpy as np
from scipy.io import loadmat

from .skeleton import reduce_to_mpi17, mpi17_to_h36m17, root_center


def _extract_camera_array(mat_obj, cam_idx):
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


def _reshape_joints(arr, dims):
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


def load_gt_npz(path, key="h36m17_3d_m", max_frames=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)
    if key not in data:
        raise KeyError(f"{path} does not contain {key}. Available keys: {list(data.files)}")

    x = data[key].astype(np.float32)
    if x.ndim != 3 or x.shape[1:] != (17, 3):
        raise ValueError(f"Expected {key} shape (T,17,3), got {x.shape}")

    if max_frames is not None:
        x = x[:int(max_frames)]

    frame_indices = data["frame_indices"] if "frame_indices" in data else np.arange(len(x), dtype=np.int32)
    frame_indices = frame_indices[:len(x)].astype(np.int32)

    return {
        "name": path.stem,
        "poses_3d_m": x,
        "frame_indices": frame_indices,
        "source": str(path),
    }


def load_mpi_annot_3d(annot_path, camera=0, start=0, num_frames=None):
    annot_path = Path(annot_path)
    if not annot_path.exists():
        raise FileNotFoundError(annot_path)

    mat = loadmat(str(annot_path))
    if "annot3" not in mat:
        raise KeyError(f"Expected annot3 in {annot_path}; keys={list(mat.keys())}")

    cam_annot3 = _extract_camera_array(mat["annot3"], int(camera))
    joints3d_raw = _reshape_joints(cam_annot3, dims=3)

    start = int(start)
    end = None if num_frames is None else start + int(num_frames)
    joints3d_raw = joints3d_raw[start:end]

    joints3d_mpi17 = reduce_to_mpi17(joints3d_raw)
    joints3d_h36m17_m = mpi17_to_h36m17(joints3d_mpi17) * 0.001

    frame_indices = np.arange(start, start + len(joints3d_h36m17_m), dtype=np.int32)

    return {
        "name": annot_path.parent.name,
        "poses_3d_m": joints3d_h36m17_m.astype(np.float32),
        "frame_indices": frame_indices,
        "source": str(annot_path),
    }


def load_sources(config):
    ds = config["dataset"]
    max_frames = ds.get("max_frames")

    prepared = ds.get("prepared_gt_npz")
    if prepared and Path(prepared).exists():
        item = load_gt_npz(prepared, key=ds.get("preferred_gt_key", "h36m17_3d_m"), max_frames=max_frames)
        x = item["poses_3d_m"]
        if ds.get("root_center", True):
            x = root_center(x)
        item["poses_3d_m"] = x.astype(np.float32)
        return [item]

    out = []
    for source in ds.get("sources", []):
        typ = source.get("type", "mpi_annot")
        if typ == "npz":
            item = load_gt_npz(
                source["path"],
                key=source.get("key", ds.get("preferred_gt_key", "h36m17_3d_m")),
                max_frames=source.get("max_frames", max_frames),
            )
        elif typ == "mpi_annot":
            item = load_mpi_annot_3d(
                source["annot_path"],
                camera=source.get("camera", 0),
                start=source.get("start", 0),
                num_frames=source.get("num_frames", max_frames),
            )
        else:
            raise ValueError(f"Unknown source type: {typ}")

        item["name"] = source.get("name", item["name"])
        item["subject"] = source.get("subject")
        item["sequence"] = source.get("sequence")
        item["camera"] = source.get("camera")
        item["annot_path"] = source.get("annot_path", item.get("source"))

        x = item["poses_3d_m"]
        if ds.get("root_center", True):
            x = root_center(x)
        item["poses_3d_m"] = x.astype(np.float32)
        out.append(item)

    if not out:
        raise RuntimeError("No GT 3D source could be loaded. Provide dataset.prepared_gt_npz or dataset.sources.")

    return out
