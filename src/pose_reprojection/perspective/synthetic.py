from pathlib import Path
import json
import re
import numpy as np

from .camera import (
    sample_camera_params,
    canonical_camera_params,
    camera_params_to_vector,
    project_np,
    normalize_screen_coordinates_np,
    raw_2d_metadata,
)


def _parse_subject(source):
    if source.get("subject"):
        return str(source["subject"])

    name = str(source.get("name", ""))
    m = re.search(r"(S\d+)", name)
    if m:
        return m.group(1)

    path = str(
        source.get("annot_path", source.get("path", source.get("source", "")))
    ).replace("\\", "/")
    m = re.search(r"/(S\d+)/Seq\d+/", path)
    if m:
        return m.group(1)

    return "unknown"


def _parse_sequence(source):
    if source.get("sequence"):
        return str(source["sequence"])

    name = str(source.get("name", ""))
    m = re.search(r"(Seq\d+)", name)
    if m:
        return m.group(1)

    path = str(
        source.get("annot_path", source.get("path", source.get("source", "")))
    ).replace("\\", "/")
    m = re.search(r"/S\d+/(Seq\d+)/", path)
    if m:
        return m.group(1)

    return "unknown"


def _perturb_2d(u_px, rng, noise_cfg, image_width, image_height):
    """Detector-like synthetic noise applied to projected 2D keypoints.

    Supported fields:
      keypoint_px_std: independent Gaussian noise per joint
      frame_translation_px_std: shared frame-level translation noise
      dropout_prob: random joint dropout; pelvis/root is not dropped
      clip_to_image: clip keypoints to image bounds
    """
    out = np.asarray(u_px, dtype=np.float32).copy()

    keypoint_std = float(noise_cfg.get("keypoint_px_std", 0.0))
    if keypoint_std > 0:
        out += rng.normal(0.0, keypoint_std, size=out.shape).astype(np.float32)

    frame_shift_std = float(noise_cfg.get("frame_translation_px_std", 0.0))
    if frame_shift_std > 0:
        if out.ndim == 4:
            # out: (N, T, J, 2), shift shared by all joints in each frame
            shift_shape = (out.shape[0], out.shape[1], 1, 2)
        elif out.ndim == 3:
            # out: (T, J, 2), shift shared by all joints in each frame
            shift_shape = (out.shape[0], 1, 2)
        else:
            raise ValueError(f"Expected 2D keypoints with shape (T,J,2) or (N,T,J,2), got {out.shape}")

        shift = rng.normal(0.0, frame_shift_std, size=shift_shape).astype(np.float32)
        out += shift

    dropout_prob = float(noise_cfg.get("dropout_prob", 0.0))
    if dropout_prob > 0:
        drop = rng.random(out.shape[:-1]) < dropout_prob
        drop[..., 0] = False  # never drop pelvis/root
        root = out[..., 0:1, :]
        out = np.where(drop[..., None], root, out)

    if bool(noise_cfg.get("clip_to_image", True)):
        out[..., 0] = np.clip(out[..., 0], 0.0, float(image_width) - 1.0)
        out[..., 1] = np.clip(out[..., 1], 0.0, float(image_height) - 1.0)

    return out.astype(np.float32)


def generate_synthetic_dataset(config, sources):
    rng = np.random.default_rng(int(config.get("seed", 1234)))
    syn = config["synthetic"]

    image_width = int(syn["image_width"])
    image_height = int(syn["image_height"])
    n_cam = int(syn["num_virtual_cameras"])

    x_list = []
    u_px_list = []
    u_px_clean_list = []
    u_norm_list = []
    u_norm_clean_list = []
    meta_list = []
    meta_clean_list = []
    z_list = []
    camera_R_list = []
    camera_K_list = []
    camera_t_list = []
    camera_records = []
    source_names = []
    source_subjects = []
    source_sequences = []
    frame_indices_list = []
    canonical_2d_list = []

    canonical = canonical_camera_params(syn["canonical_camera"], image_width, image_height)

    legacy_noise_cfg = syn.get("noise", {})
    detector_noise_enabled = bool(syn.get("detector_noise", {}).get("enabled", False))
    use_legacy_noise = (
        not detector_noise_enabled
        and (
            float(legacy_noise_cfg.get("keypoint_px_std", 0.0)) > 0
            or float(legacy_noise_cfg.get("dropout_prob", 0.0)) > 0
            or float(legacy_noise_cfg.get("frame_translation_px_std", 0.0)) > 0
        )
    )

    for source_idx, source in enumerate(sources):
        x = source["poses_3d_m"].astype(np.float32)
        frame_indices = source["frame_indices"].astype(np.int32)
        subject = _parse_subject(source)
        sequence = _parse_sequence(source)

        u_canon, _ = project_np(x, canonical)
        canonical_2d = u_canon.astype(np.float32)

        for cam_idx in range(n_cam):
            params = sample_camera_params(
                rng,
                syn["camera_ranges"],
                image_width=image_width,
                image_height=image_height,
            )
            u_px_clean, _, camera_info = project_np(x, params, return_camera=True)
            u_norm_clean = normalize_screen_coordinates_np(u_px_clean, image_width, image_height)
            meta_clean = raw_2d_metadata(u_px_clean, image_width, image_height)

            u_px = u_px_clean.copy()
            if use_legacy_noise:
                u_px = _perturb_2d(u_px, rng, legacy_noise_cfg, image_width, image_height)

            u_norm = normalize_screen_coordinates_np(u_px, image_width, image_height)
            meta = raw_2d_metadata(u_px, image_width, image_height)
            z_vec = camera_params_to_vector(params)

            x_list.append(x)
            u_px_list.append(u_px.astype(np.float32))
            u_px_clean_list.append(u_px_clean.astype(np.float32))
            u_norm_list.append(u_norm.astype(np.float32))
            u_norm_clean_list.append(u_norm_clean.astype(np.float32))
            meta_list.append(meta.astype(np.float32))
            meta_clean_list.append(meta_clean.astype(np.float32))
            z_list.append(z_vec.astype(np.float32))
            camera_R_list.append(camera_info["R_world_to_camera"].astype(np.float32))
            camera_K_list.append(camera_info["K"].astype(np.float32))
            camera_t_list.append(camera_info["t_world_to_camera"].astype(np.float32))
            canonical_2d_list.append(canonical_2d.astype(np.float32))
            source_names.append(source["name"])
            source_subjects.append(subject)
            source_sequences.append(sequence)
            frame_indices_list.append(frame_indices)
            camera_records.append({
                "source_idx": int(source_idx),
                "source_name": str(source["name"]),
                "source_subject": str(subject),
                "source_sequence": str(sequence),
                "virtual_camera_idx": int(cam_idx),
                **params,
            })

    arrays = {
        "x_gt": np.stack(x_list, axis=0).astype(np.float32),
        "u_px": np.stack(u_px_list, axis=0).astype(np.float32),
        "u_norm": np.stack(u_norm_list, axis=0).astype(np.float32),
        "raw_2d_metadata": np.stack(meta_list, axis=0).astype(np.float32),
        "u_px_clean": np.stack(u_px_clean_list, axis=0).astype(np.float32),
        "u_norm_clean": np.stack(u_norm_clean_list, axis=0).astype(np.float32),
        "raw_2d_metadata_clean": np.stack(meta_clean_list, axis=0).astype(np.float32),
        "z": np.stack(z_list, axis=0).astype(np.float32),
        "camera_R_world_to_camera": np.stack(camera_R_list, axis=0).astype(np.float32),
        "camera_K": np.stack(camera_K_list, axis=0).astype(np.float32),
        "camera_t_world_to_camera": np.stack(camera_t_list, axis=0).astype(np.float32),
        "canonical_2d_px": np.stack(canonical_2d_list, axis=0).astype(np.float32),
        "frame_indices": np.stack(frame_indices_list, axis=0).astype(np.int32),
        "source_names": np.array(source_names),
        "source_subjects": np.array(source_subjects),
        "source_sequences": np.array(source_sequences),
    }

    return arrays, camera_records


def split_camera_indices(num_items, split_cfg, seed):
    rng = np.random.default_rng(int(seed))
    idx = np.arange(num_items)
    rng.shuffle(idx)

    n_train = int(round(num_items * float(split_cfg["train_fraction"])))
    n_val = int(round(num_items * float(split_cfg["val_fraction"])))
    n_train = min(max(n_train, 1), num_items)
    n_val = min(max(n_val, 1), max(num_items - n_train, 0))
    n_test = max(num_items - n_train - n_val, 0)
    if n_test == 0 and num_items >= 3:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train = idx[:n_train]
    val = idx[n_train:n_train + n_val]
    test = idx[n_train + n_val:]
    return train.astype(np.int64), val.astype(np.int64), test.astype(np.int64)


def make_split_indices(arrays, config):
    split_cfg = config.get("splits", {})
    mode = split_cfg.get("mode", "random")
    seed = int(config.get("seed", 1234))

    if mode in ("subject_holdout", "heldout_subject", "held_out_subject"):
        subjects = np.asarray(arrays["source_subjects"]).astype(str)

        train_subjects = split_cfg.get("train_subjects", ["S1", "S2", "S3", "S4", "S5", "S6"])
        val_subjects = split_cfg.get("val_subjects", ["S7"])
        test_subjects = split_cfg.get("test_subjects", ["S8"])

        train_set = set(map(str, train_subjects))
        val_set = set(map(str, val_subjects))
        test_set = set(map(str, test_subjects))

        train = np.where(np.isin(subjects, list(train_set)))[0].astype(np.int64)
        val = np.where(np.isin(subjects, list(val_set)))[0].astype(np.int64)
        test = np.where(np.isin(subjects, list(test_set)))[0].astype(np.int64)

        rng = np.random.default_rng(seed)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            available = sorted(set(subjects.tolist()))
            raise RuntimeError(
                "Subject holdout split produced an empty split. "
                f"available={available}, train={train_subjects}, val={val_subjects}, test={test_subjects}"
            )

        return {"train": train, "val": val, "test": test}

    train, val, test = split_camera_indices(arrays["x_gt"].shape[0], split_cfg, seed)
    return {"train": train, "val": val, "test": test}


def _noise_profile_for_split(detector_cfg, split):
    profiles = detector_cfg.get("profiles", {})
    if split in profiles:
        return profiles[split]
    if "all" in profiles:
        return profiles["all"]

    # Backward/simple config style.
    return {
        "keypoint_px_std": detector_cfg.get("keypoint_px_std", 0.0),
        "frame_translation_px_std": detector_cfg.get("frame_translation_px_std", 0.0),
        "dropout_prob": detector_cfg.get("dropout_prob", 0.0),
        "clip_to_image": detector_cfg.get("clip_to_image", True),
    }


def apply_detector_noise_by_split(arrays, config):
    detector_cfg = config.get("synthetic", {}).get("detector_noise", {})
    if not detector_cfg or not bool(detector_cfg.get("enabled", False)):
        return arrays

    image_width = int(config["synthetic"]["image_width"])
    image_height = int(config["synthetic"]["image_height"])
    rng = np.random.default_rng(int(config.get("seed", 1234)) + int(detector_cfg.get("seed_offset", 10000)))

    u_px = arrays["u_px_clean"].copy()
    apply_to = detector_cfg.get("apply_to_splits", ["train"])

    applied = {}
    for split in apply_to:
        key = f"{split}_indices"
        if key not in arrays:
            continue

        idx = arrays[key].astype(np.int64)
        if len(idx) == 0:
            continue

        profile = _noise_profile_for_split(detector_cfg, split)
        u_px[idx] = _perturb_2d(u_px[idx], rng, profile, image_width, image_height)
        applied[split] = {
            "num_sequences": int(len(idx)),
            "profile": profile,
        }

    arrays["u_px"] = u_px.astype(np.float32)
    arrays["u_norm"] = normalize_screen_coordinates_np(u_px, image_width, image_height).astype(np.float32)
    arrays["raw_2d_metadata"] = raw_2d_metadata(u_px, image_width, image_height).astype(np.float32)
    arrays["detector_noise_applied_json"] = np.array(json.dumps(applied))

    print("[detector_noise] applied:", json.dumps(applied, indent=2))
    return arrays


def save_synthetic_dataset(path, arrays, camera_records, split_indices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(arrays)
    payload["train_indices"] = np.asarray(split_indices["train"], dtype=np.int64)
    payload["val_indices"] = np.asarray(split_indices["val"], dtype=np.int64)
    payload["test_indices"] = np.asarray(split_indices["test"], dtype=np.int64)
    payload["camera_records_json"] = np.array(json.dumps(camera_records))

    np.savez_compressed(path, **payload)


def load_synthetic_dataset(path):
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    arrays = {k: data[k] for k in data.files if k != "camera_records_json"}
    camera_records = json.loads(str(data["camera_records_json"]))
    return arrays, camera_records
