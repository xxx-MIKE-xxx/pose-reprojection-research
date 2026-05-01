from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np

from pose_reprojection.perspective.config import load_config, save_config
from pose_reprojection.perspective.data import load_sources
from pose_reprojection.perspective.synthetic import (
    generate_synthetic_dataset,
    make_split_indices,
    save_synthetic_dataset,
    load_synthetic_dataset,
    apply_detector_noise_by_split,
)
from pose_reprojection.perspective.lifter import run_frozen_lifter
from pose_reprojection.perspective.train import train_corrector
from pose_reprojection.perspective.evaluate import evaluate_and_save
from pose_reprojection.perspective.visualize import visualize_outputs
from pose_reprojection.perspective.features import prepare_z_features, apply_xgeo_ablation
from pose_reprojection.perspective.geometry import ensure_geometry_arrays, prepare_ray_features
from pose_reprojection.perspective.ray_fit import fit_xgeo_from_rays_and_lifter
from pose_reprojection.perspective.reproducibility import seed_everything


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_subject_summary(arrays):
    subjects = arrays.get("source_subjects", np.array(["unknown"] * arrays["x_gt"].shape[0])).astype(str)
    summary = {}
    for split in ["train", "val", "test"]:
        key = f"{split}_indices"
        if key not in arrays:
            summary[split] = {}
            continue
        counts = {}
        for idx in arrays[key].astype(int).tolist():
            subj = str(subjects[idx])
            counts[subj] = counts.get(subj, 0) + 1
        summary[split] = counts
    return summary


def _write_split_manifest(out_dir, arrays, camera_records):
    rows = []
    split_by_index = {}
    for split in ["train", "val", "test"]:
        for idx in arrays[f"{split}_indices"].astype(int).tolist():
            split_by_index[idx] = split

    source_names = arrays.get("source_names", np.array(["unknown"] * arrays["x_gt"].shape[0])).astype(str)
    subjects = arrays.get("source_subjects", np.array(["unknown"] * arrays["x_gt"].shape[0])).astype(str)
    sequences = arrays.get("source_sequences", np.array(["unknown"] * arrays["x_gt"].shape[0])).astype(str)

    for i in range(arrays["x_gt"].shape[0]):
        rows.append({
            "index": int(i),
            "split": split_by_index.get(i, "unknown"),
            "source_name": str(source_names[i]),
            "subject": str(subjects[i]),
            "sequence": str(sequences[i]),
        })

    summary = {"train": {}, "val": {}, "test": {}}
    for row in rows:
        split = row["split"]
        subj = row["subject"]
        if split in summary:
            summary[split][subj] = summary[split].get(subj, 0) + 1

    (out_dir / "split_manifest.json").write_text(
        json.dumps({"summary_by_subject": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print("[split] summary:", json.dumps(summary, indent=2))
    return summary


def _write_dataset_manifest(out_dir, dataset_path, dataset_hash, arrays, reused):
    manifest = {
        "dataset_path": str(dataset_path),
        "dataset_hash_sha256": dataset_hash,
        "reused": bool(reused),
        "num_sequences": int(arrays["x_gt"].shape[0]),
        "num_frames": int(arrays["x_gt"].shape[1]),
        "split_sizes": {
            split: int(len(arrays[f"{split}_indices"])) for split in ["train", "val", "test"]
        },
        "split_subjects": _split_subject_summary(arrays),
    }
    (out_dir / "synthetic_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _write_z_feature_manifest(out_dir, arrays):
    info = {}
    if "z_feature_info_json" in arrays:
        info = json.loads(str(arrays["z_feature_info_json"]))
    if "z_features" in arrays:
        info["z_features_shape"] = list(arrays["z_features"].shape)
    if "z_feature_permutation" in arrays:
        perm_path = out_dir / "z_feature_permutation.npz"
        np.savez_compressed(perm_path, permutation=arrays["z_feature_permutation"].astype(np.int64))
        info["permutation_path"] = str(perm_path)
        info["permutation_seed"] = int(info.get("shuffle_seed", 0))
    (out_dir / "z_features_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _write_geometry_manifest(out_dir, arrays):
    checks = {}
    if "geometry_checks_json" in arrays:
        checks = json.loads(str(arrays["geometry_checks_json"]))
    (out_dir / "geometry_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    return checks


def _write_ray_feature_manifest(out_dir, arrays):
    info = {}
    if "ray_feature_info_json" in arrays:
        info = json.loads(str(arrays["ray_feature_info_json"]))
    if "ray_features" in arrays:
        info["ray_features_shape"] = list(arrays["ray_features"].shape)
    if "ray_feature_permutation" in arrays:
        perm_path = out_dir / "ray_feature_permutation.npz"
        np.savez_compressed(perm_path, permutation=arrays["ray_feature_permutation"].astype(np.int64))
        info["permutation_path"] = str(perm_path)
    (out_dir / "ray_features_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _write_xgeo_ablation_manifest(out_dir, arrays):
    info = {}
    if "xgeo_ablation_info_json" in arrays:
        info = json.loads(str(arrays["xgeo_ablation_info_json"]))
    (out_dir / "xgeo_ablation_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _json_digest(obj):
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def step_generate(config, out_dir, force=False):
    reuse_path = config.get("dataset", {}).get("reuse_synthetic_dataset")
    if reuse_path:
        dataset_path = Path(reuse_path)
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)
        print("[generate] reusing fixed synthetic dataset:", dataset_path)
        arrays, camera_records = load_synthetic_dataset(dataset_path)
        for key in ["train_indices", "val_indices", "test_indices"]:
            if key not in arrays:
                raise KeyError(f"Reused synthetic dataset is missing {key}: {dataset_path}")
        dataset_hash = _sha256_file(dataset_path)
        split_subjects = _write_split_manifest(out_dir, arrays, camera_records)
        _write_dataset_manifest(out_dir, dataset_path, dataset_hash, arrays, reused=True)
        config["dataset_hash"] = dataset_hash
        config["dataset_source"] = str(dataset_path)
        config["split_subjects"] = split_subjects
        return arrays, camera_records

    dataset_path = out_dir / "synthetic_dataset.npz"
    if dataset_path.exists() and not force:
        print("[generate] using existing:", dataset_path)
        arrays, camera_records = load_synthetic_dataset(dataset_path)
        dataset_hash = _sha256_file(dataset_path)
        split_subjects = _write_split_manifest(out_dir, arrays, camera_records)
        _write_dataset_manifest(out_dir, dataset_path, dataset_hash, arrays, reused=False)
        config["dataset_hash"] = dataset_hash
        config["dataset_source"] = str(dataset_path)
        config["split_subjects"] = split_subjects
        return arrays, camera_records

    sources = load_sources(config)
    arrays, camera_records = generate_synthetic_dataset(config, sources)

    split_indices = make_split_indices(arrays, config)
    arrays["train_indices"] = split_indices["train"]
    arrays["val_indices"] = split_indices["val"]
    arrays["test_indices"] = split_indices["test"]

    arrays = apply_detector_noise_by_split(arrays, config)
    arrays = ensure_geometry_arrays(arrays, config)

    save_synthetic_dataset(dataset_path, arrays, camera_records, split_indices)
    dataset_hash = _sha256_file(dataset_path)
    split_subjects = _write_split_manifest(out_dir, arrays, camera_records)
    _write_dataset_manifest(out_dir, dataset_path, dataset_hash, arrays, reused=False)
    config["dataset_hash"] = dataset_hash
    config["dataset_source"] = str(dataset_path)
    config["split_subjects"] = split_subjects

    print("[generate] saved:", dataset_path)
    print("[generate] dataset_sha256:", dataset_hash)
    print("[generate] x_gt:", arrays["x_gt"].shape)
    print("[generate] train/val/test:", len(split_indices["train"]), len(split_indices["val"]), len(split_indices["test"]))
    return arrays, camera_records


def step_lifter(config, out_dir, arrays, force=False):
    lifter_path = out_dir / "frozen_lifter_outputs.npz"
    transform_path = out_dir / "lifter_alignment.json"

    if lifter_path.exists() and not force:
        print("[lifter] using existing:", lifter_path)
        data = np.load(lifter_path, allow_pickle=False)
        arrays["y_lifted"] = data["y_lifted"].astype(np.float32)
        if arrays["y_lifted"].shape[:2] != arrays["x_gt"].shape[:2]:
            raise ValueError(
                f"Existing lifter output shape {arrays['y_lifted'].shape} does not match dataset {arrays['x_gt'].shape}. "
                "Rerun with --force."
            )
        return arrays

    y, transform = run_frozen_lifter(arrays, config)
    arrays["y_lifted"] = y.astype(np.float32)

    np.savez_compressed(lifter_path, y_lifted=arrays["y_lifted"])
    if transform is not None:
        serial = {
            "scale": float(transform["scale"]),
            "R": np.asarray(transform["R"]).tolist(),
            "t": np.asarray(transform["t"]).tolist(),
        }
        transform_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")

    print("[lifter] saved:", lifter_path)
    print("[lifter] y_lifted:", arrays["y_lifted"].shape)
    return arrays


def step_geometry_refinement(config, out_dir, arrays, force=False):
    refine_cfg = config.get("geometry_refinement", {})
    if not bool(refine_cfg.get("enabled", False)):
        return arrays
    if refine_cfg.get("mode", "ray_depth_fit") != "ray_depth_fit":
        raise ValueError(f"Unknown geometry_refinement.mode: {refine_cfg.get('mode')}")
    if "y_lifted" not in arrays:
        raise KeyError("geometry_refinement requires arrays['y_lifted']")
    if "rays_input" not in arrays:
        arrays = ensure_geometry_arrays(arrays, config)

    cache_path = out_dir / "x_geo_outputs.npz"
    cfg_hash = _json_digest(refine_cfg)
    dataset_hash = str(config.get("dataset_hash", ""))
    if bool(refine_cfg.get("cache", True)) and cache_path.exists() and not force:
        data = np.load(cache_path, allow_pickle=False)
        cache_dataset_hash = str(data["dataset_hash"]) if "dataset_hash" in data else ""
        cache_cfg_hash = str(data["geometry_config_hash"]) if "geometry_config_hash" in data else ""
        if cache_dataset_hash == dataset_hash and cache_cfg_hash == cfg_hash:
            arrays["x_geo"] = data["x_geo"].astype(np.float32)
            if "fit_stats_json" in data:
                arrays["x_geo_fit_stats_json"] = np.array(str(data["fit_stats_json"]))
                (out_dir / "x_geo_fit_stats.json").write_text(
                    json.dumps(json.loads(str(data["fit_stats_json"])), indent=2),
                    encoding="utf-8",
                )
            print("[x_geo] using compatible cache:", cache_path)
            arrays = apply_xgeo_ablation(arrays, config)
            _write_xgeo_ablation_manifest(out_dir, arrays)
            return arrays

    print("[x_geo] fitting ray-depth geometry candidate")
    x_geo, stats = fit_xgeo_from_rays_and_lifter(
        arrays["y_lifted"],
        arrays["rays_input"],
        config,
        camera_params=arrays["z"],
        u_px=arrays["u_px"],
    )
    reproj = stats.get("mean_reprojection_error_to_input_px")
    max_reproj = float(refine_cfg.get("max_reprojection_error_to_input_px", 2.0))
    if reproj is not None and float(reproj) > max_reproj:
        raise RuntimeError(
            "X_geo reprojection-to-input sanity check failed: "
            f"{float(reproj):.4f}px > {max_reproj:.4f}px. Inspect ray/projection convention."
        )

    arrays["x_geo"] = x_geo.astype(np.float32)
    arrays["x_geo_fit_stats_json"] = np.array(json.dumps(stats))
    (out_dir / "x_geo_fit_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    np.savez_compressed(
        cache_path,
        x_geo=arrays["x_geo"],
        dataset_hash=np.array(dataset_hash),
        geometry_config_hash=np.array(cfg_hash),
        fit_stats_json=arrays["x_geo_fit_stats_json"],
    )
    print("[x_geo] saved:", cache_path)
    print("[x_geo] reprojection_to_input_px:", reproj)
    arrays = apply_xgeo_ablation(arrays, config)
    _write_xgeo_ablation_manifest(out_dir, arrays)
    return arrays


def run_all(args):
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    deterministic = bool(config.get("runtime", {}).get("deterministic", True))
    determinism_info = seed_everything(config.get("seed", 1234), deterministic=deterministic)
    config["runtime"] = {
        **config.get("runtime", {}),
        "deterministic": deterministic,
        "determinism_info": determinism_info,
    }

    arrays, _ = step_generate(config, out_dir, force=args.force)
    arrays = ensure_geometry_arrays(arrays, config)
    config["geometry_checks"] = _write_geometry_manifest(out_dir, arrays)
    arrays = prepare_z_features(arrays, config)
    config["z_feature_info"] = _write_z_feature_manifest(out_dir, arrays)
    arrays = prepare_ray_features(arrays, config)
    config["ray_feature_info"] = _write_ray_feature_manifest(out_dir, arrays)
    save_config(config, out_dir / "resolved_config.json")

    if args.generate_only:
        print("[done] generated/reused dataset only:", out_dir)
        return

    arrays = step_lifter(config, out_dir, arrays, force=args.force)
    arrays = step_geometry_refinement(config, out_dir, arrays, force=args.force)
    if "x_geo_fit_stats_json" in arrays:
        config["x_geo_fit_stats"] = json.loads(str(arrays["x_geo_fit_stats_json"]))
    if "xgeo_ablation_info_json" in arrays:
        config["xgeo_ablation_info"] = json.loads(str(arrays["xgeo_ablation_info_json"]))
    save_config(config, out_dir / "resolved_config.json")

    model, _ = train_corrector(arrays, config, out_dir)
    evaluate_and_save(model, arrays, config, out_dir)

    if config.get("visualization", {}).get("enabled", True):
        visualize_outputs(config, out_dir)

    print("[done] output_dir:", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
