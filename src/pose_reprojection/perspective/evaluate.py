from pathlib import Path
import csv
import json
import numpy as np
import torch

from .model import build_features
from .features import corrector_y_input, corrector_pose_input, compose_prediction
from .metrics import (
    summarize_method,
    summarize_buckets,
    load_external_baselines,
    H36M17_NAMES,
)


def _gate_stats(gate):
    if gate is None:
        return {}
    vals = np.asarray(gate, dtype=np.float64)
    return {
        "mean_gate_y_weight": float(vals.mean()),
        "std_gate_y_weight": float(vals.std()),
        "min_gate_y_weight": float(vals.min()),
        "max_gate_y_weight": float(vals.max()),
    }


def predict_corrected(model, arrays, config, indices=None, device=None, return_aux=False):
    if indices is None:
        indices = np.arange(arrays["x_gt"].shape[0])
    indices = np.asarray(indices, dtype=np.int64)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    outs = []
    gate_outs = []

    with torch.no_grad():
        for idx in indices:
            y = torch.from_numpy(arrays["y_lifted"][idx:idx + 1]).float().to(device)
            u = torch.from_numpy(arrays["u_norm"][idx:idx + 1]).float().to(device)
            m = torch.from_numpy(arrays["raw_2d_metadata"][idx:idx + 1]).float().to(device)
            z_features = torch.from_numpy(arrays["z_features"][idx:idx + 1]).float().to(device)
            ray_features = None
            if "ray_features" in arrays:
                ray_features = torch.from_numpy(arrays["ray_features"][idx:idx + 1]).float().to(device)
            x_geo = None
            x_geo_features = None
            x_geo_key = "x_geo_used" if "x_geo_used" in arrays else "x_geo"
            if x_geo_key in arrays:
                x_geo = torch.from_numpy(arrays[x_geo_key][idx:idx + 1]).float().to(device)
                x_geo_features = corrector_pose_input(x_geo, config)
            feat = build_features(
                corrector_y_input(y, config),
                u,
                m,
                z_features,
                config["corrector_inputs"],
                ray_features=ray_features,
                x_geo_features=x_geo_features,
            )
            model_output = model(feat)
            x_hat = compose_prediction(y, model_output, config, x_geo=x_geo)
            outs.append(x_hat.cpu().numpy()[0])
            if isinstance(model_output, dict) and model_output.get("gate") is not None:
                gate_outs.append(model_output["gate"].cpu().numpy()[0])

    x_hat = np.stack(outs, axis=0).astype(np.float32)
    if not return_aux:
        return x_hat
    aux = {}
    if gate_outs:
        aux["gate_y_weight"] = np.stack(gate_outs, axis=0).astype(np.float32)
        aux["gate_stats"] = _gate_stats(aux["gate_y_weight"])
    return x_hat, aux


def evaluate_and_save(model, arrays, config, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_idx = arrays["test_indices"].astype(np.int64)
    if len(test_idx) == 0:
        raise RuntimeError("No test split items available.")

    device = torch.device(config["training"].get("device", "auto") if config["training"].get("device") != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if str(device) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_hat, pred_aux = predict_corrected(model, arrays, config, test_idx, device=device, return_aux=True)
    y = arrays["y_lifted"][test_idx]
    gt = arrays["x_gt"][test_idx]
    u_px = arrays["u_px"][test_idx]
    u_px_clean = arrays["u_px_clean"][test_idx] if "u_px_clean" in arrays else None
    z = arrays["z"][test_idx]

    pck_thr = float(config["evaluation"].get("pck_threshold_m", 0.150))

    methods = {
        "frozen_lifter": summarize_method(y, gt, u_px, z, pck_threshold=pck_thr, u_px_clean=u_px_clean),
        "perspective_corrected": summarize_method(x_hat, gt, u_px, z, pck_threshold=pck_thr, u_px_clean=u_px_clean),
    }
    buckets = {
        "frozen_lifter": summarize_buckets(y, gt, z, config["evaluation"]["bucket_thresholds"]),
        "perspective_corrected": summarize_buckets(x_hat, gt, z, config["evaluation"]["bucket_thresholds"]),
    }
    x_geo_raw = arrays["x_geo"][test_idx] if "x_geo" in arrays else None
    x_geo_used = arrays["x_geo_used"][test_idx] if "x_geo_used" in arrays else x_geo_raw
    if x_geo_raw is not None:
        raw_metrics = summarize_method(x_geo_raw, gt, u_px, z, pck_threshold=pck_thr, u_px_clean=u_px_clean)
        methods["x_geo"] = raw_metrics
        methods["x_geo_raw"] = raw_metrics
        raw_buckets = summarize_buckets(x_geo_raw, gt, z, config["evaluation"]["bucket_thresholds"])
        buckets["x_geo"] = raw_buckets
        buckets["x_geo_raw"] = raw_buckets
    if x_geo_used is not None:
        methods["x_geo_used"] = summarize_method(x_geo_used, gt, u_px, z, pck_threshold=pck_thr, u_px_clean=u_px_clean)
        buckets["x_geo_used"] = summarize_buckets(x_geo_used, gt, z, config["evaluation"]["bucket_thresholds"])

    ray_info = {}
    if "ray_feature_info_json" in arrays:
        ray_info = json.loads(str(arrays["ray_feature_info_json"]))
    geometry_checks = {}
    if "geometry_checks_json" in arrays:
        geometry_checks = json.loads(str(arrays["geometry_checks_json"]))
    x_geo_fit_stats = {}
    if "x_geo_fit_stats_json" in arrays:
        x_geo_fit_stats = json.loads(str(arrays["x_geo_fit_stats_json"]))
    xgeo_ablation_info = {}
    if "xgeo_ablation_info_json" in arrays:
        xgeo_ablation_info = json.loads(str(arrays["xgeo_ablation_info_json"]))
    else:
        mode = config.get("geometry_refinement", {}).get("xgeo_ablation", "none")
        xgeo_ablation_info = {
            "xgeo_ablation": mode,
            "xgeo_ablation_applied": False,
            "xgeo_ablation_source": "true_x_geo" if mode == "none" else "unknown",
        }

    metrics = {
        "experiment_name": config["experiment_name"],
        "num_test_sequences": int(len(test_idx)),
        "methods": methods,
        "camera_buckets": buckets,
        "external_baselines": load_external_baselines(config["evaluation"].get("external_baseline_jsons", [])),
        "correction_mode": config.get("corrector_output", {}).get("mode", "residual"),
        "corrector_output_base": config.get("corrector_output", {}).get("base", "y_lifted"),
        "gate_mode": config.get("corrector_output", {}).get("gate_mode", "joint_scalar"),
        "gate_init_y_weight": float(config.get("corrector_output", {}).get("gate_init_y_weight", 0.8)),
        "corrector_normalization": config.get("corrector_normalization", {"enabled": False}),
        "camera_feature_mode": config.get("corrector_inputs", {}).get("camera_feature_mode", "raw_9d"),
        "z_ablation": config.get("corrector_inputs", {}).get("z_ablation", "true"),
        "use_ray_features": bool(config.get("corrector_inputs", {}).get("use_ray_features", False)),
        "ray_ablation": config.get("corrector_inputs", {}).get("ray_ablation", "true"),
        "ray_feature_info": ray_info,
        "ray_feature_mode": config.get("geometry_features", {}).get("ray_feature_mode", "unit_rays"),
        "canonicalize_root_ray": bool(config.get("geometry_features", {}).get("canonicalize_root_ray", False)),
        "geometry_features": config.get("geometry_features", {"enabled": False}),
        "geometry_checks": geometry_checks,
        "geometry_refinement": config.get("geometry_refinement", {"enabled": False}),
        "geometry_refinement_mode": config.get("geometry_refinement", {}).get("mode", "ray_depth_fit"),
        "xgeo_ablation": xgeo_ablation_info.get("xgeo_ablation", "none"),
        "xgeo_ablation_applied": bool(xgeo_ablation_info.get("xgeo_ablation_applied", False)),
        "xgeo_ablation_source": xgeo_ablation_info.get("xgeo_ablation_source", "true_x_geo"),
        "xgeo_ablation_info": xgeo_ablation_info,
        "use_geometry_fit_3d": bool(config.get("corrector_inputs", {}).get("use_geometry_fit_3d", False)),
        "x_geo_fit_stats": x_geo_fit_stats,
        "dataset_hash": config.get("dataset_hash"),
        "dataset_source": config.get("dataset_source"),
        "split_subjects": config.get("split_subjects"),
        "notes": [
            f"Pc correction mode: {config.get('corrector_output', {}).get('mode', 'residual')}.",
            "Frozen lifter outputs are root-centered and optionally train-split globally aligned before Pc training.",
            "External raw/smoothed/identity baselines are included only if their JSON paths exist."
        ],
    }

    if pred_aux.get("gate_stats"):
        metrics.update(pred_aux["gate_stats"])

    if "x_geo_raw" in methods:
        geo = methods["x_geo_raw"]
        metrics.update({
            "xgeo_mpjpe_mm": geo.get("mpjpe_mm"),
            "xgeo_pa_mpjpe_mm": geo.get("pa_mpjpe_mm"),
            "xgeo_pck150": geo.get("pck150"),
            "xgeo_bone_length_error_mm": geo.get("bone_length_error_mm"),
            "xgeo_reprojection_error_to_input_px": geo.get("reprojection_error_to_input_px"),
            "xgeo_reprojection_error_to_clean_px": geo.get("reprojection_error_to_clean_px"),
            "xgeo_accel_error_mm_per_frame2": geo.get("accel_error_mm_per_frame2"),
            "xgeo_raw_mpjpe_mm": geo.get("mpjpe_mm"),
            "xgeo_raw_pa_mpjpe_mm": geo.get("pa_mpjpe_mm"),
            "xgeo_raw_pck150": geo.get("pck150"),
            "xgeo_raw_reprojection_error_to_input_px": geo.get("reprojection_error_to_input_px"),
            "xgeo_raw_reprojection_error_to_clean_px": geo.get("reprojection_error_to_clean_px"),
            "xgeo_raw_accel_error_mm_per_frame2": geo.get("accel_error_mm_per_frame2"),
        })
    if "x_geo_used" in methods:
        geo = methods["x_geo_used"]
        metrics.update({
            "xgeo_used_mpjpe_mm": geo.get("mpjpe_mm"),
            "xgeo_used_pa_mpjpe_mm": geo.get("pa_mpjpe_mm"),
            "xgeo_used_pck150": geo.get("pck150"),
            "xgeo_used_reprojection_error_to_input_px": geo.get("reprojection_error_to_input_px"),
            "xgeo_used_reprojection_error_to_clean_px": geo.get("reprojection_error_to_clean_px"),
            "xgeo_used_accel_error_mm_per_frame2": geo.get("accel_error_mm_per_frame2"),
        })

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Per-joint CSV.
    with (out_dir / "per_joint_error.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["joint", "frozen_lifter_mm", "perspective_corrected_mm", "delta_mm"])
        writer.writeheader()
        base = metrics["methods"]["frozen_lifter"]["per_joint_error_mm"]
        corr = metrics["methods"]["perspective_corrected"]["per_joint_error_mm"]
        for name in H36M17_NAMES:
            writer.writerow({
                "joint": name,
                "frozen_lifter_mm": base[name],
                "perspective_corrected_mm": corr[name],
                "delta_mm": corr[name] - base[name],
            })

    # Bucket CSV.
    rows = []
    for method, vals in metrics["camera_buckets"].items():
        for row in vals:
            rows.append({"method": method, **row})
    if rows:
        with (out_dir / "bucket_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r.keys()}))
            writer.writeheader()
            writer.writerows(rows)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        test_indices=test_idx,
        x_gt=gt,
        y_lifted=y,
        x_hat=x_hat,
        x_geo=x_geo_raw if x_geo_raw is not None else np.array([], dtype=np.float32),
        x_geo_used=x_geo_used if x_geo_used is not None else np.array([], dtype=np.float32),
        gate_y_weight=pred_aux.get("gate_y_weight", np.array([], dtype=np.float32)),
        u_px=u_px,
        u_px_clean=u_px_clean if u_px_clean is not None else np.array([], dtype=np.float32),
        u_norm=arrays["u_norm"][test_idx],
        z=z,
        z_features=arrays["z_features"][test_idx],
        ray_features=arrays["ray_features"][test_idx] if "ray_features" in arrays else np.array([], dtype=np.float32),
        canonical_2d_px=arrays["canonical_2d_px"][test_idx],
    )

    print(json.dumps({
        "frozen_lifter_mpjpe_mm": metrics["methods"]["frozen_lifter"]["mpjpe_mm"],
        "corrected_mpjpe_mm": metrics["methods"]["perspective_corrected"]["mpjpe_mm"],
        "xgeo_raw_mpjpe_mm": metrics["methods"].get("x_geo_raw", {}).get("mpjpe_mm"),
        "xgeo_used_mpjpe_mm": metrics["methods"].get("x_geo_used", {}).get("mpjpe_mm"),
        "xgeo_ablation": metrics.get("xgeo_ablation"),
        "mean_gate_y_weight": metrics.get("mean_gate_y_weight"),
        "frozen_lifter_pa_mpjpe_mm": metrics["methods"]["frozen_lifter"]["pa_mpjpe_mm"],
        "corrected_pa_mpjpe_mm": metrics["methods"]["perspective_corrected"]["pa_mpjpe_mm"],
        "metrics_path": str(out_dir / "metrics.json"),
    }, indent=2))

    return metrics
