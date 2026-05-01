from pathlib import Path
import argparse
import csv
import json


def _metric(method, key, default=""):
    return method.get(key, method.get("reprojection_error_px", default) if key == "reprojection_error_to_input_px" else default)


def load_row(path):
    path = Path(path)
    metrics_path = path / "metrics.json" if path.is_dir() else path
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    frozen = data["methods"]["frozen_lifter"]
    xgeo = data["methods"].get("x_geo", {})
    corr = data["methods"]["perspective_corrected"]
    ray_info = data.get("ray_feature_info", {})
    geom_ref = data.get("geometry_refinement", {})

    return {
        "experiment_name": data.get("experiment_name", metrics_path.parent.name),
        "metrics_path": str(metrics_path),
        "test_sequences": data.get("num_test_sequences", ""),
        "correction_mode": data.get("correction_mode", "residual"),
        "corrector_output_base": data.get("corrector_output_base", "y_lifted"),
        "gate_mode": data.get("gate_mode", ""),
        "gate_init_y_weight": data.get("gate_init_y_weight", ""),
        "mean_gate_y_weight": data.get("mean_gate_y_weight", ""),
        "std_gate_y_weight": data.get("std_gate_y_weight", ""),
        "min_gate_y_weight": data.get("min_gate_y_weight", ""),
        "max_gate_y_weight": data.get("max_gate_y_weight", ""),
        "normalization_enabled": bool(data.get("corrector_normalization", {}).get("enabled", False)),
        "camera_feature_mode": data.get("camera_feature_mode", "raw_9d"),
        "z_ablation": data.get("z_ablation", "true"),
        "use_ray_features": bool(data.get("use_ray_features", False)),
        "ray_feature_mode": data.get("ray_feature_mode", "unit_rays"),
        "ray_ablation": data.get("ray_ablation", "true"),
        "canonicalize_root_ray": bool(data.get("canonicalize_root_ray", False)),
        "ray_feature_dim": ray_info.get("ray_feature_dim", ""),
        "ray_permutation_hash": ray_info.get("ray_permutation_hash", ""),
        "geometry_refinement_enabled": bool(geom_ref.get("enabled", False)),
        "geometry_refinement_mode": data.get("geometry_refinement_mode", geom_ref.get("mode", "")),
        "use_geometry_fit_3d": bool(data.get("use_geometry_fit_3d", False)),
        "dataset_hash": data.get("dataset_hash", ""),
        "frozen_mpjpe_mm": frozen["mpjpe_mm"],
        "xgeo_mpjpe_mm": xgeo.get("mpjpe_mm", ""),
        "corrected_mpjpe_mm": corr["mpjpe_mm"],
        "delta_mpjpe_mm": corr["mpjpe_mm"] - frozen["mpjpe_mm"],
        "frozen_pa_mpjpe_mm": frozen["pa_mpjpe_mm"],
        "xgeo_pa_mpjpe_mm": xgeo.get("pa_mpjpe_mm", ""),
        "corrected_pa_mpjpe_mm": corr["pa_mpjpe_mm"],
        "delta_pa_mpjpe_mm": corr["pa_mpjpe_mm"] - frozen["pa_mpjpe_mm"],
        "frozen_pck150": frozen["pck150"],
        "xgeo_pck150": xgeo.get("pck150", ""),
        "corrected_pck150": corr["pck150"],
        "delta_pck150": corr["pck150"] - frozen["pck150"],
        "frozen_reprojection_input_px": _metric(frozen, "reprojection_error_to_input_px"),
        "xgeo_reprojection_input_px": _metric(xgeo, "reprojection_error_to_input_px", ""),
        "corrected_reprojection_input_px": _metric(corr, "reprojection_error_to_input_px"),
        "frozen_reprojection_clean_px": frozen.get("reprojection_error_to_clean_px", ""),
        "xgeo_reprojection_clean_px": xgeo.get("reprojection_error_to_clean_px", ""),
        "corrected_reprojection_clean_px": corr.get("reprojection_error_to_clean_px", ""),
        "corrected_reprojection_error_to_clean_px": corr.get("reprojection_error_to_clean_px", ""),
        "frozen_accel_mm_frame2": frozen["accel_error_mm_per_frame2"],
        "xgeo_accel_mm_frame2": xgeo.get("accel_error_mm_per_frame2", ""),
        "corrected_accel_mm_frame2": corr["accel_error_mm_per_frame2"],
        "corrected_accel_error_mm_per_frame2": corr["accel_error_mm_per_frame2"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", nargs="+", required=True, help="Experiment directories or metrics.json files")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [load_row(p) for p in args.outputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("wrote:", args.output)


if __name__ == "__main__":
    main()
