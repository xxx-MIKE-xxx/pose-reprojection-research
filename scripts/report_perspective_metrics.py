from pathlib import Path
import argparse
import json


def pct_change(new, old):
    if old == 0:
        return 0.0
    return 100.0 * (new - old) / old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.metrics.read_text(encoding="utf-8"))

    frozen = data["methods"]["frozen_lifter"]
    xgeo = data["methods"].get("x_geo")
    corr = data["methods"]["perspective_corrected"]

    metric_keys = [
        ("MPJPE mm", "mpjpe_mm"),
        ("PA-MPJPE mm", "pa_mpjpe_mm"),
        ("PCK@150", "pck150"),
        ("PA-PCK@150", "pck150_pa"),
        ("Bone length error mm", "bone_length_error_mm"),
        ("Reprojection error to input px", "reprojection_error_to_input_px"),
        ("Reprojection error to clean px", "reprojection_error_to_clean_px"),
        ("Accel error mm/frame^2", "accel_error_mm_per_frame2"),
    ]

    lines = []
    lines.append(f"# Perspective POC cumulative report")
    lines.append("")
    lines.append(f"Experiment: `{data.get('experiment_name')}`")
    lines.append(f"Test synthetic sequences: `{data.get('num_test_sequences')}`")
    lines.append(f"Correction mode: `{data.get('correction_mode', 'residual')}`")
    lines.append(f"Correction base: `{data.get('corrector_output_base', 'y_lifted')}`")
    lines.append(f"Gate mode: `{data.get('gate_mode', 'joint_scalar')}`")
    lines.append(f"Gate init Y weight: `{data.get('gate_init_y_weight', 0.8)}`")
    if data.get("mean_gate_y_weight") is not None:
        lines.append(
            "Gate Y weight stats: "
            f"mean `{data.get('mean_gate_y_weight'):.4f}`, "
            f"std `{data.get('std_gate_y_weight'):.4f}`, "
            f"min `{data.get('min_gate_y_weight'):.4f}`, "
            f"max `{data.get('max_gate_y_weight'):.4f}`"
        )
    norm = data.get("corrector_normalization", {})
    lines.append(f"Corrector normalization: `{bool(norm.get('enabled', False))}`")
    lines.append(f"Camera feature mode: `{data.get('camera_feature_mode', 'raw_9d')}`")
    lines.append(f"z ablation: `{data.get('z_ablation', 'true')}`")
    lines.append(f"Ray features: `{bool(data.get('use_ray_features', False))}`")
    lines.append(f"Ray feature mode: `{data.get('ray_feature_mode', 'unit_rays')}`")
    lines.append(f"Ray ablation: `{data.get('ray_ablation', 'true')}`")
    lines.append(f"Root-ray canonicalization: `{bool(data.get('canonicalize_root_ray', False))}`")
    geom_ref = data.get("geometry_refinement", {})
    lines.append(f"Geometry refinement: `{bool(geom_ref.get('enabled', False))}`")
    lines.append(f"Geometry refinement mode: `{data.get('geometry_refinement_mode', geom_ref.get('mode', 'ray_depth_fit'))}`")
    if data.get("dataset_hash"):
        lines.append(f"Dataset SHA256: `{data.get('dataset_hash')}`")
    if data.get("split_subjects"):
        lines.append(f"Split subjects: `{json.dumps(data.get('split_subjects'), sort_keys=True)}`")
    if data.get("ray_feature_info", {}).get("ray_feature_dim") is not None:
        lines.append(f"Ray feature dim: `{data.get('ray_feature_info', {}).get('ray_feature_dim')}`")
    if data.get("ray_feature_info", {}).get("ray_permutation_hash"):
        lines.append(f"Ray permutation hash: `{data.get('ray_feature_info', {}).get('ray_permutation_hash')}`")
    if data.get("x_geo_fit_stats", {}).get("mean_reprojection_error_to_input_px") is not None:
        lines.append(
            "X_geo fit reprojection-to-input px: "
            f"`{data.get('x_geo_fit_stats', {}).get('mean_reprojection_error_to_input_px'):.6f}`"
        )
    lines.append("")
    if xgeo is None:
        lines.append("| Metric | Frozen lifter | Pc corrected | Delta | Delta % |")
        lines.append("|---|---:|---:|---:|---:|")
    else:
        lines.append("| Metric | Frozen lifter | X_geo only | Pc corrected | Pc delta vs frozen | Pc delta % |")
        lines.append("|---|---:|---:|---:|---:|---:|")

    for name, key in metric_keys:
        if key == "reprojection_error_to_input_px":
            base = frozen.get(key, frozen.get("reprojection_error_px"))
            new = corr.get(key, corr.get("reprojection_error_px"))
            geo = xgeo.get(key, xgeo.get("reprojection_error_px")) if xgeo is not None else None
        else:
            if key not in frozen or key not in corr:
                continue
            base = frozen[key]
            new = corr[key]
            geo = xgeo.get(key) if xgeo is not None else None
        delta = new - base
        if xgeo is None:
            lines.append(f"| {name} | {base:.4f} | {new:.4f} | {delta:.4f} | {pct_change(new, base):.2f}% |")
        else:
            geo_cell = "" if geo is None else f"{geo:.4f}"
            lines.append(
                f"| {name} | {base:.4f} | {geo_cell} | {new:.4f} | "
                f"{delta:.4f} | {pct_change(new, base):.2f}% |"
            )

    lines.append("")
    lines.append("## Camera buckets")
    lines.append("")
    lines.append("| Bucket | N | Frozen MPJPE | Corrected MPJPE | Delta | Frozen PA-MPJPE | Corrected PA-MPJPE | Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    frozen_buckets = {x["bucket"]: x for x in data["camera_buckets"]["frozen_lifter"]}
    corr_buckets = {x["bucket"]: x for x in data["camera_buckets"]["perspective_corrected"]}

    for bucket in sorted(set(frozen_buckets) | set(corr_buckets)):
        b = frozen_buckets.get(bucket)
        c = corr_buckets.get(bucket)
        if b is None or c is None:
            continue
        lines.append(
            f"| {bucket} | {c['num_sequences']} | "
            f"{b['mpjpe_mm']:.3f} | {c['mpjpe_mm']:.3f} | {c['mpjpe_mm'] - b['mpjpe_mm']:.3f} | "
            f"{b['pa_mpjpe_mm']:.3f} | {c['pa_mpjpe_mm']:.3f} | {c['pa_mpjpe_mm'] - b['pa_mpjpe_mm']:.3f} |"
        )

    lines.append("")
    lines.append("## Worst remaining corrected per-joint errors")
    lines.append("")
    joints = corr["per_joint_error_mm"]
    for name, value in sorted(joints.items(), key=lambda kv: kv[1], reverse=True)[:8]:
        base = frozen["per_joint_error_mm"][name]
        lines.append(f"- `{name}`: corrected `{value:.2f} mm`, frozen `{base:.2f} mm`, delta `{value - base:.2f} mm`")

    text = "\n".join(lines)

    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print("wrote:", args.output)


if __name__ == "__main__":
    main()
