from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys


SWEEP = [
    {
        "label": "clean",
        "stem": "perspective_mpi_holdout_clean_cam0_243",
        "detector_noise": False,
        "noise_px": 0.0,
        "dropout": 0.0,
        "frame_shift_px": 0.0,
    },
    {
        "label": "px04_drop03",
        "stem": "perspective_mpi_holdout_detector_noise_cam0_243",
        "detector_noise": True,
        "noise_px": 4.0,
        "dropout": 0.03,
        "frame_shift_px": 1.5,
    },
    {
        "label": "px08_drop05",
        "stem": "perspective_mpi_holdout_noise_px08_drop05_cam0_243",
        "detector_noise": True,
        "noise_px": 8.0,
        "dropout": 0.05,
        "frame_shift_px": 3.0,
    },
    {
        "label": "px16_drop10",
        "stem": "perspective_mpi_holdout_noise_px16_drop10_cam0_243",
        "detector_noise": True,
        "noise_px": 16.0,
        "dropout": 0.10,
        "frame_shift_px": 5.0,
    },
    {
        "label": "px32_drop10",
        "stem": "perspective_mpi_holdout_noise_px32_drop10_cam0_243",
        "detector_noise": True,
        "noise_px": 32.0,
        "dropout": 0.10,
        "frame_shift_px": 8.0,
    },
]


def run_cmd(cmd, dry_run=False):
    print("")
    print("[cmd]", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def pct_change(new, old):
    if old == 0:
        return 0.0
    return 100.0 * (new - old) / old


def make_config_cmd(args, item):
    config_path = Path("configs/poc") / f"{item['stem']}.json"

    cmd = [
        sys.executable,
        "scripts/make_perspective_all_mpi_config.py",
        "--root", str(args.root),
        "--output", str(config_path),
        "--num-frames", str(args.num_frames),
        "--virtual-cameras", str(args.virtual_cameras),
        "--cameras", args.cameras,
        "--split-mode", "subject_holdout",
        "--train-subjects", args.train_subjects,
        "--val-subjects", args.val_subjects,
        "--test-subjects", args.test_subjects,
        "--epochs", str(args.epochs),
        "--train-batch-size", str(args.train_batch_size),
        "--lifter-batch-size", str(args.lifter_batch_size),
    ]

    if item["detector_noise"]:
        cmd += [
            "--detector-noise",
            "--train-noise-px", str(item["noise_px"]),
            "--val-noise-px", str(item["noise_px"]),
            "--test-noise-px", str(item["noise_px"]),
            "--train-dropout", str(item["dropout"]),
            "--val-dropout", str(item["dropout"]),
            "--test-dropout", str(item["dropout"]),
            "--frame-shift-px", str(item["frame_shift_px"]),
        ]

    return cmd, config_path


def run_experiment_cmd(config_path, force):
    cmd = [
        sys.executable,
        "scripts/run_perspective_poc.py",
        "--config", str(config_path),
    ]
    if force:
        cmd.append("--force")
    return cmd


def report_cmd(stem):
    metrics_path = Path("outputs/poc") / stem / "metrics.json"
    report_path = Path("outputs/poc") / stem / "report.md"
    return [
        sys.executable,
        "scripts/report_perspective_metrics.py",
        "--metrics", str(metrics_path),
        "--output", str(report_path),
    ]


def load_metrics(item):
    metrics_path = Path("outputs/poc") / item["stem"] / "metrics.json"
    if not metrics_path.exists():
        return None

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    frozen = data["methods"]["frozen_lifter"]
    corr = data["methods"]["perspective_corrected"]

    row = {
        "label": item["label"],
        "stem": item["stem"],
        "noise_px": item["noise_px"],
        "dropout": item["dropout"],
        "frame_shift_px": item["frame_shift_px"],
        "test_sequences": data.get("num_test_sequences", ""),
        "frozen_mpjpe_mm": frozen["mpjpe_mm"],
        "corrected_mpjpe_mm": corr["mpjpe_mm"],
        "delta_mpjpe_mm": corr["mpjpe_mm"] - frozen["mpjpe_mm"],
        "delta_mpjpe_pct": pct_change(corr["mpjpe_mm"], frozen["mpjpe_mm"]),
        "frozen_pa_mpjpe_mm": frozen["pa_mpjpe_mm"],
        "corrected_pa_mpjpe_mm": corr["pa_mpjpe_mm"],
        "delta_pa_mpjpe_mm": corr["pa_mpjpe_mm"] - frozen["pa_mpjpe_mm"],
        "delta_pa_mpjpe_pct": pct_change(corr["pa_mpjpe_mm"], frozen["pa_mpjpe_mm"]),
        "frozen_pck150": frozen["pck150"],
        "corrected_pck150": corr["pck150"],
        "delta_pck150": corr["pck150"] - frozen["pck150"],
        "frozen_bone_error_mm": frozen["bone_length_error_mm"],
        "corrected_bone_error_mm": corr["bone_length_error_mm"],
        "frozen_reprojection_px": frozen.get("reprojection_error_to_input_px", frozen["reprojection_error_px"]),
        "corrected_reprojection_px": corr.get("reprojection_error_to_input_px", corr["reprojection_error_px"]),
        "frozen_reprojection_clean_px": frozen.get("reprojection_error_to_clean_px", ""),
        "corrected_reprojection_clean_px": corr.get("reprojection_error_to_clean_px", ""),
        "frozen_accel_mm_frame2": frozen["accel_error_mm_per_frame2"],
        "corrected_accel_mm_frame2": corr["accel_error_mm_per_frame2"],
        "correction_mode": data.get("correction_mode", "residual"),
        "normalization_enabled": bool(data.get("corrector_normalization", {}).get("enabled", False)),
        "camera_feature_mode": data.get("camera_feature_mode", "raw_9d"),
        "z_ablation": data.get("z_ablation", "true"),
        "dataset_hash": data.get("dataset_hash", ""),
    }

    bucket_rows = []
    frozen_b = {x["bucket"]: x for x in data["camera_buckets"]["frozen_lifter"]}
    corr_b = {x["bucket"]: x for x in data["camera_buckets"]["perspective_corrected"]}

    for bucket in sorted(set(frozen_b) | set(corr_b)):
        if bucket not in frozen_b or bucket not in corr_b:
            continue
        b = frozen_b[bucket]
        c = corr_b[bucket]
        bucket_rows.append({
            "label": item["label"],
            "noise_px": item["noise_px"],
            "dropout": item["dropout"],
            "frame_shift_px": item["frame_shift_px"],
            "bucket": bucket,
            "num_sequences": c["num_sequences"],
            "frozen_mpjpe_mm": b["mpjpe_mm"],
            "corrected_mpjpe_mm": c["mpjpe_mm"],
            "delta_mpjpe_mm": c["mpjpe_mm"] - b["mpjpe_mm"],
            "frozen_pa_mpjpe_mm": b["pa_mpjpe_mm"],
            "corrected_pa_mpjpe_mm": c["pa_mpjpe_mm"],
            "delta_pa_mpjpe_mm": c["pa_mpjpe_mm"] - b["pa_mpjpe_mm"],
        })

    return row, bucket_rows


def write_aggregate(rows, bucket_rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "noise_sweep_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    bucket_csv = out_dir / "noise_sweep_buckets.csv"
    if bucket_rows:
        with bucket_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(bucket_rows[0].keys()))
            writer.writeheader()
            writer.writerows(bucket_rows)

    md = []
    md.append("# Perspective Pc detector-noise sweep")
    md.append("")
    md.append("| Noise | Dropout | Shift px | Frozen MPJPE | Pc MPJPE | Δ MPJPE | Δ MPJPE % | Frozen PA | Pc PA | Pc PCK@150 | Pc Reproj px | Pc Accel |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        md.append(
            f"| {r['label']} | "
            f"{r['dropout']:.2f} | "
            f"{r['frame_shift_px']:.1f} | "
            f"{r['frozen_mpjpe_mm']:.2f} | "
            f"{r['corrected_mpjpe_mm']:.2f} | "
            f"{r['delta_mpjpe_mm']:.2f} | "
            f"{r['delta_mpjpe_pct']:.2f}% | "
            f"{r['frozen_pa_mpjpe_mm']:.2f} | "
            f"{r['corrected_pa_mpjpe_mm']:.2f} | "
            f"{r['corrected_pck150']:.3f} | "
            f"{r['corrected_reprojection_px']:.2f} | "
            f"{r['corrected_accel_mm_frame2']:.2f} |"
        )

    md.append("")
    md.append("## How to read this")
    md.append("")
    md.append("- MPJPE / PA-MPJPE should increase gradually as noise gets harder.")
    md.append("- Pc should remain better than the frozen lifter at every noise level.")
    md.append("- Reprojection and acceleration are the first places robustness usually breaks.")
    md.append("- If 32 px noise collapses, that is not surprising; it is closer to real detector error scale.")

    md_path = out_dir / "noise_sweep_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("")
    print("[aggregate] wrote:", csv_path)
    print("[aggregate] wrote:", bucket_csv)
    print("[aggregate] wrote:", md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/mpi_inf_3dhp"))
    parser.add_argument("--num-frames", type=int, default=243)
    parser.add_argument("--virtual-cameras", type=int, default=64)
    parser.add_argument("--cameras", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--lifter-batch-size", type=int, default=4)
    parser.add_argument("--train-subjects", type=str, default="S1,S2,S3,S4,S5,S6")
    parser.add_argument("--val-subjects", type=str, default="S7")
    parser.add_argument("--test-subjects", type=str, default="S8")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated labels to run, e.g. px08_drop05,px16_drop10")
    parser.add_argument("--run", action="store_true", help="Actually run training/evaluation, not just configs + aggregate.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--aggregate-dir", type=Path, default=Path("outputs/poc/perspective_noise_sweep_holdout_cam0_243"))
    args = parser.parse_args()

    items = SWEEP
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        items = [x for x in SWEEP if x["label"] in keep]
        missing = keep - {x["label"] for x in items}
        if missing:
            raise ValueError(f"Unknown labels: {sorted(missing)}")

    for item in items:
        metrics_path = Path("outputs/poc") / item["stem"] / "metrics.json"

        make_cmd, config_path = make_config_cmd(args, item)
        run_cmd(make_cmd, dry_run=args.dry_run)

        if args.run:
            if args.skip_existing and metrics_path.exists() and not args.force:
                print("[skip existing]", metrics_path)
            else:
                run_cmd(run_experiment_cmd(config_path, force=args.force), dry_run=args.dry_run)

            if metrics_path.exists() and not args.dry_run:
                run_cmd(report_cmd(item["stem"]), dry_run=args.dry_run)

    rows = []
    bucket_rows = []
    for item in SWEEP:
        loaded = load_metrics(item)
        if loaded is None:
            print("[aggregate] missing metrics:", Path("outputs/poc") / item["stem"] / "metrics.json")
            continue
        row, b_rows = loaded
        rows.append(row)
        bucket_rows.extend(b_rows)

    if rows:
        write_aggregate(rows, bucket_rows, args.aggregate_dir)
    else:
        print("[aggregate] no metrics found yet")


if __name__ == "__main__":
    main()
