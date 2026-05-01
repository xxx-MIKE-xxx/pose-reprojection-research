from pathlib import Path
import argparse
import json


def _csv_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/mpi_inf_3dhp"))
    parser.add_argument("--output", type=Path, default=Path("configs/poc/perspective_mpi_all_cam0_243.json"))
    parser.add_argument("--num-frames", type=int, default=243)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--virtual-cameras", type=int, default=64)
    parser.add_argument("--cameras", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--lifter-batch-size", type=int, default=4)

    parser.add_argument("--split-mode", choices=["random", "subject_holdout"], default="random")
    parser.add_argument("--train-subjects", type=str, default="S1,S2,S3,S4,S5,S6")
    parser.add_argument("--val-subjects", type=str, default="S7")
    parser.add_argument("--test-subjects", type=str, default="S8")

    parser.add_argument("--detector-noise", action="store_true")
    parser.add_argument("--train-noise-px", type=float, default=4.0)
    parser.add_argument("--val-noise-px", type=float, default=4.0)
    parser.add_argument("--test-noise-px", type=float, default=4.0)
    parser.add_argument("--train-dropout", type=float, default=0.03)
    parser.add_argument("--val-dropout", type=float, default=0.03)
    parser.add_argument("--test-dropout", type=float, default=0.03)
    parser.add_argument("--frame-shift-px", type=float, default=1.5)
    args = parser.parse_args()

    cameras = [int(x.strip()) for x in args.cameras.split(",") if x.strip()]

    sources = []
    annot_paths = sorted(args.root.glob("S*/Seq*/annot.mat"))

    for annot_path in annot_paths:
        seq_root = annot_path.parent
        subject = seq_root.parent.name
        seq = seq_root.name

        calib_path = seq_root / "camera.calibration"
        if not calib_path.exists():
            print(f"[SKIP] missing camera.calibration: {seq_root}")
            continue

        for cam in cameras:
            sources.append({
                "name": f"{subject}_{seq}_cam{cam}",
                "subject": subject,
                "sequence": seq,
                "type": "mpi_annot",
                "annot_path": str(annot_path).replace("\\", "/"),
                "camera": cam,
                "start": args.start,
                "num_frames": args.num_frames
            })

    if not sources:
        raise RuntimeError(f"No sources found under {args.root}")

    if args.split_mode == "subject_holdout":
        splits = {
            "mode": "subject_holdout",
            "train_subjects": _csv_list(args.train_subjects),
            "val_subjects": _csv_list(args.val_subjects),
            "test_subjects": _csv_list(args.test_subjects)
        }
    else:
        splits = {
            "mode": "random",
            "train_fraction": 0.65,
            "val_fraction": 0.15,
            "test_fraction": 0.20
        }

    detector_noise = {
        "enabled": bool(args.detector_noise),
        "apply_to_splits": ["train", "val", "test"] if args.detector_noise else [],
        "seed_offset": 10000,
        "profiles": {
            "train": {
                "keypoint_px_std": args.train_noise_px,
                "frame_translation_px_std": args.frame_shift_px,
                "dropout_prob": args.train_dropout,
                "clip_to_image": True
            },
            "val": {
                "keypoint_px_std": args.val_noise_px,
                "frame_translation_px_std": args.frame_shift_px,
                "dropout_prob": args.val_dropout,
                "clip_to_image": True
            },
            "test": {
                "keypoint_px_std": args.test_noise_px,
                "frame_translation_px_std": args.frame_shift_px,
                "dropout_prob": args.test_dropout,
                "clip_to_image": True
            }
        }
    }

    cfg = {
        "experiment_name": args.output.stem,
        "seed": 1234,
        "output_dir": f"outputs/poc/{args.output.stem}",

        "dataset": {
            "prepared_gt_npz": None,
            "preferred_gt_key": "h36m17_3d_m",
            "root_center": True,
            "max_frames": None,
            "sources": sources
        },

        "synthetic": {
            "num_virtual_cameras": args.virtual_cameras,
            "image_width": 2048,
            "image_height": 2048,
            "canonical_camera": {
                "distance_m": 5.0,
                "height_m": 0.0,
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 0.0,
                "fov_deg": 55.0
            },
            "camera_ranges": {
                "distance_m": [2.0, 8.0],
                "height_m": [-0.7, 1.8],
                "yaw_deg": [-55.0, 55.0],
                "pitch_deg": [-35.0, 35.0],
                "roll_deg": [-8.0, 8.0],
                "fov_deg": [35.0, 85.0]
            },
            "noise": {
                "keypoint_px_std": 0.0,
                "dropout_prob": 0.0
            },
            "detector_noise": detector_noise
        },

        "lifter": {
            "type": "videopose3d",
            "checkpoint": "checkpoints/videopose3d/pretrained_h36m_cpn.bin",
            "third_party_root": "third_party/VideoPose3D",
            "pad": 121,
            "normalize_screen_coordinates": True,
            "batch_size": args.lifter_batch_size,
            "device": "auto",
            "alignment": "train_global_similarity",
            "pseudo_bias_strength": 0.25
        },

        "splits": splits,

        "corrector_inputs": {
            "use_lifter_3d": True,
            "use_lifter_2d_normalized": True,
            "use_raw_2d_metadata": True,
            "use_camera_parameters": True
        },

        "model": {
            "hidden_dims": [512, 512, 256],
            "dropout": 0.10,
            "zero_init_last": True
        },

        "training": {
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "lr": 0.0005,
            "weight_decay": 0.00001,
            "grad_clip_norm": 1.0,
            "patience": 12,
            "device": "auto"
        },

        "losses": {
            "mpjpe": {"weight": 1.0},
            "reprojection": {"weight": 0.05, "unit": "normalized_image"},
            "bone": {"weight": 0.05},
            "temporal": {"weight": 0.01, "mode": "match_gt_accel"}
        },

        "evaluation": {
            "pck_threshold_m": 0.150,
            "bucket_thresholds": {
                "close_distance_m": 3.0,
                "low_height_m": -0.25,
                "high_height_m": 1.25,
                "high_abs_yaw_deg": 35.0,
                "high_abs_pitch_deg": 22.0,
                "wide_fov_deg": 70.0
            },
            "external_baseline_jsons": [
                "outputs/eval/baseline_mpi_s1_seq1_cam0_243.json",
                "outputs/eval/baseline_smoothed_mpi_s1_seq1_cam0_243.json",
                "outputs/eval/identity_mpi_s1_seq1_cam0_243.json"
            ]
        },

        "visualization": {
            "enabled": True,
            "frame_index": 0,
            "camera_index": "first_test",
            "make_plots": True
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("wrote:", args.output)
    print("sources:", len(sources))
    print("split mode:", args.split_mode)
    if args.split_mode == "subject_holdout":
        print("train subjects:", cfg["splits"]["train_subjects"])
        print("val subjects:", cfg["splits"]["val_subjects"])
        print("test subjects:", cfg["splits"]["test_subjects"])
    print("detector noise:", args.detector_noise)
    print("virtual cameras per source:", args.virtual_cameras)
    print("total synthetic sequences:", len(sources) * args.virtual_cameras)


if __name__ == "__main__":
    main()
