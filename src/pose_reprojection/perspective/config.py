from pathlib import Path
import copy
import json


def deep_update(base, update):
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


DEFAULT_CONFIG = {
    "experiment_name": "perspective_oracle_v1",
    "seed": 1234,
    "output_dir": "outputs/poc/perspective_oracle_v1",

    "dataset": {
        "sources": [
            {
                "name": "mpi_s1_seq1",
                "type": "mpi_annot",
                "annot_path": "data/raw/mpi_inf_3dhp/S1/Seq1/annot.mat",
                "camera": 0,
                "start": 0,
                "num_frames": 243
            }
        ],
        "prepared_gt_npz": "outputs/eval/mpi_s1_seq1_cam0_frames0_242_gt.npz",
        "reuse_synthetic_dataset": None,
        "preferred_gt_key": "h36m17_3d_m",
        "root_center": True,
        "max_frames": None
    },

    "synthetic": {
        "num_virtual_cameras": 48,
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
        }
    },

    "lifter": {
        "type": "videopose3d",
        "checkpoint": "checkpoints/videopose3d/pretrained_h36m_cpn.bin",
        "third_party_root": "third_party/VideoPose3D",
        "pad": 121,
        "normalize_screen_coordinates": True,
        "batch_size": 1,
        "device": "auto",
        "alignment": "train_global_similarity",
        "pseudo_bias_strength": 0.25
    },

    "splits": {
        "train_fraction": 0.70,
        "val_fraction": 0.15,
        "test_fraction": 0.15
    },

    "corrector_inputs": {
        "use_lifter_3d": True,
        "use_lifter_2d_normalized": True,
        "use_raw_2d_metadata": True,
        "use_camera_parameters": True,
        "camera_feature_mode": "raw_9d",
        "z_ablation": "true",
        "use_ray_features": False,
        "use_rays": False,
        "ray_ablation": "true",
        "use_geometry_fit_3d": False,
        "use_reliability_features": False
    },

    "geometry_features": {
        "enabled": False,
        "use_input_rays": True,
        "ray_feature_mode": "unit_rays",
        "canonicalize_root_ray": False,
        "root_joint": 0,
        "shuffle_rays": False
    },

    "geometry_refinement": {
        "enabled": False,
        "mode": "ray_depth_fit",
        "root_joint": 0,
        "min_depth_m": 0.5,
        "steps": 60,
        "lr": 0.03,
        "batch_size": 16,
        "device": "auto",
        "losses": {
            "pose_prior": 1.0,
            "bone_y": 0.25,
            "temporal": 0.01,
            "depth_smooth": 0.001
        },
        "cache": True,
        "xgeo_ablation": "none",
        "xgeo_fit_mode": "free_depth",
        "allow_unframed_closest_y": False
    },

    "corrector_normalization": {
        "enabled": False,
        "root_joint": 0,
        "scale_mode": "stable_bones",
        "eps": 0.000001
    },

    "corrector_output": {
        "mode": "residual",
        "base": "y_lifted",
        "gate_init_y_weight": 0.8,
        "gate_mode": "joint_scalar"
    },

    "gate_regularization": {
        "enabled": False,
        "smoothness_weight": 0.0,
        "entropy_weight": 0.0,
        "mean_gate_prior": None,
        "mean_gate_prior_weight": 0.0
    },

    "model": {
        "hidden_dims": [512, 512, 256],
        "dropout": 0.10,
        "zero_init_last": True
    },

    "training": {
        "epochs": 80,
        "batch_size": 4,
        "lr": 0.0005,
        "weight_decay": 0.00001,
        "grad_clip_norm": 1.0,
        "patience": 15,
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


def load_config(path):
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)

    path = Path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return deep_update(DEFAULT_CONFIG, cfg)


def save_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
