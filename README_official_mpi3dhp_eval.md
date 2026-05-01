# Official MPI-INF-3DHP Test-Set Evaluation

This pipeline prepares and evaluates the official MPI-INF-3DHP test set without using synthetic virtual-camera labels. The expected layout is:

```text
data/raw/mpi_inf_3dhp/mpi_inf_3dhp_test_set/
  TS1/
  TS2/
  TS3/
  TS4/
  TS5/
  TS6/
  test_util/
```

The annotation files are MATLAB v7.3 HDF5 files, so the loader uses `h5py` rather than `scipy.io.loadmat`.

## Protocol

The implementation follows the MATLAB utilities in `test_util`:

- Official GT already contains 17 relevant joints in official order.
- The official root/pelvis is MATLAB joint 15, which is Python index `14`.
- Prediction and GT are root-centered at index `14` before per-joint errors.
- PCK thresholds are `0:5:150` mm.
- PCK@150 is reported at `150` mm.
- AUC is `100 * mean(PCK curve over thresholds 0:5:150)`.
- Activities are the official seven labels from `mpii_get_activity_name.m`.
- Joint groups follow `mpii_get_pck_auc_joint_groups.m` with zero-based Python indices.

TS1-TS6 are the official test set. They must never be used for training, validation, checkpoint selection, feature normalization fitting, or hyperparameter selection.

## Prepare GT-2D/GT-3D

```powershell
python scripts\prepare_mpi3dhp_official_eval.py `
  --root data/raw/mpi_inf_3dhp `
  --output outputs/official_mpi3dhp/official_test_gt2d.npz
```

This writes `official_test_gt2d.npz` plus `official_test_gt2d.manifest.json`. The NPZ stores official GT 2D pixels, universal GT 3D in millimeters, valid-frame metadata, sequence/frame/activity labels, image paths, image sizes, joint names, root index, and annotation hashes.

## GT-2D Sanity Evaluation

The GT-2D path measures the frozen lifter and optional Pc using official ground-truth 2D keypoints. This is a sanity/evaluation bridge, not a deployable real-video result.

```powershell
python scripts\evaluate_mpi3dhp_official_gt2d.py `
  --dataset outputs/official_mpi3dhp/official_test_gt2d.npz `
  --config configs/poc/perspective_official_mpi3dhp_gt2d_h_noz.json `
  --baseline-only `
  --output outputs/official_mpi3dhp/eval_gt2d_frozen_lifter
```

The script saves `metrics.json`, `report.md`, `predictions.npz`, and `official_eval_manifest.json`. The manifest records `no_oracle_z=true`, dataset/config hashes, the official root index, and PCK thresholds.

## Pc/PAGRC Evaluation

Pc checkpoints can be evaluated when they avoid synthetic/oracle camera `z`, or when `--disable-z` is used to replace z features with zeros. Ray-feature H configs use official test-util camera calibration when it can be parsed; otherwise the script falls back to the focal constants from `mpii_perspective_correction_code.m` and image-center principal points. The output manifest records `intrinsics_mode` as either `official_calibration` or `official_test_util_fallback`.

The H/ray/X_geo path fits `X_geo` from the frozen lifter output and official GT-2D rays only. It does not use test-set GT 3D except for final metrics. Official eval defaults to `--xgeo-fit-mode closest_y`, which solves ray depths whose root-relative shape is closest to the frozen lifter and then stores `X_geo_used` in the same root-relative coordinate convention as `Y`; `--xgeo-fit-mode free_depth` preserves the older camera-absolute diagnostic path.

```powershell
python scripts\evaluate_mpi3dhp_official_gt2d.py `
  --dataset outputs/official_mpi3dhp/official_test_gt2d.npz `
  --config configs/poc/perspective_v3_px08_H_true_rays_noz.json `
  --pc-checkpoint outputs/poc/perspective_v3_px08_H_true_rays_noz/best_corrector.pt `
  --disable-z `
  --output outputs/official_mpi3dhp/eval_gt2d_pagrc_h_noz
```

For incompatible configs, the script fails with a clear error instead of silently running baseline-only or using oracle synthetic geometry. The GT-2D PAGRC-H result remains a sanity evaluation, not a deployable detected-video benchmark.

### Official Pc Debug Modes

Use `--base-only` to run feature construction plus gate/base composition while forcing the applied residual to zero. Use `--residual-alpha` to evaluate extra diagnostic outputs of the form `X_base + alpha * dX`. Use `--xgeo-ablation none|y_lifted|zero` to override the config's `X_geo` used by Pc features and base composition at official-eval time.

```powershell
python scripts\evaluate_mpi3dhp_official_gt2d.py `
  --dataset outputs/official_mpi3dhp/official_test_gt2d.npz `
  --config configs/poc/perspective_v3_px08_H_true_rays_noz.json `
  --pc-checkpoint outputs/poc/perspective_v3_px08_H_true_rays_noz/best_corrector.pt `
  --disable-z `
  --xgeo-fit-mode closest_y `
  --residual-alpha 0 0.25 0.5 1.0 `
  --xgeo-ablation none `
  --output outputs/official_mpi3dhp/eval_gt2d_pagrc_h_noz_debug
```

`predictions.npz` includes `x_geo_camera_abs_official_mm`, `x_geo_raw_official_mm`, `x_geo_used_official_mm`, `gated_base_official_mm`, `residual_official_mm`, `xgeo_depths_mm`, `xgeo_depth_prior_mm`, `xgeo_fit_mode`, and stacked `residual_alpha_official_mm` diagnostics. The report lists MPJPE/PCK/AUC for the frozen lifter, camera-absolute/root-centered `X_geo`, raw/used `X_geo`, `X_base`, each residual-alpha output, and final `X_hat`. PA-MPJPE is included only as a fixed diagnostic Procrustes-aligned number; it is not part of the official 3DHP PCK/AUC protocol.

## Detected-2D Evaluation

`scripts/evaluate_mpi3dhp_official_detected2d.py` is a scaffold for the deployable setting. It loads official image paths and creates a manifest for future cached detector outputs. The GT-2D path above should not be described as a detected-video deployment result.
