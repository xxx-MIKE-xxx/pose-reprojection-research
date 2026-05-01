from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared official GT NPZ")
    parser.add_argument("--detections", type=Path, default=None, help="Future cached 2D detections NPZ")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {
        "status": "scaffold_only",
        "dataset": str(args.dataset),
        "detections": str(args.detections) if args.detections else None,
        "num_official_gt_frames": int(data["u_gt2d_px"].shape[0]),
        "image_paths_available": "image_paths" in data.files,
        "next_step": "run or load cached detector keypoints, then reuse official metric code",
    }
    (args.output / "official_detected2d_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
