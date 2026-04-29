from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from pose_reprojection.core.keypoint_io import load_keypoint_npz, save_keypoint_npz
from pose_reprojection.poc.registry import get_method

def load_config(path):
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load_keypoint_npz(args.input)
    config = load_config(args.config)
    method = get_method(args.method)

    out = method(data, config)
    save_keypoint_npz(args.output, out)

    print("method:", args.method)
    print("input:", args.input)
    print("output:", args.output)
    print("input keypoints:", data["keypoints"].shape)
    print("output keypoints:", out["keypoints"].shape)

if __name__ == "__main__":
    main()
