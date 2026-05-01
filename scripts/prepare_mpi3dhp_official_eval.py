from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np

from pose_reprojection.perspective.mpi3dhp_official import (
    prepare_official_gt2d_gt3d,
    sha256_file,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="MPI-INF-3DHP root or mpi_inf_3dhp_test_set directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    arrays, manifest = prepare_official_gt2d_gt3d(args.root)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest["prepared_npz"] = str(args.output)
    manifest["prepared_npz_sha256"] = sha256_file(args.output)

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    counts = manifest["sequence_valid_counts"]
    for seq in sorted(counts):
        print(f"{seq}: {counts[seq]} valid frames")
    print("total_valid_frames:", manifest["num_valid_frames"])
    print("wrote:", args.output)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
