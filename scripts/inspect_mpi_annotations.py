from pathlib import Path
import re
import numpy as np
from scipy.io import loadmat

ANNOT_PATH = Path("data/raw/mpi_inf_3dhp/S1/Seq1/annot.mat")
CALIB_PATH = Path("data/raw/mpi_inf_3dhp/S1/Seq1/camera.calibration")

def describe_obj(name, obj, indent=0):
    pad = " " * indent
    print(f"{pad}{name}: type={type(obj)}")
    if isinstance(obj, np.ndarray):
        print(f"{pad}  shape={obj.shape}, dtype={obj.dtype}")
        if obj.dtype == object:
            flat = obj.flatten()
            for i in range(min(8, len(flat))):
                item = flat[i]
                print(f"{pad}  object[{i}]: type={type(item)}")
                if isinstance(item, np.ndarray):
                    print(f"{pad}    shape={item.shape}, dtype={item.dtype}")
                    squeezed = np.asarray(item).squeeze()
                    print(f"{pad}    squeezed={squeezed.shape}, dtype={squeezed.dtype}")
        else:
            squeezed = np.asarray(obj).squeeze()
            print(f"{pad}  squeezed={squeezed.shape}")
            if squeezed.size:
                print(f"{pad}  min={np.nanmin(squeezed):.3f}, max={np.nanmax(squeezed):.3f}")

def parse_floats(line):
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]

def inspect_calibration(path):
    print("\n== camera.calibration ==")
    if not path.exists():
        print(f"Missing: {path}")
        return

    lines = path.read_text(errors="ignore").splitlines()
    print("line count:", len(lines))
    print("first 10 lines:")
    for i, line in enumerate(lines[:10]):
        print(f"{i:02d}: {line}")

    for cam in [0, 1, 2, 4, 5, 6, 7, 8]:
        start = cam * 7
        if start + 6 >= len(lines):
            continue
        size_nums = parse_floats(lines[start + 3])
        k_nums = parse_floats(lines[start + 5])
        rt_nums = parse_floats(lines[start + 6])
        print(
            f"cam {cam}: "
            f"size_nums={size_nums[-2:] if len(size_nums) >= 2 else size_nums}, "
            f"K_count={len(k_nums)}, RT_count={len(rt_nums)}"
        )

def main():
    print("== annot.mat ==")
    if not ANNOT_PATH.exists():
        raise FileNotFoundError(ANNOT_PATH)

    mat = loadmat(str(ANNOT_PATH))
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print("keys:", keys)

    for key in keys:
        describe_obj(key, mat[key])

    inspect_calibration(CALIB_PATH)

if __name__ == "__main__":
    main()
