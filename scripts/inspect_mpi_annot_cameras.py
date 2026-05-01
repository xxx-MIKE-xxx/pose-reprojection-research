from pathlib import Path
import numpy as np
from scipy.io import loadmat

MAT_PATH = Path("data/raw/mpi_inf_3dhp/S1/Seq1/annot.mat")
mat = loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)

for key in ["annot2", "annot3", "univ_annot3"]:
    if key not in mat:
        continue

    arr = mat[key]
    print("\n" + "=" * 80)
    print(key)
    print("loaded shape:", getattr(arr, "shape", None), "dtype:", getattr(arr, "dtype", None))

    # MATLAB cell arrays often load as object arrays.
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        print("num cells/cameras:", arr.size)

        for cam_idx in range(min(arr.size, 14)):
            cam_data = arr.flat[cam_idx]
            print(f"camera/cell {cam_idx}: shape={cam_data.shape}, dtype={cam_data.dtype}")

            if cam_data.ndim == 2:
                frames = cam_data.shape[0]
                dims = cam_data.shape[1]
                print(f"  frames={frames}, flattened_dims={dims}")

                if key == "annot2":
                    print(f"  likely joints={dims // 2}, coords_per_joint=2")
                    sample = cam_data[0].reshape(-1, 2)
                else:
                    print(f"  likely joints={dims // 3}, coords_per_joint=3")
                    sample = cam_data[0].reshape(-1, 3)

                print("  first frame first 5 joints:")
                print(sample[:5])
    else:
        print("not object array")
        print(arr)
