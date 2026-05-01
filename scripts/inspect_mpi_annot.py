from pathlib import Path
import numpy as np
from scipy.io import loadmat

MAT_PATH = Path("data/raw/mpi_inf_3dhp/S1/Seq1/annot.mat")

mat = loadmat(MAT_PATH, squeeze_me=False, struct_as_record=False)

print("File:", MAT_PATH)
print("Top-level keys:")
for k, v in mat.items():
    if k.startswith("__"):
        continue
    print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

print("\nDetailed contents:")
for k, v in mat.items():
    if k.startswith("__"):
        continue

    print("\n" + "=" * 80)
    print(k)
    print("type:", type(v))
    print("shape:", getattr(v, "shape", None))
    print("dtype:", getattr(v, "dtype", None))

    if isinstance(v, np.ndarray):
        print("ndim:", v.ndim)
        print("size:", v.size)

        if v.dtype == object:
            print("object/cell array contents:")
            flat = v.ravel()
            for i, item in enumerate(flat[:20]):
                print(f"  cell {i}: type={type(item)}, shape={getattr(item, 'shape', None)}, dtype={getattr(item, 'dtype', None)}")
        else:
            print("min/max:", np.nanmin(v), np.nanmax(v))
            print("first values:", v.ravel()[:10])
