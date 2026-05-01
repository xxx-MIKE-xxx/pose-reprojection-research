from pathlib import Path
import numpy as np
from scipy.io import loadmat

mat_path = Path("data/raw/mpi_inf_3dhp/S1/Seq1/annot.mat")
out_path = Path("data/processed/mpi_s1_seq1_gt3d_debug.npz")
out_path.parent.mkdir(parents=True, exist_ok=True)

mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

arr = mat["univ_annot3"]
cam_data = arr.flat[0] if isinstance(arr, np.ndarray) and arr.dtype == object else arr

gt_3d = cam_data[:243].reshape(243, -1, 3)

np.savez_compressed(
    out_path,
    gt_3d=gt_3d,
    frame_indices=np.arange(243),
    source=np.array("MPI-INF-3DHP S1 Seq1 univ_annot3 cell0 first243")
)

print("saved:", out_path)
print("gt_3d shape:", gt_3d.shape)
print("first frame first 5 joints:")
print(gt_3d[0, :5])
