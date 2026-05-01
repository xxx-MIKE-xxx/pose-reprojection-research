import numpy as np

H36M17_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
]

H36M17_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# MPI-INF-3DHP raw 28-joint annotation -> MPI17 subset.
RAW28_TO_MPI17 = [7, 5, 14, 15, 16, 9, 10, 11, 23, 24, 25, 18, 19, 20, 4, 3, 6]

MPI17_NAMES = [
    "head_top", "neck", "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "root", "spine", "head",
]

MPI17_TO_H36M17 = [
    14,
    8, 9, 10,
    11, 12, 13,
    15,
    1,
    1,
    16,
    5, 6, 7,
    2, 3, 4,
]


def root_center(x, root=0):
    x = np.asarray(x, dtype=np.float32)
    return x - x[..., root:root + 1, :]


def reduce_to_mpi17(joints):
    if joints.shape[1] == 17:
        return joints
    if joints.shape[1] >= 28:
        return joints[:, RAW28_TO_MPI17]
    raise ValueError(f"Expected 17 or >=28 joints, got {joints.shape[1]}")


def mpi17_to_h36m17(joints):
    return joints[:, MPI17_TO_H36M17]


def bone_lengths(x, bones=H36M17_BONES):
    x = np.asarray(x)
    vals = []
    for a, b in bones:
        vals.append(np.linalg.norm(x[..., a, :] - x[..., b, :], axis=-1))
    return np.stack(vals, axis=-1)
