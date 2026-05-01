import os
import random

import numpy as np


def seed_everything(seed, deterministic=True):
    """Seed common RNGs used by the perspective POC."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    info = {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_seeded": False,
        "torch_cuda_seeded": False,
        "deterministic_requested": bool(deterministic),
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "cublas_workspace_config": None,
    }

    try:
        import torch

        torch.manual_seed(seed)
        info["torch_seeded"] = True

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            info["torch_cuda_seeded"] = True

        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            info["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            info["cudnn_deterministic"] = True
            info["cudnn_benchmark"] = False

            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                info["torch_deterministic_algorithms"] = True
            except TypeError:
                torch.use_deterministic_algorithms(True)
                info["torch_deterministic_algorithms"] = True
    except Exception as exc:
        info["torch_error"] = str(exc)

    return info
