from pathlib import Path
import torch

ckpt_path = Path("checkpoints/videopose3d/pretrained_h36m_cpn.bin")

if not ckpt_path.exists():
    raise FileNotFoundError(ckpt_path)

ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

print("checkpoint:", ckpt_path)
print("type:", type(ckpt))

if isinstance(ckpt, dict):
    print("top-level keys:", list(ckpt.keys()))
    if "model_pos" in ckpt:
        print("model_pos params:", len(ckpt["model_pos"]))
        first_key = next(iter(ckpt["model_pos"]))
        print("first model_pos key:", first_key)
        print("first tensor shape:", tuple(ckpt["model_pos"][first_key].shape))
