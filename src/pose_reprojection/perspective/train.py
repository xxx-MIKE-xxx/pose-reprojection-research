from pathlib import Path
import csv
import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .model import ResidualMLP, build_features, infer_input_dim
from .losses import compute_losses
from .features import corrector_y_input, corrector_pose_input, compose_prediction
from .reproducibility import seed_everything


class CameraSequenceDataset(Dataset):
    def __init__(self, arrays, indices):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        keys = ["x_gt", "u_px", "u_norm", "raw_2d_metadata", "z", "z_features", "y_lifted"]
        for optional in ["ray_features", "x_geo"]:
            if optional in self.arrays:
                keys.append(optional)
        return {k: self.arrays[k][idx].astype(np.float32) for k in keys}


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _to_device(batch, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in batch.items()}


def evaluate_epoch(model, loader, config, device):
    model.eval()
    sums = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            features = build_features(
                corrector_y_input(batch["y_lifted"], config),
                batch["u_norm"],
                batch["raw_2d_metadata"],
                batch["z_features"],
                config["corrector_inputs"],
                ray_features=batch.get("ray_features"),
                x_geo_features=corrector_pose_input(batch["x_geo"], config) if "x_geo" in batch else None,
            )
            model_output = model(features)
            x_hat = compose_prediction(batch["y_lifted"], model_output, config, x_geo=batch.get("x_geo"))
            losses = compute_losses(x_hat, batch["x_gt"], batch["u_px"], batch["z"], config["losses"])
            bs = batch["x_gt"].shape[0]
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach().cpu()) * bs
            count += bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def train_corrector(arrays, config, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.get("seed", 1234), deterministic=config.get("runtime", {}).get("deterministic", True))

    train_idx = arrays["train_indices"]
    val_idx = arrays["val_indices"]

    train_ds = CameraSequenceDataset(arrays, train_idx)
    val_ds = CameraSequenceDataset(arrays, val_idx)

    device = _device(config["training"].get("device", "auto"))

    input_dim = infer_input_dim({k: arrays[k] for k in arrays}, config["corrector_inputs"])
    model = ResidualMLP(
        input_dim=input_dim,
        num_joints=17,
        hidden_dims=config["model"]["hidden_dims"],
        dropout=float(config["model"].get("dropout", 0.1)),
        zero_init_last=bool(config["model"].get("zero_init_last", True)),
        output_cfg=config.get("corrector_output", {}),
    ).to(device)

    train_generator = torch.Generator()
    train_generator.manual_seed(int(config.get("seed", 1234)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []

    epochs = int(config["training"]["epochs"])
    patience = int(config["training"].get("patience", epochs))
    grad_clip = config["training"].get("grad_clip_norm", None)

    for epoch in range(1, epochs + 1):
        model.train()
        train_sums = {}
        train_count = 0

        for batch in train_loader:
            batch = _to_device(batch, device)
            features = build_features(
                corrector_y_input(batch["y_lifted"], config),
                batch["u_norm"],
                batch["raw_2d_metadata"],
                batch["z_features"],
                config["corrector_inputs"],
                ray_features=batch.get("ray_features"),
                x_geo_features=corrector_pose_input(batch["x_geo"], config) if "x_geo" in batch else None,
            )
            model_output = model(features)
            x_hat = compose_prediction(batch["y_lifted"], model_output, config, x_geo=batch.get("x_geo"))

            losses = compute_losses(x_hat, batch["x_gt"], batch["u_px"], batch["z"], config["losses"])
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            bs = batch["x_gt"].shape[0]
            for k, v in losses.items():
                train_sums[k] = train_sums.get(k, 0.0) + float(v.detach().cpu()) * bs
            train_count += bs

        train_metrics = {f"train_{k}": v / max(train_count, 1) for k, v in train_sums.items()}
        val_metrics_raw = evaluate_epoch(model, val_loader, config, device)
        val_metrics = {f"val_{k}": v for k, v in val_metrics_raw.items()}
        row = {"epoch": epoch, **train_metrics, **val_metrics}
        history.append(row)

        val_total = val_metrics_raw.get("total", float("inf"))
        print(
            f"[epoch {epoch:03d}] "
            f"train_total={train_metrics.get('train_total', 0.0):.6f} "
            f"val_total={val_total:.6f}"
        )

        if val_total < best_val:
            best_val = val_total
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            torch.save(
                {
                    "model_state": best_state,
                    "input_dim": input_dim,
                    "config": config,
                    "best_val_total": best_val,
                    "epoch": epoch,
                },
                out_dir / "best_corrector.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[early stop] no validation improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_path = out_dir / "training_history.csv"
    if history:
        with hist_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({k for row in history for k in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    return model, history
