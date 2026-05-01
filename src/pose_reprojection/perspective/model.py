import torch
from torch import nn
import math


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_joints=17,
        hidden_dims=(512, 512, 256),
        dropout=0.1,
        zero_init_last=True,
        output_cfg=None,
    ):
        super().__init__()
        layers = []
        input_dim = int(input_dim)
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.LayerNorm(int(h)))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)

        out_dim = int(num_joints) * 3
        self.trunk = nn.Sequential(*layers)
        self.residual_head = nn.Linear(prev, out_dim)
        if zero_init_last:
            nn.init.zeros_(self.residual_head.weight)
            nn.init.zeros_(self.residual_head.bias)

        self.num_joints = int(num_joints)
        output_cfg = output_cfg or {}
        self.output_base = output_cfg.get("base", "y_lifted")
        self.gate_mode = output_cfg.get("gate_mode", "joint_scalar")
        self.gate_head = None
        if self.output_base == "gated_y_xgeo":
            if self.gate_mode == "sequence_scalar":
                gate_dim = 1
            elif self.gate_mode == "joint_scalar":
                gate_dim = self.num_joints
            else:
                raise ValueError(f"Unknown corrector_output.gate_mode: {self.gate_mode}")

            gate_init = float(output_cfg.get("gate_init_y_weight", 0.8))
            gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
            gate_bias = math.log(gate_init / (1.0 - gate_init))
            self.gate_head = nn.Linear(input_dim, gate_dim)
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_bias)

    def forward(self, features):
        # features: (B,T,F)
        b, t, f = features.shape
        flat = features.reshape(b * t, f)
        hidden = self.trunk(flat)
        dx = self.residual_head(hidden).reshape(b, t, self.num_joints, 3)
        if self.gate_head is None:
            return dx

        gate_logits = self.gate_head(flat)
        if self.gate_mode == "sequence_scalar":
            gate = torch.sigmoid(gate_logits).reshape(b, t, 1, 1).expand(-1, -1, self.num_joints, -1)
        elif self.gate_mode == "joint_scalar":
            gate = torch.sigmoid(gate_logits).reshape(b, t, self.num_joints, 1)
        else:
            raise ValueError(f"Unknown corrector_output.gate_mode: {self.gate_mode}")
        return {"residual": dx, "gate": gate}


def build_features(y_lifted, u_norm, raw_meta, z_features, input_cfg, ray_features=None, x_geo_features=None):
    """Build per-frame corrector features.

    All tensors are torch tensors.
    y_lifted: (B,T,17,3)
    u_norm: (B,T,17,2)
    raw_meta: (B,T,M)
    z_features: (B,D), Pc-only camera features. True z is used separately for projection losses/metrics.
    ray_features: optional (B,T,R), Pc-only ray/PE features.
    x_geo_features: optional (B,T,17,3), Pc-only geometry-fit 3D features.
    returns: (B,T,F)
    """
    parts = []
    b, t = y_lifted.shape[:2]

    if input_cfg.get("use_lifter_3d", True):
        parts.append(y_lifted.reshape(b, t, -1))

    if input_cfg.get("use_lifter_2d_normalized", True):
        parts.append(u_norm.reshape(b, t, -1))

    if input_cfg.get("use_raw_2d_metadata", True):
        parts.append(raw_meta.reshape(b, t, -1))

    if input_cfg.get("use_camera_parameters", True):
        if z_features is None:
            raise ValueError("z_features are required when use_camera_parameters=true")
        parts.append(z_features[:, None, :].expand(-1, t, -1))

    if input_cfg.get("use_ray_features", False):
        if ray_features is None:
            raise ValueError("ray_features are required when use_ray_features=true")
        parts.append(ray_features.reshape(b, t, -1))

    if input_cfg.get("use_geometry_fit_3d", False):
        if x_geo_features is None:
            raise ValueError("x_geo_features are required when use_geometry_fit_3d=true")
        parts.append(x_geo_features.reshape(b, t, -1))

    if not parts:
        raise ValueError("At least one corrector input must be enabled.")

    return torch.cat(parts, dim=-1)


def infer_input_dim(sample, input_cfg):
    import torch
    y = torch.from_numpy(sample["y_lifted"][:1])
    u = torch.from_numpy(sample["u_norm"][:1])
    m = torch.from_numpy(sample["raw_2d_metadata"][:1])
    z_key = "z_features" if "z_features" in sample else "z"
    z = torch.from_numpy(sample[z_key][:1])
    ray = torch.from_numpy(sample["ray_features"][:1]) if "ray_features" in sample else None
    x_geo_key = "x_geo_used" if "x_geo_used" in sample else "x_geo"
    x_geo = torch.from_numpy(sample[x_geo_key][:1]) if x_geo_key in sample else None
    feat = build_features(y, u, m, z, input_cfg, ray_features=ray, x_geo_features=x_geo)
    return int(feat.shape[-1])
