import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, num_joints=17, hidden_dims=(512, 512, 256), dropout=0.1, zero_init_last=True):
        super().__init__()
        layers = []
        prev = int(input_dim)

        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.LayerNorm(int(h)))
            layers.append(nn.GELU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)

        out_dim = int(num_joints) * 3
        last = nn.Linear(prev, out_dim)
        if zero_init_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)

        self.net = nn.Sequential(*layers)
        self.num_joints = int(num_joints)

    def forward(self, features):
        # features: (B,T,F)
        b, t, f = features.shape
        flat = features.reshape(b * t, f)
        dx = self.net(flat).reshape(b, t, self.num_joints, 3)
        return dx


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
    x_geo = torch.from_numpy(sample["x_geo"][:1]) if "x_geo" in sample else None
    feat = build_features(y, u, m, z, input_cfg, ray_features=ray, x_geo_features=x_geo)
    return int(feat.shape[-1])
