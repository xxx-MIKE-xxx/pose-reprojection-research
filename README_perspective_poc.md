# Perspective-aware residual correction POC

This POC tests whether perspective-induced errors from a frozen 2D-to-3D lifter are structured and learnable by a residual corrector.

The implemented v1 method is:

```text
X_gt  = ground-truth 3D skeleton
z     = sampled virtual camera parameters
u_z   = Project(X_gt, z)
u_in  = NormalizeForLifter(u_z)
Y     = FrozenLifter(u_in)
dX    = Pc(Y, u_in, raw_2D_metadata, z)
X_hat = Y + dX
loss  = L(X_hat, X_gt)
