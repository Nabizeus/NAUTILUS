"""
model_flower.py -- Flower (Muser et al. 2026, arXiv:2603.04430) as NAUTILUS models (NAUTIFLOWER).

    cfg.model.arch   model
    "flower2d"       whole column as channels, as E: all levels, variables and input times
                     as channels of one 2-D Flower U-Net, one output channel per level;
                     the warp moves values horizontally
    "flower3d"       3-D Flower U-Net over (Z, H, W): variables and input times as channels,
                     one output channel; the warp moves values horizontally and vertically

Flower blocks warp their features, u(x) -> u(x + delta(x)), with a displacement
delta predicted from the features themselves (a learned semi-Lagrangian step).
src/flower_standalone.py is the Flower code from ~/Codex/flowers, unchanged.
Three things differ from it here, so the models fit the NAUTILUS training and
evaluation (48x48 patches, 78x78 boxes, 24x24 random boxes, global tiles):

- any grid size: inputs are padded with zeros (as land) to a multiple of
  2^(n_levels-1) and the output is cropped back; warp grids are made per size;
- the displacement is in grid cells (times model.flow_scale), not in units of
  the domain, so the learned flow means the same distance on every grid size;
- no coordinate channels by default (model.coords), as the other NAUTILUS
  models: positions inside a patch mean nothing.

Ocean mask (model.mask_warp, needs data.mask_channel: true): the mask channel
is taken off the input (the network sees the same 6 variables as the other
models) and used in every warp. Values are set to 0 outside the ocean, sampled
with the ocean weight of the same interpolation and divided by it, so a warped
ocean cell is an average of ocean values only, and warped features are 0
outside the ocean. Coarser U-Net levels use the mask max-pooled (a coarse cell
is ocean if any of its cells is). flower3d uses the 3-D mask (land and sea
floor); flower2d, whose features mix all levels, the surface ocean mask, with
the sea floor masked level by level in the output.

Like all NAUTILUS models: (B, T, C, Z, H, W) in, (B, 1, Z, H, W) out.
"""

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src import flower_standalone as fs


def _plain(fn):
    """The function under @torch.compile (compiling fails with icpc on MN5)."""
    return getattr(fn, "_torchdynamo_orig_callable", fn)


@lru_cache(maxsize=64)
def _grid(sizes, device, dtype):
    """Coordinate grid in [-1, 1] for spatial sizes (D, H, W) or (H, W); last axis (x, y[, z])."""
    return fs.create_grid(list(sizes)).to(device=device, dtype=dtype)


class CellWarp(fs.SelfWarp):
    """fs.SelfWarp for any grid size, with the displacement in grid cells."""

    flow_scale = 1.0
    masks = None                                # {spatial sizes: (B, 1, ...) ocean mask}, set by OceanFlower
    mask_min_weight = 0.1                       # floor of the ocean share in a masked warp (model.mask_min_weight)
    grid_samples = {n: _plain(f) for n, f in fs.grid_samples.items()}

    def forward(self, u, meta=None):
        n = self.n_spatial_dims
        sizes = tuple(u.shape[2:])
        flow = rearrange(self.flow_head(u), fs.flow_fold_heads[n], dir=n, heads=self.num_heads)
        value = rearrange(self.value_head(u), fs.value_fold_heads[n], heads=self.num_heads)
        # one cell along x, y (, z) in the [-1, 1] coordinates of grid_sample
        cell = torch.tensor([2.0 / max(s - 1, 1) for s in reversed(sizes)], device=u.device, dtype=flow.dtype)
        grid = _grid(sizes, u.device, flow.dtype).unsqueeze(0) + flow * (self.flow_scale * cell)
        sample = lambda a, g: self.grid_samples[n](a, g, mode="bilinear",
                                                   padding_modes=self.padding_modes, align_corners=True)
        m = self.masks.get(sizes) if self.masks else None
        if m is None:
            u_warp = sample(value, grid)
        else:
            # float32 outside autocast: the division by the ocean share overflows in fp16. The share is
            # floored at mask_min_weight, so warped values fade to 0 where the sample point hardly
            # touches ocean, and the gradient of a / w (-a / w**2) stays bounded.
            with torch.autocast(device_type=u.device.type, enabled=False):
                g = grid.float()
                m = m.float().repeat_interleave(self.num_heads, dim=0)       # (B heads) 1 ...
                weight = sample(m, g)                                         # ocean share of the interpolation
                u_warp = sample(value.float() * m, g) / weight.clamp_min(self.mask_min_weight) * m
            u_warp = u_warp.to(value.dtype)
        return rearrange(u_warp, fs.value_unfold_heads[n], heads=self.num_heads)


class OceanFlower(nn.Module):
    """Flower U-Net on the NAUTILUS interface; n_dims 2 (column as channels) or 3."""

    n_dims = 2

    def __init__(self, cfg, input_channels):
        super().__init__()
        m = cfg.model
        T = int(cfg.data.window.input_steps)
        Z = int(cfg.data.vertical_levels)
        n = self.n_dims
        n_levels = int(m.get("n_levels", 3))
        self.div = 2 ** (n_levels - 1)
        self.n_pools = n_levels - 1
        self.coords = bool(m.get("coords", False))
        self.mask_warp = bool(m.get("mask_warp", False))
        if self.mask_warp:
            if not bool(cfg.data.get("mask_channel", True)):
                raise ValueError("model.mask_warp needs data.mask_channel: true (the mask is the last input channel)")
            input_channels -= 1                    # the mask is used in the warps, not as an input
        dim_in = T * input_channels * (Z if n == 2 else 1)
        dim_out = Z if n == 2 else 1

        self.flower = fs.Flower(
            dim_in=dim_in, dim_out=dim_out, n_spatial_dims=n,
            spatial_resolution=[self.div] * n,             # only sizes the unused grid buffers
            lifting_dim=int(m.get("lifting_dim", 128 if n == 2 else 48)),
            n_levels=n_levels,
            num_heads=int(m.get("num_heads", 32 if n == 2 else 16)),
            boundary_condition_types=[],                   # zeros outside: tiles and boxes are not periodic
            groups=int(m.get("groups", 32 if n == 2 else 16)),
            dropout_rate=float(m.get("flower_dropout", 0.0)),
        )
        f = self.flower
        del f.coord_grid
        if not self.coords:
            f.lift = fs.conv_modules[n](dim_in, f.lifting_dim, kernel_size=1)
        flow_scale = float(m.get("flow_scale", 1.0))
        self.masks = {}                            # shared with every warp, filled per forward call
        for mod in f.modules():
            if isinstance(mod, fs.SelfWarp):
                mod.__class__ = CellWarp
                mod.flow_scale = flow_scale
                mod.masks = self.masks
                mod.mask_min_weight = float(m.get("mask_min_weight", 0.1))
                del mod.base_grid

    def _unet(self, x):
        f = self.flower
        if self.coords:
            grid = _grid(tuple(x.shape[2:]), x.device, x.dtype)
            grid = grid.movedim(-1, 0).unsqueeze(0).expand(x.shape[0], *([-1] * grid.dim()))
            x = torch.cat((x, grid), dim=1)
        x = f.lift(x)
        skips = []
        for block in f.encoder_blocks:
            skip, x = block(x)
            skips.append(skip)
        x = f.bottleneck(x)
        for block, skip in zip(f.decoder_blocks, reversed(skips)):
            x = torch.cat([block(x), skip], dim=1)
        return f.project(f.dropout(x))

    def _set_masks(self, mask):
        """Ocean mask (B, 1, ...) at the padded size and every coarser U-Net level."""
        pool = F.max_pool2d if self.n_dims == 2 else F.max_pool3d
        self.masks.clear()
        for _ in range(self.n_pools + 1):
            self.masks[tuple(mask.shape[2:])] = mask
            mask = pool(mask, kernel_size=2)

    def forward(self, x):                       # (B, T, C, Z, H, W) -> (B, 1, Z, H, W)
        if self.mask_warp:
            ocean = x[:, -1, -1:]               # (B, 1, Z, H, W), the mask channel
            x = x[:, :, :-1]
        B, T, C, Z, H, W = x.shape
        if self.n_dims == 2:
            x = x.reshape(B, T * C * Z, H, W)
        else:
            x = x.reshape(B, T * C, Z, H, W)
        sizes = x.shape[2:]
        pad = []
        for s in reversed(sizes):               # F.pad starts from the last axis
            pad += [0, (-s) % self.div]
        if self.mask_warp:
            warp_mask = ocean.amax(dim=2) if self.n_dims == 2 else ocean    # 2-D: surface ocean
            self._set_masks(F.pad(warp_mask.float(), pad))
        try:
            out = self._unet(F.pad(x, pad))
        finally:
            self.masks.clear()
        out = out[(slice(None), slice(None)) + tuple(slice(0, s) for s in sizes)]
        if self.n_dims == 2:
            out = out.unsqueeze(1)
        if self.mask_warp:
            out = out * ocean.to(out.dtype)     # 0 below the sea floor and on land
        return out


class OceanFlower2D(OceanFlower):
    n_dims = 2


class OceanFlower3D(OceanFlower):
    n_dims = 3
