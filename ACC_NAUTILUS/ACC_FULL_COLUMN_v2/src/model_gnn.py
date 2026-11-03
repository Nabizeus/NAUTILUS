"""
model_gnn.py -- graph neural network over the ocean columns (v7, G).

    cfg.model.arch   model
    "gnn"            nodes = ocean columns, each with its whole column (all levels,
                     variables and input times) as features, as in E; edges = the
                     8 horizontal neighbours (3 x 3 stencil), east-west periodic.
                     Land columns are not nodes, so no compute and no messages there.
                     Encoder MLP -> n_layers message-passing layers -> decoder MLP,
                     one output per level.
    "gnn_grid"       the same network, with the neighbours taken by shifting the
                     dense (y, x) grid instead of index lists (land masked).

Message-passing layer (MeshGraphNet-style, residual):
    a_ij = act(A h_i + B h_j + e_d(ij))          edge MLP, first layer
    m_i  = W sum_j a_ij                          its second layer, applied after the sum
    h_i  = norm(h_i + MLP([h_i, m_i]))
The edge MLP on [h_i, h_j, e_ij] is computed as A h_i + B h_j + e (the same as one
linear layer on the concatenation), and its second linear layer W after the sum over
neighbours (a linear map commutes with the sum). So all matrix products are per node;
per edge only a gather, an add and the activation remain.

Spatial reach: n_layers neighbours, so tiles need a halo of n_layers cells.

All models take (B, T, C, Z, H, W) and return (B, 1, Z, H, W). Land columns of the
output are 0. The ocean columns are those set with set_mask(mask (H, W)) (fixed, e.g.
once per subdomain in a coupled run), or, if no mask is set or its shape differs, those
whose inputs are not all 0, per sample and on every call (land is 0 in the data; the
training patches all have different coasts). The samples of a batch are one graph
without edges between them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# the 8 neighbours (dy, dx) of the 3 x 3 stencil
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def build_edges(mask, periodic_x=True):
    """Edges between the ocean columns of mask (B, H, W), within each sample:
    (flat indices of the ocean columns in (B, H, W), src, dst, direction), dst sorted."""
    B, H, W = mask.shape
    idx = torch.full((B, H, W), -1, dtype=torch.long, device=mask.device)
    n = int(mask.sum())
    idx[mask] = torch.arange(n, device=mask.device)
    bs, ys, xs = torch.nonzero(mask, as_tuple=True)
    src, dst, dirs = [], [], []
    for d, (dy, dx) in enumerate(OFFSETS):
        yn, xn = ys + dy, xs + dx
        if periodic_x:
            xn = xn % W
        ok = (yn >= 0) & (yn < H) & (xn >= 0) & (xn < W)
        j = idx[bs[ok], yn[ok], xn[ok]]
        i = idx[bs[ok], ys[ok], xs[ok]]
        src.append(j[j >= 0]); dst.append(i[j >= 0])
        dirs.append(torch.full((int((j >= 0).sum()),), d, device=mask.device))
    src, dst, dirs = torch.cat(src), torch.cat(dst), torch.cat(dirs)
    order = torch.argsort(dst, stable=True)
    flat = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    return flat, src[order], dst[order], dirs[order]


class MLP(nn.Sequential):
    def __init__(self, cin, hidden, cout, norm=True):
        layers = [nn.Linear(cin, hidden), nn.SiLU(), nn.Linear(hidden, cout)]
        if norm:
            layers.append(nn.LayerNorm(cout))
        super().__init__(*layers)


class MessagePassing(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.A = nn.Linear(h, h)
        self.B = nn.Linear(h, h, bias=False)
        self.e = nn.Parameter(torch.zeros(len(OFFSETS), h))   # one learned edge feature per direction
        self.W = nn.Linear(h, h, bias=False)
        self.node = MLP(2 * h, h, h, norm=False)
        self.norm = nn.LayerNorm(h)

    def aggregate(self, h, src, dst, dirs):           # h (N, h) -> sum over neighbours (N, h)
        p, q = self.A(h), self.B(h)
        a = F.silu(p.index_select(0, dst) + q.index_select(0, src) + self.e[dirs].to(p.dtype))
        return torch.zeros(h.shape, dtype=a.dtype, device=h.device).index_add_(0, dst, a)

    def aggregate_grid(self, h, ocean, periodic_x):  # h (B, h, H, W), ocean (B, 1, H, W) -> (B, h, H, W)
        p = F.conv2d(h, self.A.weight[:, :, None, None], self.A.bias)
        q = F.conv2d(h, self.B.weight[:, :, None, None])

        def pad(a):                                   # 1-cell frame, east-west periodic or 0
            a = torch.cat([a[..., -1:], a, a[..., :1]], dim=-1) if periodic_x else F.pad(a, (1, 1))
            return F.pad(a, (0, 0, 1, 1))

        q, on = pad(q), pad(ocean)
        H, W = h.shape[-2:]
        s = torch.zeros_like(p)
        for d, (dy, dx) in enumerate(OFFSETS):
            win = (..., slice(1 + dy, 1 + dy + H), slice(1 + dx, 1 + dx + W))
            # land neighbours send nothing
            s = s + F.silu(p + q[win] + self.e[d][None, :, None, None].to(p.dtype)) * on[win].to(p.dtype)
        return s

    def update(self, h, s):                           # (..., h) node features and neighbour sums
        m = self.W(s)
        return self.norm(h + self.node(torch.cat([h, m.to(h.dtype)], dim=-1)).to(h.dtype))


class OceanColumnGNN(nn.Module):
    """G: message passing over the ocean columns, whole column as node features."""

    grid = False

    def __init__(self, cfg, input_channels):
        super().__init__()
        T = int(cfg.data.window.input_steps)
        Z = int(cfg.data.vertical_levels)
        h = int(cfg.model.get("hidden_dim", 128))
        self.Z = Z
        self.periodic_x = bool(cfg.model.get("periodic_x", False))   # true only for the global domain
        self.encoder = MLP(T * input_channels * Z, h, h)
        self.layers = nn.ModuleList([MessagePassing(h) for _ in range(int(cfg.model.get("n_layers", 6)))])
        self.decoder = MLP(h, h, Z, norm=False)
        self._fixed = None                            # (mask (H, W), graph of one sample), from set_mask

    def set_mask(self, mask):
        """Fix the ocean columns (H, W bool), e.g. once per subdomain in a coupled run."""
        mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
        self._fixed = (mask, build_edges(mask[None], self.periodic_x))

    def graph(self, x):
        """Ocean mask (B, H, W) and the graph of the batch."""
        B, H, W = x.shape[0], x.shape[-2], x.shape[-1]
        if self._fixed is not None and tuple(self._fixed[0].shape) == (H, W):
            mask, g = self._fixed
            mask = mask.to(x.device)[None].expand(B, H, W)
            if B == 1:
                return mask, tuple(t.to(x.device) for t in g)
        else:
            mask = x.detach().abs().amax(dim=(1, 2, 3)) > 0         # per sample: land is 0 in every input
        return mask, build_edges(mask, self.periodic_x)

    def forward(self, x):                             # (B, T, C, Z, H, W) -> (B, 1, Z, H, W)
        B, T, C, Z, H, W = x.shape
        if self.grid:
            return self.forward_grid(x)
        mask, (flat, src, dst, dirs) = self.graph(x)
        cols = x.reshape(B, T * C * Z, H * W).transpose(1, 2).reshape(B * H * W, T * C * Z)
        h = self.encoder(cols.index_select(0, flat))                 # (N, h), N = ocean columns of the batch
        for layer in self.layers:
            h = layer.update(h, layer.aggregate(h, src, dst, dirs))
        y = self.decoder(h)
        out = y.new_zeros(B * H * W, Z).index_copy_(0, flat, y)
        return out.reshape(B, H, W, Z).permute(0, 3, 1, 2).unsqueeze(1)

    def forward_grid(self, x):
        B, T, C, Z, H, W = x.shape
        if self._fixed is not None and tuple(self._fixed[0].shape) == (H, W):
            mask = self._fixed[0].to(x.device)[None].expand(B, H, W)
        else:
            mask = x.detach().abs().amax(dim=(1, 2, 3)) > 0
        ocean = mask[:, None].to(x.dtype)                             # (B, 1, H, W)
        h = self.encoder(x.reshape(B, T * C * Z, H, W).permute(0, 2, 3, 1))  # (B, H, W, h)
        for layer in self.layers:
            s = layer.aggregate_grid(h.permute(0, 3, 1, 2), ocean, self.periodic_x)
            h = layer.update(h, s.permute(0, 2, 3, 1))
        out = self.decoder(h).permute(0, 3, 1, 2) * ocean.to(h.dtype)  # (B, Z, H, W)
        return out.unsqueeze(1)


class OceanColumnGNNGrid(OceanColumnGNN):
    grid = True
