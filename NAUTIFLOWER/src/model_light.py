"""
model_light.py -- lighter variants of the full-column model (v6) and a
factory that builds the model named in cfg.model.arch.

    cfg.model.arch           model                                     compute per call (vs conv2dlstm, 128 ch)
    "conv2dlstm" (default)   OceanConv2DAsymXPU: per level, +/-2-level  1x  (A: hidden_dim 64 -> about 4x less)
                             window, ConvLSTM over the input times
    "conv2dlstm_sep"         as conv2dlstm with depthwise-separable     about 9x less at 128 channels
                             3x3 convolutions (B)
    "timestack"              per level, +/-2-level window, the input     about 40x less at 64 channels
                             times stacked as channels, residual conv
                             blocks, no recurrence (C)
    "column"                 whole column as channels: all levels,      about 100x less at 256 channels
                             variables and input times in one 2-D
                             network, one output channel per level (E)
    "flower2d", "flower3d"   Flower U-Nets (NAUTIFLOWER), see src/model_flower.py

All models take (B, T, C, Z, H, W) and return (B, 1, Z, H, W), so training,
evaluation, prediction and roll-out work with any of them.
"""

import torch
import torch.nn as nn

from src.model_conv2d_asym_hybrid import OceanConv2DAsymXPU
from src.model_flower import OceanFlower2D, OceanFlower3D


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------
class SepConv2d(nn.Module):
    """Depthwise 3x3 (per channel) followed by pointwise 1x1."""

    def __init__(self, cin, cout, kernel_size=3):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, kernel_size, padding=kernel_size // 2, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


def conv3(cin, cout, separable=False):
    return SepConv2d(cin, cout) if separable else nn.Conv2d(cin, cout, 3, padding=1)


class ConvLSTMCellLight(nn.Module):
    def __init__(self, dim, separable=False):
        super().__init__()
        self.hidden_dim = dim
        self.conv = conv3(2 * dim, 4 * dim, separable)

    def forward(self, x, h, c):
        i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        return torch.sigmoid(o) * torch.tanh(c), c


class Conv2LSTMLight(nn.Module):
    """Same structure as src.model.Conv2LSTM (encoder, ConvLSTM, decoder), convolutions selectable."""

    def __init__(self, input_channels, hidden_dim=64, separable=False, dropout=0.25):
        super().__init__()
        h = hidden_dim
        self.encoder = nn.Sequential(
            conv3(input_channels, h, separable), nn.BatchNorm2d(h), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            conv3(h, h, separable), nn.BatchNorm2d(h), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
        )
        self.cell = ConvLSTMCellLight(h, separable)
        self.decoder = nn.Sequential(
            conv3(h, h // 2, separable), nn.BatchNorm2d(h // 2), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(h // 2, 1, 1),
        )

    def forward(self, x):                       # (B, T, C, H, W) -> (B, 1, H, W)
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for t in range(T):
            f = self.encoder(x[:, t])
            h, c = self.cell(f, h, c)
        return self.decoder(h + f)


class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True), nn.Dropout2d(dropout),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.body(x))


class TimeStackNet(nn.Module):
    """C: the input times stacked as channels, residual conv blocks, no recurrence."""

    def __init__(self, T, reduce_ch, hidden_dim=64, n_blocks=2, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(T * reduce_ch, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(hidden_dim // 2, 1, 1))

    def forward(self, x):                       # (B, T, C, H, W) -> (B, 1, H, W)
        B, T, C, H, W = x.shape
        return self.head(self.blocks(self.stem(x.reshape(B, T * C, H, W))))


# ----------------------------------------------------------------------
# B and C: same depth-window wrapper as the reference model
# ----------------------------------------------------------------------
class OceanConv2DLSTMSep(OceanConv2DAsymXPU):
    """B: the reference model with depthwise-separable 3x3 convolutions."""

    def __init__(self, cfg, input_channels):
        super().__init__(cfg, input_channels)
        self.model2d = Conv2LSTMLight(self.hidden_dim, self.hidden_dim, separable=True)


class OceanTimeStack(OceanConv2DAsymXPU):
    """C: per level, +/-2-level window; the input times as channels instead of a ConvLSTM."""

    def __init__(self, cfg, input_channels):
        super().__init__(cfg, input_channels)
        reduce_ch = int(cfg.model.get("reduce_channels", 16))
        T = int(cfg.data.window.input_steps)
        self.input_reduce = nn.Conv2d(self.channels_per_depth_stack * self.max_window, reduce_ch, 1)
        self.model2d = TimeStackNet(T, reduce_ch, self.hidden_dim, int(cfg.model.get("n_blocks", 2)))


# ----------------------------------------------------------------------
# E: whole column as channels
# ----------------------------------------------------------------------
class OceanColumnNet(nn.Module):
    """E: all levels, variables and input times as channels of one 2-D network."""

    def __init__(self, cfg, input_channels):
        super().__init__()
        T = int(cfg.data.window.input_steps)
        Z = int(cfg.data.vertical_levels)
        w = int(cfg.model.get("hidden_dim", 256))
        self.Z = Z
        self.stem = nn.Sequential(nn.Conv2d(T * input_channels * Z, w, 1), nn.BatchNorm2d(w), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[ResBlock(w, 0.1) for _ in range(int(cfg.model.get("n_blocks", 3)))])
        self.head = nn.Conv2d(w, Z, 1)

    def forward(self, x):                       # (B, T, C, Z, H, W) -> (B, 1, Z, H, W)
        B, T, C, Z, H, W = x.shape
        out = self.head(self.blocks(self.stem(x.reshape(B, T * C * Z, H, W))))
        return out.unsqueeze(1)


ARCHS = {"conv2dlstm": OceanConv2DAsymXPU, "conv2dlstm_sep": OceanConv2DLSTMSep,
         "timestack": OceanTimeStack, "column": OceanColumnNet,
         "flower2d": OceanFlower2D, "flower3d": OceanFlower3D}


def build_model(cfg, input_channels):
    """The model named in cfg.model.arch (default: the reference conv2dlstm)."""
    arch = str(cfg.model.get("arch", "conv2dlstm") or "conv2dlstm")
    if arch not in ARCHS:
        raise ValueError(f"model.arch must be one of {list(ARCHS)}, not {arch!r}")
    return ARCHS[arch](cfg, input_channels=input_channels)
