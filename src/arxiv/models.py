from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        # x: [B, C, H, W], h_prev: [B, Hdim, H, W]
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for i, in_d in enumerate(dims):
            self.layers.append(ConvLSTMCell(in_d, hidden_dim, kernel_size))
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h = [None] * len(self.layers)
        c = [None] * len(self.layers)
        outputs = []
        for t in range(T):
            inp = x[:, t]
            for l, cell in enumerate(self.layers):
                if h[l] is None:
                    h[l] = torch.zeros(B, cell.hidden_dim, H, W, device=x.device)
                    c[l] = torch.zeros(B, cell.hidden_dim, H, W, device=x.device)
                h[l], c[l] = cell(inp, h[l], c[l])
                inp = self.dropout(h[l])
            outputs.append(h[-1])
        return outputs[-1]  # [B, hidden, H, W]

class CNNEncoder2D(nn.Module):
    """Treat depth as channels, run 2D convs on (y,x)."""
    def __init__(self, in_channels: int, enc_channels: List[int], k: int = 3, dropout: float = 0.0):
        super().__init__()
        layers = []
        c = in_channels
        for ch in enc_channels:
            layers += [
                nn.Conv2d(c, ch, k, padding=k//2),
                nn.GroupNorm(4, ch),
                nn.SiLU(),
            ]
            c = ch
        self.net = nn.Sequential(*layers)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        return self.drop(self.net(x))

class Head1x1(nn.Module):
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.proj(x)

class CNNLSTMModel(nn.Module):
    """
    Spatial encoder (2D CNN) applied frame-wise with depth folded into channels.
    Temporal core: ConvLSTM over spatial feature maps.
    """
    def __init__(self, in_channels: int, enc_channels=[64, 128], hidden=64, k=3, convlstm_layers=1, dropout=0.1):
        super().__init__()
        # The input to encoder will have channels = in_channels * depth
        # but we fold depth into channels *before* passing to encoder outside this class.
        self.encoder = CNNEncoder2D(in_channels, enc_channels, k, dropout)
        self.convlstm = ConvLSTM(enc_channels[-1], hidden, k, num_layers=convlstm_layers, dropout=dropout)
        self.head = Head1x1(hidden, out_ch=1)

    def forward(self, x):
        # x: [B, T, C, Z, Y, X]
        B, T, C, Z, H, W = x.shape
        # Fold depth into channels
        x = x.permute(0,1,3,2,4,5).reshape(B, T, C*Z, H, W)  # [B, T, C*Z, H, W]
        # Encode each frame
        feats = []
        for t in range(T):
            feats.append(self.encoder(x[:, t]))  # [B, F, H, W]
        feats = torch.stack(feats, dim=1)  # [B, T, F, H, W]
        hT = self.convlstm(feats)          # [B, Hdim, H, W]
        out = self.head(hT)                # [B, 1, H, W]
        # Predict per (y,x); if Z>1 we repeat the plane along Z to match target shape.
        out = out[:, :, None, :, :]  # [B, 1, 1, H, W]
        return out

def build_model(cfg, in_channels: int):
    mcfg = cfg["model"]
    typ = mcfg.get("type", "cnnlstm")
    spatial_mode = mcfg.get("spatial_mode", "2d")
    if typ == "cnnlstm":
        return CNNLSTMModel(
            in_channels=in_channels,
            enc_channels=mcfg.get("enc_channels", [64,128]),
            hidden=mcfg.get("hidden_channels", 64),
            k=mcfg.get("kernel_size", 3),
            convlstm_layers=mcfg.get("convlstm_layers", 1),
            dropout=mcfg.get("dropout", 0.1),
        )
    else:
        raise NotImplementedError(f"Model type {typ} not implemented in this template.")
