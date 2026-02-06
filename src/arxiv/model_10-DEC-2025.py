# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell (gated) operating on (B, C, H, W) tensors.
    Equations:
      i,f,o,g = conv([x, h])
      c' = f * c + i * g
      h' = o * tanh(c')
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size,
                              padding=padding)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, h, c):
        # x: (B, in_ch, H, W), h & c: (B, hidden_ch, H, W)
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        ci, cf, co, cg = torch.chunk(conv_out, 4, dim=1)
        i = torch.sigmoid(ci)
        f = torch.sigmoid(cf)
        o = torch.sigmoid(co)
        g = torch.tanh(cg)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class SmallEncoder(nn.Module):
    """Simple 2-block encoder to extract spatial features per time step."""
    def __init__(self, in_ch, hidden_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.block(x)  # (B, hidden_dim, H, W)


class ImprovedDecoder(nn.Module):
    """
    Multi-conv decoder that maps hidden feature -> single channel output.
    Upsampling is done with conv transpose then convs.
    """
    def __init__(self, hidden_dim, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.up = nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size, padding=pad)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.conv_block:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if isinstance(self.up, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(self.up.weight, nonlinearity="relu")
            if self.up.bias is not None:
                nn.init.zeros_(self.up.bias)

    def forward(self, x, output_size=None):
        # x: (B, hidden_dim, H_enc, W_enc)
        up = self.up(x)  # (B, hidden_dim, H_dec, W_dec) -> doubled
        out = self.conv_block(up)
        if output_size is not None:
            out = F.interpolate(out, size=output_size, mode="bilinear", align_corners=False)
        return out  # (B, 1, H_out, W_out)


class Conv2ConvLSTM(nn.Module):
    """
    Conv encoder per time step + ConvLSTM cell + improved decoder.
    Accepts input: (B, T, C, H, W) -> outputs (B, 1, H, W)
    """
    def __init__(self,
                 input_channels: int,
                 hidden_dim: int = 64,
                 kernel_size: int = 3,
                 H: int = None,
                 W: int = None):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.H = H
        self.W = W

        # encoder maps input channels -> hidden_dim feature maps
        self.encoder = SmallEncoder(in_ch=input_channels, hidden_dim=hidden_dim, kernel_size=kernel_size)

        # ConvLSTMCell expects (in_channels=hidden_dim, hidden_channels=hidden_dim)
        self.cell = ConvLSTMCell(in_channels=hidden_dim, hidden_channels=hidden_dim, kernel_size=kernel_size)

        # decoder maps final hidden state -> output map
        self.decoder = ImprovedDecoder(hidden_dim=hidden_dim, kernel_size=kernel_size)

        # Note: if we want to support H/W inference when encoded size is half,
        # we will adapt with interpolation at the end.

        self.reset_parameters()

    def reset_parameters(self):
        # already reset in submodules. keep for completeness
        pass

    def forward(self, x, debug: bool = False):
        """
        x: (B, T, C, H, W)
        returns out: (B, 1, H, W)
        """
        if debug:
            print("[Conv2ConvLSTM] input shape:", x.shape)

        B, T, C, H, W = x.shape

        # encode each timestep -> (B, hidden_dim, H, W)
        encoded_seq = []
        for t in range(T):
            xt = x[:, t]  # (B, C, H, W)
            feats = self.encoder(xt)  # (B, hidden_dim, H, W)
            encoded_seq.append(feats)

        # initialize hidden & cell to zeros on same device/dtype
        device = x.device
        h = torch.zeros(B, self.hidden_dim, encoded_seq[0].shape[-2], encoded_seq[0].shape[-1], device=device, dtype=x.dtype)
        c = torch.zeros_like(h)

        # run ConvLSTMCell over time
        for t, feats in enumerate(encoded_seq):
            h, c = self.cell(feats, h, c)

        # h is final hidden (B, hidden_dim, H_enc, W_enc)
        # decode -> produce (B, 1, H_dec, W_dec)
        out = self.decoder(h, output_size=(H, W))

        # safety clamp
        out = torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

        if debug:
            print("[Conv2ConvLSTM] out shape:", out.shape)

        return out


class OceanPredictor(nn.Module):
    """
    Wrapper that keeps same interface. Creates Conv2ConvLSTM backbone.
    """
    def __init__(self, cfg=None, input_channels=None, H=None, W=None):
        super().__init__()

        # If cfg provided, compute input_channels automatically
        if cfg is not None and input_channels is None:
            in_ch = 0
            # velocity paths count
            if hasattr(cfg.data, "neighbors") and hasattr(cfg.data.neighbors, "velocity_paths"):
                in_ch += len(cfg.data.neighbors.velocity_paths)
            # tracers variables count
            if hasattr(cfg.data, "tracers") and hasattr(cfg.data.tracers, "variables"):
                in_ch += len(cfg.data.tracers.variables)
            input_channels = in_ch

        if input_channels is None:
            raise ValueError("input_channels must be provided either via cfg or directly.")

        hidden_dim = getattr(cfg.model, "hidden_dim", 64) if cfg is not None else 64
        kernel_size = getattr(cfg.model, "kernel_size", 3) if cfg is not None else 3

        print(f"[OceanPredictor] Using dynamic H={H}, W={W}")
        self.model = Conv2ConvLSTM(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            H=H, W=W
        )

    def forward(self, x, debug: bool = False):
        return self.model(x, debug=debug)

