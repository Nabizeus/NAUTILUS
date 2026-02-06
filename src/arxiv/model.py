import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------
# ConvLSTM Cell (stable version)
# -------------------------------------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, (h, c)

    def init_hidden(self, batch, H, W):
        return (
            torch.zeros(batch, self.hidden_dim, H, W),
            torch.zeros(batch, self.hidden_dim, H, W)
        )


# -------------------------------------------------------
# The FIXED Model
# -------------------------------------------------------
class OceanPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()


        # automatic input channel detection
        if hasattr(cfg.data, "num_input_channels"):
            in_ch = cfg.data.num_input_channels
        else:
            in_ch = (
                len(cfg.data.tracers.variables)
                + len(cfg.data.neighbors.velocity_paths)
            )

        # final fallback
        in_ch = getattr(cfg.model, "in_channels", in_ch)

        hid = cfg.model.hidden_dim
        out_ch = cfg.model.out_channels

        # ------------------------
        # Encoder (no depthwise!!)
        # ------------------------
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.GELU(),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(hid, hid * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid * 2, hid * 2, 3, padding=1),
            nn.GELU(),
        )

        # ------------------------
        # ConvLSTM core
        # ------------------------
        self.convlstm = ConvLSTMCell(input_dim=hid * 2, hidden_dim=hid * 2)

        # ------------------------
        # Decoder (smooth upsampling)
        # ------------------------
        self.dec1 = nn.Sequential(
            nn.Conv2d(hid * 2, hid, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hid, hid, 3, padding=1),
            nn.GELU(),
        )

        self.output_conv = nn.Conv2d(hid, out_ch, 1)

        # ------------------------
        # Initialization
        # ------------------------
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # -------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------
    def forward(self, x_seq): 
        # x_seq: [B, T, C, H, W]
        B, T, C, H, W = x_seq.shape

        h, c = self.convlstm.init_hidden(B, H, W)
        h = h.to(x_seq.device)
        c = c.to(x_seq.device)

        for t in range(T):
            xt = x_seq[:, t]

            e1 = self.enc1(xt)
            e2 = self.enc2(e1)

            h, (h, c) = self.convlstm(e2, (h, c))

        # decoder
        d = self.dec1(h)
        out = self.output_conv(d)
        return out

