import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2LSTM(nn.Module):
    """
    Spatiotemporal convolutional LSTM-style network for 4D data (B, T, C, H, W).

    Compared to the old version, this one:
      ✅ Keeps spatial structure (no flattening)
      ✅ Updates hidden state via convolution (no global averaging)
      ✅ Allows velocities (U,V,W) to drive local advection-like evolution
    """

    def __init__(self, input_channels, hidden_dim=64, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Encoder to extract local spatial features
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

        # ConvLSTM cell (single-layer recurrent conv)
        # Input = encoded features + hidden state
        self.conv_lstm = nn.Conv2d(hidden_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)

        # Decoder maps hidden state to final tracer output
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

    def forward(self, x, debug=False):
        """
        Forward pass
        Args:
            x: (B, T, C, H, W)
        Returns:
            out: (B, 1, H, W)
        """
        B, T, C, H, W = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, H, W, device=device)

        for t in range(T):
            # Encode current frame (spatial features)
            f_t = self.encoder(x[:, t])  # (B, hidden_dim, H, W)

            # Recurrent conv: concatenate input features + previous hidden
            combined = torch.cat([f_t, h], dim=1)
            h_new = torch.tanh(self.conv_lstm(combined))
            h = h_new  # update hidden state

            if debug and t == T - 1:
                print(f"[Conv2LSTM] Step {t}: hidden shape {h.shape}")

        out = self.decoder(h)
        return out


class OceanPredictor(nn.Module):
    """
    Wrapper around Conv2LSTM for predicting tracer evolution.
    Takes multiple input channels (e.g. tracer, U, V, W).
    """

    def __init__(self, cfg, H=None, W=None):
        super().__init__()
        self.H = H
        self.W = W

        print(f"[OceanPredictor] Using dynamic H={self.H}, W={self.W}")

        # Infer number of channels: tracer(s) + velocity components
        input_channels = len(cfg.data.neighbors.velocity_paths)
        if hasattr(cfg.data.tracers, "variables"):
            input_channels += len(cfg.data.tracers.variables)

        hidden_dim = getattr(cfg.model, "hidden_dim", 64)
        kernel_size = getattr(cfg.model, "kernel_size", 3)

        self.model = Conv2LSTM(
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )

    def forward(self, x, debug=False):
        return self.model(x, debug=debug)

