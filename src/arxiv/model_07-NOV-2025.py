import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2LSTM(nn.Module):
    """
    CNN + LSTM hybrid for spatiotemporal 4D data (B, T, C, H, W)
    """
    def __init__(self, input_channels, hidden_dim=64, kernel_size=3, H=None, W=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.H = H
        self.W = W

        print(f"[Conv2LSTM] Using dynamic H={H}, W={W}")

        # Encoder CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Downsample by 2
        )

        # If spatial size is known during initialization, we can predefine LSTM
        if H is not None and W is not None:
            H_enc = H // 2
            W_enc = W // 2
            lstm_input_dim = hidden_dim * H_enc * W_enc
            self.lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
        else:
            # Will be initialized on first forward if not known
            self.lstm = None

        # Decoder: upsample back to original resolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, debug=False):
        B, T, C, H, W = x.shape
        encoded_seq = []

        for t in range(T):
            feats = self.encoder(x[:, t])      # (B, hidden_dim, H/2, W/2)
            feats_flat = feats.flatten(1)       # (B, hidden_dim * H_enc * W_enc)
            encoded_seq.append(feats_flat)

        encoded = torch.stack(encoded_seq, dim=1)   # (B, T, F)

        # Lazy init of LSTM if needed
        if self.lstm is None:
            self.lstm = nn.LSTM(
                input_size=encoded.size(-1),
                hidden_size=self.hidden_dim,
                batch_first=True
            ).to(x.device)

        lstm_out, _ = self.lstm(encoded)            # (B, T, hidden_dim)
        last_hidden = lstm_out[:, -1, :]            # (B, hidden_dim)

        spatial = last_hidden.view(B, self.hidden_dim, 1, 1)
        H_enc, W_enc = feats.shape[-2:]
        spatial = spatial.expand(-1, -1, H_enc, W_enc)

        out = self.decoder(spatial)                 # (B, 1, H/2, W/2)
        out = F.interpolate(out, size=(H, W), mode="bilinear")

        return out


class OceanPredictor(nn.Module):
    """Wrapper model with Conv2LSTM backend"""
    def __init__(self, cfg, H=None, W=None):
        super().__init__()
        print(f"[OceanPredictor] Using dynamic H={H}, W={W}")

        input_channels = len(cfg.data.neighbors.velocity_paths)
        if hasattr(cfg.data.tracers, "variables"):
            input_channels += len(cfg.data.tracers.variables)

        self.model = Conv2LSTM(
            input_channels=input_channels,
            hidden_dim=getattr(cfg.model, "hidden_dim", 64),
            kernel_size=getattr(cfg.model, "kernel_size", 3),
            H=H,
            W=W
        )

    def forward(self, x, debug=False):
        return self.model(x, debug=debug)

